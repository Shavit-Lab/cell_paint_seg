from pathlib import Path
import os
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess
import time
from skimage import io, measure, segmentation

from cell_paint_seg import utils, apply_ilastik, apply_cpose, image_io


###########Inputs###########

parent_dir = "/Users/thomasathey/Documents/shavit-lab/fraenkel/96_well/exp2/train_set/"

ilastik_path = (
    "/Applications/ilastik-1.4.0.post1-OSX.app/Contents/ilastik-release/run_ilastik.sh"
)

order = [-1, 0, 3, 2, 1, 4]

#################################
models_dir_path = Path(os.path.realpath(__file__)).parents[2] / "models"

cell_pxl_path = models_dir_path / "hier-cell-pxl.ilp"
nuc_pxl_path = models_dir_path / "hier-nucleus-pxl.ilp"
obj_class_path = models_dir_path / "celltype.ilp"

# convert to hdf5
parent_dir = Path(parent_dir)
tif_path = parent_dir / "tifs"
hdf5_path = parent_dir / "hdf5s"
twochan_path = parent_dir / "twochannel_cpose"
output_path = parent_dir / "segmentations"

reg_stat_limits = {"area": (-1, 4000)}


time_start = time.time()

# Convert to hdf5
id_to_path = utils.get_id_to_path(
    tif_path, tag=".tif", id_from_name=utils.get_id_from_name_96
)
image_ids = list(id_to_path.keys())
n_files = len(image_ids)
n_channels = len(id_to_path[image_ids[0]])

im_shape = image_io.convert(
    id_to_path,
    cp_tif_dir=twochan_path,
    hdf5_dir=hdf5_path,
    order=order,
    preprocess=True,
)

time_convert = time.time()


# run headless cell pixel classification

files = os.listdir(hdf5_path)
h5_files = [hdf5_path / f for f in files if ".h5" in f]

for project in tqdm(
    [cell_pxl_path],
    desc="running pixel segmentation models...",
):
    apply_ilastik.apply_ilastik_images(h5_files, ilastik_path, project)

time_pxl = time.time()


# run cellpose soma segmentation
blank_seg = np.zeros(im_shape, dtype="int32")
blank_seg = Image.fromarray(blank_seg)

apply_cpose.apply_cpose(twochan_path, output_path, nuclei=True)

time_cp = time.time()


# Combine hierarchical segmentation
for h5_file in tqdm(h5_files, desc="combining segmentations"):
    im_id = h5_file.stem

    seg_soma_path = output_path / f"{im_id}-ch8sk1fk1fl1.tif"
    seg_soma_filtered = utils.path_to_filtered_seg(seg_soma_path, reg_stat_limits)
    seg_soma_path = output_path / f"{im_id}-ch8sk1fk1fl1.tif"
    seg_soma_filtered = seg_soma_filtered.astype(np.uint32)
    io.imsave(seg_soma_path, seg_soma_filtered)

    cell_probs_path = hdf5_path / f"{im_id}_Probabilities_cell.h5"
    with h5py.File(cell_probs_path, "r") as f:
        cell_probs = np.squeeze(f["exported_data"][()])
    seg_cell = (cell_probs[:, :, 1] > 0.5).astype(np.uint8)
    seg_cell[seg_soma_filtered > 0] = 1  # somas must be contained within cells
    seg_cell_instance = utils.combine_soma_cell_labels(seg_soma_filtered, seg_cell)
    seg_cell_path = output_path / f"{im_id}-ch7sk1fk1fl1.tif"
    io.imsave(seg_cell_path, seg_cell_instance)

    nuc_probs_path = twochan_path / f"{im_id}_cp_masks.tif"
    nuc_probs = io.imread(nuc_probs_path)
    seg_nuc = (nuc_probs[:, :] > 0.5).astype(np.uint8)
    seg_nuc = utils.combine_soma_nucleus_labels(seg_soma_filtered, seg_nuc)
    seg_nuc_path = output_path / f"{im_id}-ch9sk1fk1fl1.tif"
    io.imsave(seg_nuc_path, seg_nuc)


time_combine = time.time()

# Run Object Classification
apply_ilastik.apply_ilastik_obj_class(
    h5_files, output_path, ilastik_path, obj_class_path
)
time_obj_class = time.time()

# Filter alive/dead cells
id_to_path_obj = utils.get_id_to_path(
    hdf5_path, tag="Object", id_from_name=utils.get_id_from_name_first_us
)
id_to_path_seg = utils.get_id_to_path(
    output_path, tag=".tif", id_from_name=utils.get_id_from_name_first_hyph
)

for id in id_to_path_seg.keys():
    with h5py.File(id_to_path_obj[id], "r") as f:
        ctypes = np.squeeze(f["exported_data"][()])

    seg_soma = io.imread(id_to_path_seg[id][1])

    # save alive somas
    seg_soma_class = np.copy(seg_soma)
    seg_soma_class[ctypes != 1] = 0
    io.imsave(output_path / f"{id}-ch11sk1fk1fl1.tif", seg_soma_class)
    alive_ids = np.unique(seg_soma_class)

    # save dead somas
    seg_soma_class = np.copy(seg_soma)
    seg_soma_class[ctypes != 2] = 0
    io.imsave(output_path / f"{id}-ch14sk1fk1fl1.tif", seg_soma_class)
    dead_ids = np.unique(seg_soma_class)

    for seg_channel in [0, 2]:
        seg = io.imread(id_to_path_seg[id][seg_channel])

        for i_ctype, ctype_list in enumerate([alive_ids, dead_ids]):
            seg = np.copy(seg)
            for lbl in np.unique(seg):
                if lbl not in ctype_list:
                    seg[seg == lbl] = 0
            io.imsave(
                output_path / f"{id}-ch{10+i_ctype*3+seg_channel}sk1fk1fl1.tif", seg
            )

time_filter_dead = time.time()

print(
    f"Time for {n_files} image sites w/{n_channels} channels: "
    + f"(Convert, {time_convert-time_start}), (Pixel pred., {time_pxl-time_convert}), "
    + f"(Cellpose, {time_cp-time_pxl}), (Combine, {time_combine-time_cp}), "
    + f"(Obj. class., {time_obj_class-time_combine}), (Filter dead, {time_filter_dead-time_obj_class}), "
    + f"Total: {time_filter_dead-time_start}"
)
