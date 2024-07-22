from pathlib import Path
import os
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess
import time
from skimage import io, measure, segmentation

from cell_paint_seg import utils, apply_ilastik, image_io


###########Inputs###########

parent_dir = "/Users/thomasathey/Documents/shavit-lab/fraenkel/first-sample/Assay Dev 20230329/BR00142688__2024-03-29T19_57_13-Measurement 1/deployment-test-small"

ilastik_path = (
    "/Applications/ilastik-1.4.0.post1-OSX.app/Contents/ilastik-release/run_ilastik.sh"
)

#################################
models_dir_path = Path(os.path.realpath(__file__)).parents[2] / "models"

bdry_pxl_path = models_dir_path / "hier-cellbody-pxl.ilp"
multicut_path = models_dir_path / "hier-cellbody-multicut.ilp"
cell_pxl_path = models_dir_path / "hier-cell-pxl.ilp"
nuc_pxl_path = models_dir_path / "hier-nucleus-pxl.ilp"

# convert to hdf5
parent_dir = Path(parent_dir)
tif_path = parent_dir / "tifs"
hdf5_path = parent_dir / "hdf5s"
output_path = parent_dir / "segmentations"

reg_stat_limits = {"area": (-1, 4000)}


time_start = time.time()

# Convert to hdf5
id_to_path = utils.get_id_to_path(tif_path, tag=".tif")
image_ids = list(id_to_path.keys())
n_files = len(image_ids)
n_channels = len(id_to_path[image_ids[0]])

im_shape = image_io.convert_to_hdf5(id_to_path, hdf5_path)

time_convert = time.time()


# run headless cell pixel classification

files = os.listdir(hdf5_path)
h5_files = [hdf5_path / f for f in files if "p01.h5" in f]

for project in tqdm(
    [bdry_pxl_path, cell_pxl_path, nuc_pxl_path],
    desc="running pixel segmentation models...",
):
    apply_ilastik.apply_ilastik_images(h5_files, ilastik_path, project)

time_pxl = time.time()


# run headless multicut
blank_seg = np.zeros(im_shape, dtype="int32")
blank_seg = Image.fromarray(blank_seg)

apply_ilastik.apply_ilastik_multicut(
    h5_files, ilastik_path, multicut_path, output_path, blank_seg
)

time_cut = time.time()

# Combine hierarchical segmentation
for h5_file in tqdm(h5_files, desc="combining segmentations"):
    im_id = h5_file.stem

    seg_soma_path = output_path / f"{im_id}-ch8sk1fk1fl1.tif"
    seg_soma_filtered = utils.path_to_filtered_seg(seg_soma_path, reg_stat_limits)
    seg_soma_path = output_path / f"{im_id}-ch8sk1fk1fl1.tif"
    io.imsave(seg_soma_path, seg_soma_filtered)

    cell_probs_path = hdf5_path / f"{im_id}_Probabilities_cell.h5"
    with h5py.File(cell_probs_path, "r") as f:
        cell_probs = np.squeeze(f["exported_data"][()])
    seg_cell = (cell_probs[:, :, 1] > 0.5).astype(np.uint8)
    seg_cell[seg_soma_filtered > 0] = 1  # somas must be contained within cells
    seg_cell_instance = utils.combine_soma_cell_labels(seg_soma_filtered, seg_cell)
    seg_cell_path = output_path / f"{im_id}-ch7sk1fk1fl1.tif"
    io.imsave(seg_cell_path, seg_cell_instance)

    nuc_probs_path = hdf5_path / f"{im_id}_Probabilities_nuc.h5"
    with h5py.File(nuc_probs_path, "r") as f:
        nuc_probs = np.squeeze(f["exported_data"][()])
    seg_nuc = (nuc_probs[:, :, 1] > 0.5).astype(np.uint8)
    seg_nuc = utils.combine_soma_nucleus_labels(seg_soma_filtered, seg_nuc)
    seg_nuc_path = output_path / f"{im_id}-ch9sk1fk1fl1.tif"
    io.imsave(seg_nuc_path, seg_nuc)


time_combine = time.time()

print(
    f"Time for {n_files} image sites w/{n_channels} channels: (Convert, {time_convert-time_start}), (Pixel pred., {time_pxl-time_convert}), (Multicut, {time_cut-time_pxl}), (Combine, {time_combine-time_cut})"
)
