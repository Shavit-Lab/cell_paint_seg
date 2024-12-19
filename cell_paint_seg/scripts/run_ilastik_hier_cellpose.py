from pathlib import Path
import os
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from skimage import io
import argparse

from cell_paint_seg import utils, apply_ilastik, apply_cpose, image_io


#############Outputs###############
# hdf5_path:
#   - *.h5 - 6 channel images
#   - *_Object Predictions.h5 - predictions of whether somas are alive are dead
#   - *_Probabilities_cell.h5 - pixel predictions of whether pixels are part of a cell
# twochan_oath:
#   - *.tif - 2 channel images
#   - *.npy - segmented objects from cellpsoe
#   - *_cp_masks.tif - cellpose segmentation
# output_path:
#   - *.h5 - soma segmentation
#   - *.tif - segmentation of objects with channels:
#       - 7,10,13 - all,alive,dead cells respectively
#       - 8,11,14 - all,alive,dead somas respectively
#       - 9,12,15 - all,alive,dead nuclei respectively
################################


def main():
    parser = argparse.ArgumentParser(
        description="Perform hierarchical segmentation on cell painting images"
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="C:\\Users\\zeiss\\projects\\athey_als\\test-images-96",
        help="Path to parent directory",
    )
    parser.add_argument(
        "--ilastik_path",
        type=str,
        default="C:\\Program Files\\ilastik-1.4.0.post1\\ilastik.exe",
        help="Path to ilastik executable",
    )
    parser.add_argument(
        "--channel_order",
        type=list,
        default=[-1, 0, 3, 2, 1, 4],
        help="Channel indexes for brightfield, ER, Actin, Mito, DNA, RNA",
    )
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir)
    ilastik_path = args.ilastik_path
    order = args.channel_order

    no_cells_any = []
    no_cells_alive = []

    print(
        f"Running hierarchical segmentation on cell painting images in directory {parent_dir}"
    )

    # Models paths
    cell_pxl_path, nuc_pxl_path, obj_class_path = get_model_paths()

    # create subfolders if they don't exist
    tif_path, hdf5_path, twochan_path, output_path = get_create_subfolders(parent_dir)

    reg_stat_limits = {"area": (-1, 4000)}

    time_start = time.time()
    # Convert to hdf5
    id_to_path = utils.get_id_to_path(
        tif_path, tag=".tif", id_from_name=get_id_from_name
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

    # Headless ilastik cell pixel classification
    files = os.listdir(hdf5_path)
    h5_files = [hdf5_path / f for f in files if ".h5" in f]

    for project in tqdm(
        [cell_pxl_path],
        desc="running pixel segmentation models...",
    ):
        apply_ilastik.apply_ilastik_images(h5_files, ilastik_path, project)
    time_pxl = time.time()

    # Cellpose soma segmentation
    blank_seg = np.zeros(im_shape, dtype="int32")
    blank_seg = Image.fromarray(blank_seg)

    apply_cpose.apply_cpose(
        twochan_path, output_path, id_from_name=get_id_from_name, nuclei=True
    )
    time_cp = time.time()

    # Combine hierarchical segmentation
    for h5_file in tqdm(h5_files, desc="combining segmentations"):
        im_id = h5_file.stem

        seg_soma_path = output_path / f"{im_id}c8.tif"
        seg_soma_filtered = utils.path_to_filtered_seg(seg_soma_path, reg_stat_limits)
        if np.sum(seg_soma_filtered) == 0:
            no_cells_any.append(im_id)
        seg_soma_filtered = seg_soma_filtered.astype(np.uint32)
        io.imsave(seg_soma_path, seg_soma_filtered)
        seg_soma_path_h5 = output_path / f"{im_id}c8.h5"
        with h5py.File(seg_soma_path_h5, "a") as h5:
            h5.create_dataset("segmentation", data=seg_soma_filtered)

        cell_probs_path = hdf5_path / f"{im_id}_Probabilities_cell.h5"
        with h5py.File(cell_probs_path, "r") as f:
            cell_probs = np.squeeze(f["exported_data"][()])
        seg_cell = (cell_probs[:, :, 1] > 0.5).astype(np.uint8)
        seg_cell[seg_soma_filtered > 0] = 1  # add somas to cell segmentation
        seg_cell_instance = utils.combine_soma_cell_labels(seg_soma_filtered, seg_cell)
        seg_cell_path = output_path / f"{im_id}c7.tif"
        io.imsave(seg_cell_path, seg_cell_instance)

        nuc_probs_path = twochan_path / f"{im_id}_cp_masks.tif"
        nuc_probs = io.imread(nuc_probs_path)
        seg_nuc = (nuc_probs[:, :] > 0.5).astype(np.uint8)
        seg_nuc = utils.combine_soma_nucleus_labels(seg_soma_filtered, seg_nuc)
        seg_nuc_path = output_path / f"{im_id}c9.tif"
        io.imsave(seg_nuc_path, seg_nuc)
    time_combine = time.time()

    # Run Object Classification
    apply_ilastik.apply_ilastik_obj_class(
        h5_files,
        output_path,
        ilastik_path,
        obj_class_path,
        id_from_name=get_id_from_name,
    )
    time_obj_class = time.time()

    # Filter alive/dead cells
    id_to_path_obj = utils.get_id_to_path(
        hdf5_path, tag="Object", id_from_name=get_id_from_name
    )
    id_to_path_seg = utils.get_id_to_path(
        output_path, tag=".tif", id_from_name=get_id_from_name
    )

    for id in id_to_path_seg.keys():
        with h5py.File(id_to_path_obj[id], "r") as f:
            ctypes = np.squeeze(f["exported_data"][()])

        seg_soma = io.imread(id_to_path_seg[id][1])

        # save alive somas
        seg_soma_class = np.copy(seg_soma)
        seg_soma_class[ctypes != 1] = 0
        if np.sum(seg_soma_class) == 0:
            no_cells_alive.append(id)
        io.imsave(output_path / f"{id}c11.tif", seg_soma_class)
        alive_ids = np.unique(seg_soma_class)

        # save dead somas
        seg_soma_class = np.copy(seg_soma)
        seg_soma_class[ctypes != 2] = 0
        io.imsave(output_path / f"{id}c14.tif", seg_soma_class)
        dead_ids = np.unique(seg_soma_class)

        for seg_channel in [0, 2]:
            seg = io.imread(id_to_path_seg[id][seg_channel])

            for i_ctype, ctype_list in enumerate([alive_ids, dead_ids]):
                seg_class = np.copy(seg)
                for lbl in np.unique(seg):
                    if lbl not in ctype_list:
                        seg_class[seg_class == lbl] = 0
                io.imsave(
                    output_path / f"{id}c{10+i_ctype*3+seg_channel}.tif",
                    seg_class,
                )
    time_filter_dead = time.time()

    print(
        f"Time for {n_files} image sites w/{n_channels} channels: "
        + f"(Convert, {time_convert-time_start}), (Pixel pred., {time_pxl-time_convert}), "
        + f"(Cellpose, {time_cp-time_pxl}), (Combine, {time_combine-time_cp}), "
        + f"(Obj. class., {time_obj_class-time_combine}), (Filter dead, {time_filter_dead-time_obj_class}), "
        + f"Total: {time_filter_dead-time_start}"
    )

    print(
        f"No cells detected in {len(no_cells_any)}/{len(h5_files)} images: {no_cells_any}"
    )
    print(
        f"No alive cells detected in {len(no_cells_alive)}/{len(list(id_to_path_seg.keys()))} images: {no_cells_alive}"
    )


def get_id_from_name(name):
    id = name[:48]
    return id


def get_model_paths():
    models_dir_path = Path(os.path.realpath(__file__)).parents[2] / "models"
    cell_pxl_path = models_dir_path / "hier-cell-pxl.ilp"
    nuc_pxl_path = models_dir_path / "hier-nucleus-pxl.ilp"
    obj_class_path = models_dir_path / "celltype.ilp"
    return cell_pxl_path, nuc_pxl_path, obj_class_path


def get_create_subfolders(parent_dir):
    tif_path = parent_dir / "tifs"  # 4 - change number
    hdf5_path = parent_dir / "tommy" / "hdf5s"
    twochan_path = parent_dir / "tommy" / "twochannel_cpose"
    output_path = parent_dir / "tommy" / "segmentations"

    for path in [hdf5_path, twochan_path, output_path]:
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)

    return tif_path, hdf5_path, twochan_path, output_path


if __name__ == "__main__":
    main()
