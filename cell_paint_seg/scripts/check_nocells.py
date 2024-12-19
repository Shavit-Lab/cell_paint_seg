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
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir)
    get_id_from_name = utils.get_id_from_name_trailing_c

    no_cells_any = []
    no_cells_alive = []

    tif_path, hdf5_path, twochan_path, output_path = get_create_subfolders(parent_dir)
    id_to_path = utils.get_id_to_path(
        output_path, tag=".tif", id_from_name=get_id_from_name
    )

    n_samples = len(list(id_to_path.keys()))
    for id, paths in id_to_path.items():

        ims = image_io.read_ims(paths)

        for im, path in zip(ims, paths):
            if "c8" in path.name and np.sum(im) == 0:
                no_cells_any.append(id)
            elif "c11" in path.name and np.sum(im) == 0:
                no_cells_alive.append(id)

    print(
        f"No cells detected in {len(no_cells_any)}/{n_samples} images: {no_cells_any}"
    )
    print(
        f"No alive cells detected in {len(no_cells_alive)}/{n_samples} images: {no_cells_alive}"
    )


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
