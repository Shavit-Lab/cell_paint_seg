from PIL import Image
from pathlib import Path
import os
import numpy as np
import h5py
from scipy.stats import mode
from tqdm import tqdm


def read_ims(paths, sftp_client=None):
    """Read image path(s)

    Args:
        paths (str or list): Image paths.

    Returns:
        nd.array or list: Image arrays.
    """
    if isinstance(paths, str):
        return read_seg(paths, sftp_client)
    else:
        images = [read_seg(p, sftp_client) for p in paths]
        return images


def read_seg(path, sftp_client=None):
    """Read image/segmentation file. Supports .tif, .h5, and .npy

    Args:
        path (str): Image/segmentation path.

    Returns:
        nd.array: Image array.
    """
    path = Path(path)
    if ".tif" in path.suffix:
        if sftp_client:
            local_fname = (
                Path(
                    "/Users/thomasathey/Documents/shavit-lab/fraenkel/cell_paint_seg/data/temp"
                )
                / path.name
            )
            sftp_client.get(str(path), str(local_fname))
            im = read_seg_tiff(local_fname)
            os.remove(local_fname)
            return im
        else:
            return read_seg_tiff(path)
    elif ".h5" in path.suffix:
        return read_seg_hdf5(path)
    elif ".npy" in path.suffix:
        return read_seg_npy(path)


def read_seg_tiff(path_tif):
    image = Image.open(path_tif)
    image = np.array(image)

    return image


def read_seg_hdf5(path_hdf5):
    with h5py.File(path_hdf5, "r") as f:
        image = np.squeeze(f["exported_data"][()])

    bg_lbl = mode(image.flatten()).mode

    if bg_lbl != 0:
        assert np.sum(image == 0) == 0
        image[image == bg_lbl] = 0

    return image


def read_seg_npy(path_npy):
    image = np.load(path_npy, allow_pickle=True).item()["masks"]
    return image


def convert_to_hdf5(id_to_path, hdf5_dir):
    hdf5_dir = Path(hdf5_dir)
    for image_id, image_paths in tqdm(id_to_path.items(), "converting to hdf5..."):
        images = read_ims(image_paths)
        channel_shape = images[0].shape
        images = np.stack(images, axis=2)
        with h5py.File(hdf5_dir / f"{image_id}.h5", "a") as h5:
            h5.create_dataset(f"image", data=images)

    return channel_shape
