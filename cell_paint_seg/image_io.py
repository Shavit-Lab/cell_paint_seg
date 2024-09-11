from PIL import Image
from pathlib import Path
import os
import numpy as np
import h5py
from scipy.stats import mode
from tqdm import tqdm
from skimage import exposure, io


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
    elif ".png" in path.suffix:
        return read_seg_png(path)
    else:
        raise ValueError("File type not supported.")


def read_seg_tiff(path_tif):
    image = Image.open(path_tif)
    image = np.array(image)

    return image


def read_seg_png(path_png):
    image = Image.open(path_png)
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


def normalize(images):
    ims = [im.astype("float64") for im in images]
    ims = [np.linalg.norm(im, axis=-1) for im in ims]
    ims = [im / np.amax(im) for im in ims]
    return ims


def preprocess_images(images):
    ims = [exposure.equalize_adapthist(im, clip_limit=0.03) for im in images]
    ims = [im * (2**16 - 1) for im in ims]
    ims = [im.astype("uint16") for im in ims]
    return ims


def convert(
    id_to_path,
    cp_tif_dir=None,
    hdf5_dir=None,
    order=[0, 1, 2, 3, 4, 5],
    preprocess=False,
):
    hdf5_dir = Path(hdf5_dir)
    for image_id, image_paths in tqdm(id_to_path.items(), "converting..."):
        images = read_ims(image_paths)
        channel_shape = images[0].shape

        images = [images[i] if i != -1 else np.zeros(channel_shape) for i in order]
        images = normalize(images)

        if cp_tif_dir is not None:
            ims_2channel = [images[i] * (2**16 - 1) for i in [5, 4]]
            ims_2channel = [im.astype("uint16") for im in ims_2channel]
            ims_2channel = np.stack(ims_2channel, axis=-1)

            fname = cp_tif_dir / f"{image_id}.tif"
            io.imsave(fname, ims_2channel)

        if hdf5_dir is not None:
            if preprocess:
                images = preprocess_images(images)

            channel_shape = images[0].shape
            images = np.stack(images, axis=2)
            with h5py.File(hdf5_dir / f"{image_id}.h5", "a") as h5:
                h5.create_dataset(f"image", data=images)

    return channel_shape


def write_visual_jpg(out_path, in_paths, rgb_tags=["ch3", "ch6", "ch5"]):
    image_RGB = []
    for n_ch, rgb_tag in enumerate(rgb_tags):
        for path in in_paths:
            path = str(path)
            if rgb_tag in path:
                image = read_seg(path)
                if n_ch < 2:
                    image = exposure.equalize_adapthist(image, clip_limit=0.03)
                else:
                    image = exposure.equalize_adapthist(
                        image,
                        clip_limit=0.03,
                        kernel_size=[s // 64 for s in image.shape],
                    )
                image /= np.amax(image)
                image *= 255
                image = image.astype("uint8")
                image_RGB.append(image)

    image_RGB = np.stack(image_RGB, axis=2)

    image = Image.fromarray(image_RGB.astype("uint8")).convert("RGB")
    image.save(out_path)
