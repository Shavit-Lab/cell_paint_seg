import pytest
import numpy as np
from skimage import io
from cell_paint_seg import image_io
import h5py
from PIL import Image


@pytest.fixture
def make_im_files(tmp_path):
    im_dir = tmp_path

    image = np.eye(10, dtype=np.uint16)

    path_tif = im_dir / "image.tif"
    io.imsave(path_tif, image)

    path_png = im_dir / "image.png"
    image_PIL = Image.fromarray(image)
    image_PIL.save(path_png)

    path_hdf5 = im_dir / "image.h5"
    with h5py.File(path_hdf5, "w") as f:
        f.create_dataset("exported_data", data=image)

    return image, path_tif, path_hdf5, path_png

@pytest.fixture
def make_multic_im_files(tmp_path):
    im_dir = tmp_path
    image = np.eye(100, dtype=np.uint16)

    id_to_paths = {}
    for id in np.arange(3):
        paths = []
        for c in np.arange(7):
            path = im_dir / f"{id}_ch{c}.tif"
            io.imsave(path, image)
            paths.append(path)
        id_to_paths[id] = paths

    return image, id_to_paths

@pytest.fixture
def make_rgb_im_files(tmp_path):
    im_dir = tmp_path
    image = np.eye(100, dtype=np.uint8)
    image = np.stack([image, image, image], axis=-1)

    id_to_paths = {}
    for id in np.arange(3):
        id_name = f"id{id}"
        paths = []
        for c in np.arange(6):
            path = im_dir / f"{id_name}_ch{c}.tif"
            io.imsave(path, image)
            paths.append(path)
        id_to_paths[id_name] = paths

    for id in id_to_paths.keys():
        for path in id_to_paths[id]:
            image = io.imread(path)
            assert np.sum(image) > 0

    return image, id_to_paths


def test_read_ims(make_im_files):
    image_true, path_tif, path_hdf5, path_png = make_im_files

    image_test = image_io.read_seg_tiff(path_tif)
    np.testing.assert_array_equal(image_true, image_test)

    image_test = image_io.read_seg_hdf5(path_hdf5)
    np.testing.assert_array_equal(image_true, image_test)

    image_test = image_io.read_seg_png(path_png)
    np.testing.assert_array_equal(image_true, image_test)

    for path in [path_tif, path_hdf5, path_png]:
        image_test = image_io.read_seg(path)
        np.testing.assert_array_equal(image_true, image_test)

        image_test = image_io.read_ims(str(path))
        np.testing.assert_array_equal(image_true, image_test)

    images_test = image_io.read_ims([path_tif, path_hdf5, path_png])
    for image_test in images_test:
        np.testing.assert_array_equal(image_true, image_test)

def test_convert(tmp_path, make_rgb_im_files):
    image, id_to_paths = make_rgb_im_files
    #assert False, id_to_paths
    image_io.convert(id_to_paths, cp_tif_dir=tmp_path, hdf5_dir=tmp_path)

    for id in id_to_paths.keys():
        with h5py.File(tmp_path / f"{id}.h5", "r") as h5:
            images = h5["image"][:]
        assert images.shape == (100, 100, 6)
        assert np.sum(images > 0) == 600

        images = io.imread(tmp_path / f"{id}.tif")
        assert images.shape == (100, 100, 2)
        assert np.sum(images > 0) == 200

def test_write_visual_jpg(tmp_path, make_multic_im_files):
    _, id_to_paths = make_multic_im_files

    out_path = tmp_path / "visual.jpg"
    in_paths = list(id_to_paths.values())[0]

    image_io.write_visual_jpg(out_path, in_paths)

