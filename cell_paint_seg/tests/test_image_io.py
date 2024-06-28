import pytest
import numpy as np
from skimage import io
from cell_paint_seg import image_io
import h5py


@pytest.fixture
def make_im_files(tmp_path):
    im_dir = tmp_path

    image = np.eye(10, dtype=np.uint16)

    path_tif = im_dir / "image.tif"
    io.imsave(path_tif, image)

    path_hdf5 = im_dir / "image.h5"
    with h5py.File(path_hdf5, "w") as f:
        f.create_dataset("exported_data", data=image)

    return image, path_tif, path_hdf5


def test_read_ims(make_im_files):
    image_true, path_tif, path_hdf5 = make_im_files

    image_test = image_io.read_seg_tiff(path_tif)
    np.testing.assert_array_equal(image_true, image_test)

    image_test = image_io.read_seg_hdf5(path_hdf5)
    np.testing.assert_array_equal(image_true, image_test)

    for path in [path_tif, path_hdf5]:
        image_test = image_io.read_seg(path)
        np.testing.assert_array_equal(image_true, image_test)

        image_test = image_io.read_ims(str(path))
        np.testing.assert_array_equal(image_true, image_test)

    images_test = image_io.read_ims([path_tif, path_hdf5])
    for image_test in images_test:
        np.testing.assert_array_equal(image_true, image_test)
