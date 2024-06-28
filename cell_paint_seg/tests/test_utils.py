import pytest
from cell_paint_seg import utils
from skimage import io
import numpy as np

@pytest.fixture
def make_im_channels(tmp_path):
    im_channels_dir = tmp_path
    tag = "fl1.tiff"
    image = np.eye(10, dtype=np.uint16)
    ids = []

    for f in range(1,3):
        for c in range(1, 7):
            id = f"r01c01f0{f}p01"
            path_tif = im_channels_dir / f"{id}-ch{c}sk1fk1{tag}"
            io.imsave(path_tif, c*image)
            ids.append(id)

    io.imsave(im_channels_dir / "redherring.tiff", image)
    
    return im_channels_dir, tag, image, ids


def test_get_id_to_path(make_im_channels):
    im_channels_dir, tag, image, ids = make_im_channels

    id_to_path = utils.get_id_to_path(im_channels_dir, tag)

    assert set(id_to_path.keys()) == set(ids)

    for paths in id_to_path.values():
        for c, path in enumerate(paths):
            assert f"ch{c+1}" in str(path)
    
