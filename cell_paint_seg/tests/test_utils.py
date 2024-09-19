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

    for f in range(1, 3):
        for c in range(1, 7):
            id = f"r01c01f0{f}p01"
            path_tif = im_channels_dir / f"{id}-ch{c}sk1fk1{tag}"
            io.imsave(path_tif, c * image)
            ids.append(id)

    io.imsave(im_channels_dir / "redherring.tiff", image)

    def get_id_from_name(name):
        id = name.split("-")[0]
        return id

    return im_channels_dir, tag, image, ids, get_id_from_name


def test_get_id_to_path(make_im_channels):
    im_channels_dir, tag, image, ids, get_id_from_name = make_im_channels

    id_to_path = utils.get_id_to_path(
        im_channels_dir, id_from_name=get_id_from_name, tag=tag
    )

    assert set(id_to_path.keys()) == set(ids)

    for paths in id_to_path.values():
        for c, path in enumerate(paths):
            assert f"ch{c+1}" in str(path)


def test_combine_soma_nucleus_labels():
    # nuclei are restricted to somas
    seg_soma = np.zeros((10, 10))
    seg_nuc = np.zeros((10, 10))
    seg_soma[:3, :3] = 1
    seg_nuc[2:4, 2:4] = 1
    seg_nuc_filtered_true = np.zeros((10, 10))
    seg_nuc_filtered_true[2, 2] = 1
    seg_nuc_filtered = utils.combine_soma_nucleus_labels(seg_soma, seg_nuc)
    assert np.array_equal(seg_nuc_filtered, seg_nuc_filtered_true)

    # Nuclei have same label as surrounding soma
    seg_soma = np.zeros((10, 10))
    seg_nuc = np.zeros((10, 10))
    seg_soma[4, 4] = 2
    seg_nuc[4, 4] = 1
    seg_nuc_filtered_true = seg_soma.copy()
    seg_nuc_filtered = utils.combine_soma_nucleus_labels(seg_soma, seg_nuc)
    assert np.array_equal(seg_nuc_filtered, seg_nuc_filtered_true)

    # largest connected component is chosen as nucleus
    seg_soma = np.zeros((10, 10))
    seg_nuc = np.zeros((10, 10))
    seg_soma[:5, :5] = 1
    seg_nuc[:3, :3] = 1
    seg_nuc[4, 4] = 1
    seg_nuc_filtered_true = np.zeros((10, 10))
    seg_nuc_filtered_true[:3, :3] = 1
    seg_nuc_filtered = utils.combine_soma_nucleus_labels(seg_soma, seg_nuc)
    assert np.array_equal(seg_nuc_filtered, seg_nuc_filtered_true)

    # multiple cells
    seg_soma = np.zeros((10, 10))
    seg_nuc = np.zeros((10, 10))
    seg_soma[:5, :5] = 1
    seg_soma[8:, 8:] = 2
    seg_nuc = np.zeros((10, 10))
    seg_nuc[:3, :3] = 1
    seg_nuc[4, 4] = 1
    seg_nuc[7:, 7:] = 1
    seg_nuc_filtered_true = np.zeros((10, 10))
    seg_nuc_filtered_true[:3, :3] = 1
    seg_nuc_filtered_true[8:, 8:] = 2
    seg_nuc_filtered = utils.combine_soma_nucleus_labels(seg_soma, seg_nuc)
    assert np.array_equal(seg_nuc_filtered, seg_nuc_filtered_true)


def test_combine_soma_cell_labels():
    seg_soma = np.zeros((10, 10), dtype=np.int32)
    seg_cell = np.zeros((10, 10))

    # cells are connected to somas
    seg_soma[:3, :3] = 1
    seg_cell[2:4, 2:4] = 1
    seg_cell[8, 8] = 1
    seg_cell_filtered_true = np.zeros((10, 10))
    seg_cell_filtered_true[:3, :3] = 1
    seg_cell_filtered_true[2:4, 2:4] = 1
    seg_cell_filtered = utils.combine_soma_cell_labels(seg_soma, seg_cell)
    assert np.array_equal(seg_cell_filtered, seg_cell_filtered_true)

    # cell is assigned to closest soma
    seg_soma = np.zeros((10, 10), dtype=np.int32)
    seg_soma[:3, 0] = 1
    seg_soma[5:6, 0] = 2
    seg_cell = np.zeros((10, 10))
    seg_cell[:, 0] = 1
    seg_cell_filtered_true = np.zeros((10, 10))
    seg_cell_filtered_true[:4, 0] = 1
    seg_cell_filtered_true[4:, 0] = 2
    seg_cell_filtered = utils.combine_soma_cell_labels(seg_soma, seg_cell)
    assert np.array_equal(seg_cell_filtered, seg_cell_filtered_true)

    # cell contains soma
    seg_soma = np.zeros((10, 10), dtype=np.int32)
    seg_soma[:3, 0] = 1
    seg_soma[5:6, 0] = 2
    seg_cell = np.zeros((10, 10))
    seg_cell[2:, 0] = 1
    seg_cell_filtered_true = np.zeros((10, 10))
    seg_cell_filtered_true[:4, 0] = 1
    seg_cell_filtered_true[4:, 0] = 2
    seg_cell_filtered = utils.combine_soma_cell_labels(seg_soma, seg_cell)
    assert np.array_equal(seg_cell_filtered, seg_cell_filtered_true)

    # multiple cells
    seg_soma[0, 5] = 3
    seg_cell[:, 8] = 1
    seg_cell_filtered_true[0, 5] = 3
    seg_cell_filtered = utils.combine_soma_cell_labels(seg_soma, seg_cell)
    assert np.array_equal(seg_cell_filtered, seg_cell_filtered_true)


def test_get_id_from_name():
    id = "s012"
    assert utils.get_id_from_name_96(f"junk_{id}junk") == id


def test_check_valid_labels_comp():
    seg = np.zeros((10, 10))

    seg[0, 0] = 1
    seg[5, 5] = 2
    assert utils.check_valid_labels_comp(seg)

    # majority is not background
    with pytest.raises(Exception):
        utils.check_valid_labels_comp(np.ones((10, 10)))

    # labels not consecutive
    seg[0, 0] = 0
    with pytest.raises(Exception):
        utils.check_valid_labels_comp(seg)

    # one of the labels is not connected
    seg[0, 0] = 1
    seg[9, 9] = 1
    assert utils.check_valid_labels_comp(seg) == False
