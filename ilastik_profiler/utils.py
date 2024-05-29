# attributes = [gt-dir, gt-tag, seg-dir, seg-tag, results-dir]
# fxns = [false neg rate, positive breakdown]

import os
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
from skimage import measure
import pandas as pd
import scipy.ndimage as ndi
from scipy.stats import mode


def get_fn_rates(path_dir_gt, path_dir_seg, tag_gt = None, tag_seg = None, path_cp_stats_nuc = None, path_cp_stats_cell = None, cp_stat_limits = None, reg_stat_limits = None):
    id_to_path_gt = get_id_to_path(path_dir_gt, tag_gt)
    id_to_path_seg = get_id_to_path(path_dir_seg, tag_seg)

    assert len(id_to_path_gt.keys()) == len(id_to_path_seg.keys())
    assert sorted(id_to_path_gt.keys()) == sorted(id_to_path_seg.keys())

    data_id = []
    data_fns = []
    data_fnr = []
    for id in id_to_path_gt.keys():
        seg_pred = read_seg(id_to_path_seg[id])
        seg_gt = read_seg(id_to_path_gt[id])

        regionprops_pred = measure.regionprops(seg_pred)
        regionprops_gt = measure.label(seg_gt)
        regionprops_gt = measure.regionprops(regionprops_gt)

        # filter based on stats
        if path_cp_stats_nuc:
            regionprops_pred = cp_filter(id, regionprops_pred, path_cp_stats_nuc, path_cp_stats_cell, cp_stat_limits)
        if reg_stat_limits:
            regionprops_pred = reg_prop_filter(regionprops_pred, reg_stat_limits)

        # create filtered label mask
        filtered_labels = np.zeros_like(seg_pred)
        for region in regionprops_pred:
            filtered_labels[seg_pred == region.label] = region.label

        edt = ndi.distance_transform_edt(filtered_labels == 0)

        fns = 0
        for region in regionprops_gt:
            c = region.centroid
            if edt[int(c[0]), int(c[1])] > 10:
                fns += 1

        data_id.append(id)
        data_fns.append(fns)
        data_fnr.append(fns/len(regionprops_gt))

    data = {"Sample ID": data_id, "Number of Cells Missed": data_fns, "Proportion of Cells Missed": data_fnr}
    return pd.DataFrame(data=data)
        

def cp_filter(id, regionprops_pred, path_cp_stats_nuc, path_cp_stats_cell, cp_stat_limits):
    row, col, _ = row_col_field_from_id(id)
    df_cp_stats_nuc = pd.read_csv(path_cp_stats_nuc)
    df_cp_stats_cell = pd.read_csv(path_cp_stats_cell)

    valid_regions = []
    for region in regionprops_pred:
        lbl = region.label

        object_data_nuc = df_cp_stats_nuc[(df_cp_stats_nuc["Metadata_WellColumn"] == col) & (df_cp_stats_nuc["Metadata_WellRow"] == row) & (df_cp_stats_nuc["ObjectNumber"] == lbl)]

        num_cells = object_data_nuc["Children_Cells_Count"].to_numpy()[0]

        if num_cells == 1:
            object_data_cell = df_cp_stats_cell[(df_cp_stats_cell["Metadata_WellColumn"] == col) & (df_cp_stats_cell["Metadata_WellRow"] == row) & (df_cp_stats_cell["Parent_Nuclei"] == lbl)]

            valid_regions.append(region)
            for feature_name in cp_stat_limits.keys():
                if cp_stat_limits[feature_name][0] != -1 and object_data_cell[feature_name].to_numpy()[0] < cp_stat_limits[feature_name][0]:
                    del valid_regions[-1]
                    break

                if cp_stat_limits[feature_name][1] != -1 and object_data_cell[feature_name].to_numpy()[0] > cp_stat_limits[feature_name][1]:
                    del valid_regions[-1]
                    break

        elif num_cells > 1:
            raise ValueError(f"Multiple cells for nucleus {lbl}")


    return valid_regions

def reg_prop_filter(regionprops_pred, reg_stat_limits):

    valid_regions = []
    for region in regionprops_pred:
        valid_regions.append(region)

        for feature_name in reg_stat_limits.keys():
            if reg_stat_limits[feature_name][0] != -1 and region[feature_name] < reg_stat_limits[feature_name][0]:
                del valid_regions[-1]
                break

            if reg_stat_limits[feature_name][1] != -1 and region[feature_name] > reg_stat_limits[feature_name][1]:
                del valid_regions[-1]
                break

        
    return valid_regions



def row_col_field_from_id(id):
    row, col, field = int(id[1:3]), int(id[4:6]), int(id[7:9])
    return row, col, field


def get_id_to_path(path_dir, tag = None):
    path_dir = Path(path_dir)

    files = os.listdir(path_dir)

    if tag:
        files = [f for f in files if tag in f]

    id_to_path = {f[:12]: path_dir / f for f in files}

    for val in id_to_path.values():
        assert os.path.exists(val)

    return id_to_path

def read_seg(path):
    if ".tif" in path.suffix:
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
    with h5py.File(path_hdf5, 'r') as f:
        image = np.squeeze(f["exported_data"][()])

    bg_lbl = mode(image.flatten()).mode

    if bg_lbl != 0:
        assert np.sum(image == 0) == 0
        image[image == bg_lbl] = 0

    return image

def read_seg_npy(path_npy):
    image = np.load(path_npy, allow_pickle=True).item()["masks"]
    return image


