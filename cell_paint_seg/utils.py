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
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output

def eval_detections(path_dir_im, channels, path_dir_gt, path_dir_seg, out_dir, tag_gt = None, tag_seg = None, path_cp_stats_nuc = None, path_cp_stats_cell = None, cp_stat_limits = None, reg_stat_limits = None):
    out_dir = Path(out_dir)
    id_to_path_im = get_id_to_path(path_dir_im)
    id_to_path_gt = get_id_to_path(path_dir_gt, tag_gt)
    id_to_path_seg = get_id_to_path(path_dir_seg, tag_seg)

    assert len(id_to_path_gt.keys()) == len(id_to_path_seg.keys())
    assert sorted(id_to_path_gt.keys()) == sorted(id_to_path_seg.keys())

    data = []

    for id in id_to_path_gt.keys():
        save = True
        images = read_ims(id_to_path_im[id])
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

        for i, region in enumerate(regionprops_pred):
            if i%10 == 0:
                print(f"{i}/{len(regionprops_pred)}")

            sub_ims, border_masked, sub_gt_masked = get_viewing_images(images, seg_pred, seg_gt, region)


            f, axs = plt.subplots(nrows=2, ncols=7)

            for i, (sub_im,ax_col) in enumerate(zip(sub_ims, axs.T[:6])):
                ax_col[0].imshow(sub_im, cmap='gray')
                ax_col[0].set_title(f"{channels[i]}")
                ax_col[1].imshow(sub_im, cmap='gray')
                ax_col[1].set_title(f"{channels[i]}")
                ax_col[1].imshow(border_masked, cmap='autumn', alpha=0.3)
                ax_col[0].axis("off")
                ax_col[1].axis("off")
            axs[0,-1].imshow(sub_im, cmap='gray')
            axs[0,-1].imshow(sub_gt_masked, cmap='autumn', alpha=0.3)
            axs[0,-1].axis("off")

            axs[1,-1].set_title(f"{id}: {region.label} {region.bbox}")
            axs[1,-1].axis("off")


            f.set_figwidth(15)
            f.set_figheight(5)
            plt.tight_layout()

            plt.pause(0.1)
            plt.show(block=True)


            answer = input("1:good, 2:partial, 3:false pos. s:skip sample")
            if answer == "s":
                save = False
                break
            data_point = (id, region.label, answer)

            data.append(data_point)
            clear_output()

        if save:
            print(f"Saving {out_dir}/{id}.pickle")
            with open(out_dir / f"{id}.pickle", "wb") as handle:
                pickle.dump(data, handle)

    
    all_data = []
    for id in id_to_path_gt.keys():
        with open(out_dir / f"{id}.pickle", "rb") as handle:
            all_data += pickle.load(handle)

    with open(out_dir / f"all_data.pickle", "wb") as handle:
        pickle.dump(all_data, handle)


def get_viewing_images(images, seg_pred, seg_gt, region):
    viewing_radius = 5
    bbox = region.bbox

    xmin, xmax = np.amax([bbox[0]-viewing_radius, 0]), bbox[2]+viewing_radius
    ymin, ymax = np.amax([bbox[1]-viewing_radius, 0]), bbox[3]+viewing_radius

    sub_ims = [im[xmin:xmax, ymin:ymax] for im in images]
    sub_seg = seg_pred[xmin:xmax, ymin:ymax]
    sub_mask = sub_seg == region.label
    sub_gt = seg_gt[xmin:xmax, ymin:ymax]
    sub_gt_masked = np.ma.masked_array(sub_gt, sub_gt==0)

    border = ndi.binary_dilation(sub_mask==0) & sub_mask
    border = ndi.binary_dilation(border)
    border_masked = np.ma.masked_array(border, mask=border==0)

    return sub_ims, border_masked , sub_gt_masked   

def get_detection_types(path_pickle):
    with open(path_pickle, "rb") as handle:
        all_data = pickle.load(handle)

    
    data_sample = []
    data_result = []

    for data in all_data:
        sample, _, answer = data
        sample = sample[:9]
        if answer == "1":
            data_result.append("True detection")
            data_sample.append(sample)
        elif answer == "2":
            data_result.append("Partial detection")
            data_sample.append(sample)
        elif answer == "3":
            data_result.append("False positive")
            data_sample.append(sample)
    
    data = {"Sample ID": data_sample, "Detection Type": data_result}

    df = pd.DataFrame(data=data)

    # collect counts
    df = df.value_counts().reset_index()

    # collect totals
    sample_to_count = {}
    for i, row in df.iterrows():
        id = row["Sample ID"]
        count = row["count"]
        if id not in sample_to_count.keys():
            sample_to_count[id] = count
        else:
            sample_to_count[id] = sample_to_count[id] + count

    # compute proportions
    data_frac = []
    for i, row in df.iterrows():
        id = row["Sample ID"]
        count = row["count"]
        data_frac.append(count / sample_to_count[id])
    df["Proportion"] = data_frac

    # change ID name
    data_name_count = []
    for i, row in df.iterrows():
        id = row["Sample ID"]
        data_name_count.append(f"{id} ({sample_to_count[id]} total)")
    df["Sample ID"] = data_name_count

    df = df.sort_values(by=["Sample ID"])

    return df


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
    df = pd.DataFrame(data=data)
    df = df.sort_values(by=["Sample ID"])

    return df
        

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

    id_to_path = {}
    for f in files:
        if f[:12] in id_to_path.keys():
            if isinstance(id_to_path[f[:12]], list):
                id_to_path[f[:12]] = sorted(id_to_path[f[:12]] + [path_dir / f])
            else:
                id_to_path[f[:12]] = sorted([id_to_path[f[:12]]] + [path_dir / f])
        else:
            id_to_path[f[:12]] = path_dir / f

    for val in id_to_path.values():
        if isinstance(val, list):
            assert all([os.path.exists(v) for v in val])
        else:
            assert os.path.exists(val)

    return id_to_path

def read_seg(path):
    if ".tif" in path.suffix:
        return read_seg_tiff(path)
    elif ".h5" in path.suffix:
        return read_seg_hdf5(path)
    elif ".npy" in path.suffix:
        return read_seg_npy(path)
    
def read_ims(paths):
    if isinstance(paths, str):
        return read_seg(paths)
    else:
        images = [read_seg(p) for p in paths]
        return images

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


