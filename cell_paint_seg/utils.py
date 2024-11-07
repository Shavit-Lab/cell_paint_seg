# attributes = [gt-dir, gt-tag, seg-dir, seg-tag, results-dir]
# fxns = [false neg rate, positive breakdown]

import os
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
from skimage import measure, segmentation, exposure
import pandas as pd
import scipy.ndimage as ndi
from scipy.stats import mode
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
import paramiko
from cell_paint_seg.image_io import read_ims, read_seg
import warnings


def eval_detections(
    path_dir_im,
    channels,
    path_dir_gt,
    path_dir_seg,
    out_dir,
    tag_gt=None,
    tag_seg=None,
    path_cp_stats_nuc=None,
    path_cp_stats_cell=None,
    cp_stat_limits=None,
    reg_stat_limits=None,
):
    """Display predicted cell segmentations and ask for user classification to collect detection type data.

    Args:
        path_dir_im (str): Path to images
        channels (list): List of channel names
        path_dir_gt (str): Path to ground truth detections
        path_dir_seg (str): Path to predicted segmentation
        out_dir (str): Path to save detection type responses
        tag_gt (str, optional): Substring to identify ground truth detection files. Defaults to None.
        tag_seg (str, optional): Substring to identify predicted segmentation files. Defaults to None.
        path_cp_stats_nuc (str, optional): Path to nucleus CellProfiler statistics if CellProfiler filtering is desired. Defaults to None.
        path_cp_stats_cell (str, optional): Path to cell CellProfiler statistics if CellProfiler filtering is desired. Defaults to None.
        cp_stat_limits (dict, optional): CellProfiler statistic names and limits (inclusive) if CellProfiler filtering is desired. Defaults to None.
        reg_stat_limits (dict, optional): RegionProperties statistic names and limits (inclusive) if RegionProperties filtering is desired. Defaults to None.
    """
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
            regionprops_pred = cp_filter(
                id,
                regionprops_pred,
                path_cp_stats_nuc,
                path_cp_stats_cell,
                cp_stat_limits,
            )
        if reg_stat_limits:
            regionprops_pred = reg_prop_filter(regionprops_pred, reg_stat_limits)

        for i, region in enumerate(regionprops_pred):
            if i % 10 == 0:
                print(f"{i}/{len(regionprops_pred)}")

            sub_ims, border_masked, sub_gt_masked = get_viewing_images(
                images, seg_pred, seg_gt, region
            )

            f, axs = plt.subplots(nrows=2, ncols=7)

            for i, (sub_im, ax_col) in enumerate(zip(sub_ims, axs.T[:6])):
                ax_col[0].imshow(sub_im, cmap="gray")
                ax_col[0].set_title(f"{channels[i]}")
                ax_col[1].imshow(sub_im, cmap="gray")
                ax_col[1].set_title(f"{channels[i]}")
                ax_col[1].imshow(border_masked, cmap="autumn", alpha=0.3)
                ax_col[0].axis("off")
                ax_col[1].axis("off")
            axs[0, -1].imshow(sub_im, cmap="gray")
            axs[0, -1].imshow(sub_gt_masked, cmap="autumn", alpha=0.3)
            axs[0, -1].axis("off")

            axs[1, -1].set_title(f"{id}: {region.label} {region.bbox}")
            axs[1, -1].axis("off")

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

    # Collect responses for all samples
    all_data = []
    for id in id_to_path_gt.keys():
        with open(out_dir / f"{id}.pickle", "rb") as handle:
            all_data += pickle.load(handle)

    # Save collected responses
    with open(out_dir / f"all_data.pickle", "wb") as handle:
        pickle.dump(all_data, handle)


def get_viewing_images(images, seg_pred, seg_gt, region):
    viewing_radius = 5
    bbox = region.bbox

    xmin, xmax = np.amax([bbox[0] - viewing_radius, 0]), bbox[2] + viewing_radius
    ymin, ymax = np.amax([bbox[1] - viewing_radius, 0]), bbox[3] + viewing_radius

    sub_ims = [im[xmin:xmax, ymin:ymax] for im in images]
    sub_seg = seg_pred[xmin:xmax, ymin:ymax]
    sub_mask = sub_seg == region.label
    sub_gt = seg_gt[xmin:xmax, ymin:ymax]
    sub_gt_masked = np.ma.masked_array(sub_gt, sub_gt == 0)

    border = ndi.binary_dilation(sub_mask == 0) & sub_mask
    border = ndi.binary_dilation(border)
    border_masked = np.ma.masked_array(border, mask=border == 0)

    return sub_ims, border_masked, sub_gt_masked


def get_detection_types(path_pickle):
    """Read and organize detection type data into a dataframe

    Args:
        path_pickle (str): path of pickle files with (ground truth) labels predicted detections

    Returns:
        pd.DataFrame: dataframe with detection type data across different image IDs.
    """
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


def get_fn_rates(
    path_dir_gt,
    path_dir_seg,
    tag_gt=None,
    tag_seg=None,
    path_cp_stats_nuc=None,
    path_cp_stats_cell=None,
    cp_stat_limits=None,
    reg_stat_limits=None,
):
    """Get false negative (missed cells) data for the results of a segmentation algorithm.

    Args:
        path_dir_gt (str): Path where ground truth cell detections are. Files are binary masks where each foreground connected component is near the center of a true cell instance.
        path_dir_seg (str): Path where predicted segmentations are.
        tag_gt (str, optional): Substring that identifies ground truth files in path_dir_gt. Defaults to None.
        tag_seg (str, optional): Substring that identifies desired segmentation predictions in path_dir_seg. Defaults to None.
        path_cp_stats_nuc (str, optional): Path to nucleus CellProfiler statistics if CellProfiler filtering is desired. Defaults to None.
        path_cp_stats_cell (str, optional): Path to cell CellProfiler statistics if CellProfiler filtering is desired. Defaults to None.
        cp_stat_limits (dict, optional): CellProfiler statistic names and limits (inclusive) if CellProfiler filtering is desired. Defaults to None.
        reg_stat_limits (dict, optional): RegionProperties statistic names and limits (inclusive) if RegionProperties filtering is desired. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe which contains false negative counts and rates for each image ID.
    """
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
            regionprops_pred = cp_filter(
                id,
                regionprops_pred,
                path_cp_stats_nuc,
                path_cp_stats_cell,
                cp_stat_limits,
            )
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
        data_fnr.append(fns / len(regionprops_gt))

    data = {
        "Sample ID": data_id,
        "Number of Cells Missed": data_fns,
        "Proportion of Cells Missed": data_fnr,
    }
    df = pd.DataFrame(data=data)
    df = df.sort_values(by=["Sample ID"])

    return df


def cp_filter(
    id, regionprops_pred, path_cp_stats_nuc, path_cp_stats_cell, cp_stat_limits
):
    """Filter objects based on limits of specified CellProfiler statistics.

    Args:
        id (str): Image ID
        regionprops_pred (list): List of unfiltered RegionProperties
        path_cp_stats_nuc (str): Path to nucleus CellProfiler statistics table
        path_cp_stats_cell (str): Path to cell CellProfiler statistics table
        cp_stat_limits (dict): Specified CellProfiler statistcs (str) and inclusive lower/upper limits (tuple)

    Raises:
        ValueError: If there are multiple nuclei associated with a cell in the CellProfiler statistics tables

    Returns:
        list: List of filtered RegionProperties
    """
    row, col, _ = row_col_field_from_id(id)
    df_cp_stats_nuc = pd.read_csv(path_cp_stats_nuc)
    df_cp_stats_cell = pd.read_csv(path_cp_stats_cell)

    valid_regions = []
    for region in regionprops_pred:
        lbl = region.label

        object_data_nuc = df_cp_stats_nuc[
            (df_cp_stats_nuc["Metadata_WellColumn"] == col)
            & (df_cp_stats_nuc["Metadata_WellRow"] == row)
            & (df_cp_stats_nuc["ObjectNumber"] == lbl)
        ]

        num_cells = object_data_nuc["Children_Cells_Count"].to_numpy()[0]

        if num_cells == 1:
            object_data_cell = df_cp_stats_cell[
                (df_cp_stats_cell["Metadata_WellColumn"] == col)
                & (df_cp_stats_cell["Metadata_WellRow"] == row)
                & (df_cp_stats_cell["Parent_Nuclei"] == lbl)
            ]

            valid_regions.append(region)
            for feature_name in cp_stat_limits.keys():
                if (
                    cp_stat_limits[feature_name][0] != -1
                    and object_data_cell[feature_name].to_numpy()[0]
                    < cp_stat_limits[feature_name][0]
                ):
                    del valid_regions[-1]
                    break

                if (
                    cp_stat_limits[feature_name][1] != -1
                    and object_data_cell[feature_name].to_numpy()[0]
                    > cp_stat_limits[feature_name][1]
                ):
                    del valid_regions[-1]
                    break

        elif num_cells > 1:
            raise ValueError(f"Multiple cells for nucleus {lbl}")

    return valid_regions


def reg_prop_filter(regionprops_pred, reg_stat_limits):
    """Filter objects based on limits of specified RegionProperties statistics.

    Args:
        regionprops_pred (list): List of unfiltered RegionProperties
        reg_stat_limits (dict): Specified region properties (str) and inclusive lower/upper limits (tuple)

    Returns:
        list: List of filtered RegionProperties
    """

    valid_regions = []
    for region in regionprops_pred:
        valid_regions.append(region)

        for feature_name in reg_stat_limits.keys():
            if (
                reg_stat_limits[feature_name][0] != -1
                and region[feature_name] < reg_stat_limits[feature_name][0]
            ):
                del valid_regions[-1]
                break

            if (
                reg_stat_limits[feature_name][1] != -1
                and region[feature_name] > reg_stat_limits[feature_name][1]
            ):
                del valid_regions[-1]
                break

    return valid_regions


def path_to_filtered_seg(path_seg, reg_stat_limits):
    seg_soma = read_seg(path_seg)
    regions = reg_prop_filter(measure.regionprops(seg_soma), reg_stat_limits)
    seg_soma_filtered = np.zeros_like(seg_soma)
    # Make new labels consecutive
    for i_reg, region in enumerate(regions):
        seg_soma_filtered[seg_soma == region.label] = i_reg + 1

    return seg_soma_filtered


def row_col_field_from_id(id):
    row, col, field = int(id[1:3]), int(id[4:6]), int(id[7:9])
    return row, col, field


def get_id_from_name_96(name):
    items = name.split("_")
    id = items[-1][:4]
    return id


def get_id_to_path(path_dir, id_from_name, tag=None, remote=False):
    """Collect file paths at a directory into a dictionary organized by image ID.

    Args:
        path_dir (str): path to folder of images
        tag (str, optional): substring that will identify whether a file should be identified or not. Defaults to None.

    Returns:
        dict: Key is image ID (str) and value is file path (str) or list of file paths (in the case where different channels are different files).
        SFTPClient: optional, sftp client if files are from server. Only returned if remote is True
    """
    path_dir = Path(path_dir)

    if remote:
        ip, user, pswd = get_connection_details()
        transport = paramiko.Transport((ip, 22))
        transport.connect(None, user, pswd)
        sftp = paramiko.SFTPClient.from_transport(transport)
        files = [entry.filename for entry in sftp.listdir_attr(str(path_dir))]
    else:
        files = os.listdir(path_dir)

    if tag:
        files = [f for f in files if tag in f]

    id_to_path = {}
    for f in files:
        id = id_from_name(f)
        if id in id_to_path.keys():
            if isinstance(id_to_path[id], list):
                id_to_path[id] = sorted(id_to_path[id] + [path_dir / f])
            else:
                id_to_path[id] = sorted([id_to_path[id]] + [path_dir / f])
        else:
            id_to_path[id] = path_dir / f

    if not remote:
        for val in id_to_path.values():
            if isinstance(val, list):
                assert all([os.path.exists(v) for v in val])
            else:
                assert os.path.exists(val)

    if remote:
        return id_to_path, sftp
    else:
        return id_to_path


def get_connection_details():
    with open("/Users/thomasathey/.ssh/config") as f:
        data = f.readlines()
        for i, item in enumerate(data):
            if "fraenkel" in item:
                ip = None
                user = None
                for subitem in data[i + 1 : i + 10]:
                    if "User" in subitem:
                        user = subitem.split(" ")[-1].strip()
                    elif "HostName" in subitem:
                        ip = subitem.split(" ")[-1].strip()
                    if ip and user:
                        break
    with open("/Users/thomasathey/.ssh/fraenkel_password.txt") as f:
        pswd = f.readline()
        pswd = pswd.strip()

    return ip, user, pswd


def combine_soma_cell_labels(seg_soma, seg_cell):
    """Modify a cell semantic segmentation so it matches with a soma instance segmentation.
    Output satisfies:
    - Every cell pixel is connected to their soma
    - Every cell contains its soma
    - Cell pixels are assigned to their closest soma (that is connected and overlaps with the original cell footprint)

    Args:
        seg_soma (np.array): Soma instance segmentation. Integer type.
        seg_cell (np.array): Cell segmentation (can be semantic or instance).

    Returns:
        np.array: Cell instance segmentation such that each soma has at most one cell.
    """
    # assert mode(seg_soma.flatten()).mode == 0
    # assert mode(seg_cell.flatten()).mode == 0

    lbl_cell = measure.label(seg_cell)  # cell CC's

    lbl_soma_filtered = np.copy(lbl_cell)
    lbl_soma_filtered[seg_soma == 0] = 0  # intersection of soma and cell

    set_cell_lbls = set(np.unique(lbl_cell))  # set of cell CC's
    set_filtered_cell_lbls = set(
        np.unique(lbl_soma_filtered)
    )  # set of cell CC's that have somas
    set_filtered_out = (
        set_cell_lbls - set_filtered_cell_lbls
    )  # set of cell CC's that do not have soma

    # remove CC's that are not connected to soma
    for lbl in set_filtered_out:
        lbl_cell[lbl_cell == lbl] = 0
    seg_cell = lbl_cell > 0

    # Somas are part of cells
    seg_cell[seg_soma > 0] = 1

    # assign cell pixels to closest soma
    seg_cell_instance = segmentation.watershed(
        seg_cell, markers=seg_soma, mask=seg_cell
    )

    return seg_cell_instance
    


def combine_small_big_labels_instance(seg_small, seg_big):
    """Modify big object instance segmentation so it matches with small object instance segmentation.
    There must be a clear 1-to-1 mapping between big and small objects.

    Output satisfies:
    - Every small object pixel is a cell pixel
    - The big and small labels match

    Args:
        seg_small (np.array): Small object instance segmentation.
        seg_big (np.array): Big object instance segmentation. 1-to-1 mapping between cells and somas.
    """
    matches, unmatched_soma, unmatched_cell = match_small_to_big(seg_small, seg_big)
    assert len(unmatched_soma) == 0, "Some small objects do not have big objects"
    assert len(unmatched_cell) == 0, "Some big objects do not have small objects"

    seg_big_filtered = np.zeros_like(seg_big)
    for match in matches:
        if match[0] == 0:
            continue
        seg_big_filtered[seg_big == match[1]] = match[0]
    for match in matches:
        if match[0] == 0:
            continue
        seg_big_filtered[seg_small == match[0]] = match[0]

    # only take largest CC
    for big_id in np.unique(seg_big_filtered):
        if big_id == 0:
            continue

        single_obj = measure.label(seg_big_filtered == big_id)
        if single_obj.max() > 1:
            largestCC = (
                single_obj == np.argmax(np.bincount(single_obj.flat)[1:]) + 1
            )
            mask = np.logical_and(largestCC == 0, seg_big_filtered == big_id)
            seg_big_filtered[mask] = 0

    return seg_big_filtered

def combine_soma_nucleus_labels(seg_soma, seg_nuc):
    """Modify a nucleus segmentation so it matches with a soma instance segmentation.
    Output will satisfy:
    - Nuclei fall within soma
    - Nuclei have same label as surrounding soma
    - There is at most one (connected) nuclei per soma

    It does this by:
    - Removing nuclei that do not fall within soma
    - For each soma, removing all but the largest nucleus


    Args:
        seg_soma (np.array): Soma instance segmentation.
        seg_nuc (np.array): Nucleus segmentation (can be semantic or instance).

    Returns:
        np.array: Nucleus instance segmentation such that each soma has at most one nucleus.
    """
    assert mode(seg_soma.flatten()).mode == 0
    assert mode(seg_nuc.flatten()).mode == 0

    seg_nuc_filtered = np.copy(seg_soma)
    seg_nuc_filtered[seg_nuc == 0] = 0  # all nuclei must lie within somas

    for soma_id in np.unique(seg_nuc_filtered):
        if soma_id == 0:
            continue

        single_nuc_seg = measure.label(seg_nuc_filtered == soma_id)
        if single_nuc_seg.max() > 1:
            largestCC = (
                single_nuc_seg == np.argmax(np.bincount(single_nuc_seg.flat)[1:]) + 1
            )
            mask = np.logical_and(largestCC == 0, seg_soma == soma_id)
            seg_nuc_filtered[mask] = 0

    return seg_nuc_filtered


def match_small_to_big(seg_small, seg_big):
    small_ids = np.unique(seg_small)
    big_ids = np.unique(seg_big)

    assert len(small_ids) == np.amax(seg_small)+1, "Nucleus IDs are not consecutive"
    assert len(big_ids) == np.amax(seg_big)+1, "Soma IDs are not consecutive"

    assert mode(seg_small.flatten()).mode == 0, "Nucleus segmentation does not have background of 0"
    assert mode(seg_big.flatten()).mode == 0, "Soma segmentation does not have background of 0"

    small_to_big = np.zeros((len(small_ids), len(big_ids)))

    for small_id in small_ids:
        area_nuc = np.sum(seg_small == small_id)
        big_lbls = seg_big[seg_small == small_id]

        for big_lbl in np.unique(big_lbls):
            if big_lbl == 0:
                continue
            area_soma = np.sum(big_lbls == big_lbl)

            if area_soma / area_nuc > 0.5:
                small_to_big[small_id, big_lbl] = 1
                break
    # force background match
    small_to_big[:,0] = 0
    small_to_big[0,:] = 0
    small_to_big[0,0] = 1

    matches = []
    for small_id, row in enumerate(small_to_big):
        if np.sum(row) == 1:
            big_id = np.argmax(row)
            if np.sum(small_to_big[:, big_id]) == 1:
                matches.append((small_id, big_id))
            
    unmatched_small = np.where(np.sum(small_to_big, axis=1) != 1)[0]
    unmatched_big = np.where(np.sum(small_to_big, axis=0) != 1)[0]

    return matches, unmatched_small, unmatched_big


def check_valid_labels(seg_nuc, seg_soma, seg_cell):
    """Check if the soma, nucleus, and cell segmentations have valid labels.
    - For each compartment, the majority of pixels should be background.
    - For each compartment, all objects should be connected and have unique, consecutive labels.
    - Each soma should contain a nucleus with the same id.
    - Each cell should contain a soma with the same id.


    Args:
        seg_nuc (np.array): nucleus instance segmentation
        seg_soma (np.array): soma instance segmentation
        seg_cell (np.array): cell instance segmentation
    """
    check_valid_labels_comp(seg_nuc)
    check_valid_labels_comp(seg_soma)
    check_valid_labels_comp(seg_cell)

    check_valid_labels_pair(seg_nuc, seg_soma)
    check_valid_labels_pair(seg_soma, seg_cell)

def check_valid_labels_pair(seg_small, seg_big):
    """Check if two instance segmentation have valid labels.
    - Small objects should be contained within big objects of the same label

    Args:
        seg_small (np.array): Small object instance segmentation.
        seg_big (np.array): Big object instance segmentation.

    Raises:
        AssertionError: If labels are not a valid pair.
    """

    label_match = seg_small[seg_small > 0] == seg_big[seg_small > 0]
    if not label_match.all():
        mismatches = np.where(label_match == False)
        print(seg_small[seg_small > 0][mismatches])
        print("-------------->>>>>>>>>>>")
        print(seg_big[seg_small > 0][mismatches])
        raise AssertionError(f"Small and big labels do not match")


def check_valid_labels_comp(seg):
    """Check if a an instance segmentation has valid labels.
    - The majority of pixels should be background.
    - All objects should be connected and have unique, consecutive labels.

    Args:
        seg (np.array): Instance segmentation.

    Returns:
        bool: True if the conditions above are met to make the labels valid.
    """
    assert mode(seg.flatten()).mode == 0

    unq = np.unique(seg)
    assert len(unq) == np.amax(seg) + 1

    for id in unq:
        if id == 0:
            continue

        single_comp_seg = measure.label(seg == id)
        reg_props = measure.regionprops(single_comp_seg)

        if single_comp_seg.max() != 1:
            print(f"{single_comp_seg.max()} components with id {id} with centroids:")
            for reg in reg_props:
                print(reg.centroid)


def create_rgb(images, channels):
    """Create an RGB image from a list of grayscale images.

    Args:
        images (list): List of grayscale images.
        channels (list): List of channel indexes for RGB image.

    Returns:
        np.array: RGB image.
    """
    images = [im.astype("float64") for im in images]
    images = [np.linalg.norm(im, axis=-1) for im in images]
    images = [im / np.amax(im) for im in images]
    images = [exposure.equalize_adapthist(im, clip_limit=0.03) for im in images]
    im_rgb = np.stack([images[c] for c in channels], axis=2)

    return im_rgb


def label_celltype(
    path_dir_im,
    channels,
    path_dir_gt,
    out_dir,
    tag_im=".tif",
    tag_gt=".tif",
):
    out_dir = Path(out_dir)
    id_to_path_im = get_id_to_path(
        path_dir_im, tag_im, id_from_name=get_id_from_name_96
    )
    id_to_path_gt = get_id_to_path(
        path_dir_gt, tag_gt, id_from_name=get_id_from_name_96
    )

    margin = 3

    data = []

    for id in id_to_path_gt.keys():
        save = True
        images = read_ims(id_to_path_im[id])
        seg_gt = read_seg(id_to_path_gt[id][0])

        # create rgb image
        im_rgb = create_rgb(images, channels)

        regionprops_gt = measure.regionprops(seg_gt)

        for i, region in enumerate(regionprops_gt):
            if i % 10 == 0:
                print(f"{i}/{len(regionprops_gt)}")

            im_rgb_cropped = im_rgb[
                region.bbox[0] : region.bbox[2], region.bbox[1] : region.bbox[3], :
            ]
            seg_cropped = seg_gt[
                region.bbox[0] : region.bbox[2], region.bbox[1] : region.bbox[3]
            ]
            seg_cropped = seg_cropped == region.label

            # make a binary array that outlines the region of interest
            border = ndi.binary_dilation(seg_cropped == 0) & seg_cropped
            border = ndi.binary_dilation(border)
            border_masked = np.ma.masked_array(border, mask=border == 0)

            f, axs = plt.subplots(nrows=1, ncols=2)

            for ax in axs:
                ax.imshow(im_rgb_cropped)
                ax.axis("off")

            axs[1].imshow(border_masked, cmap="winter", alpha=0.3)

            f.set_figwidth(10)
            f.set_figheight(5)
            plt.tight_layout()

            plt.pause(0.1)
            plt.show(block=True)

            answer = input("1:alive, 2:dead, 3:unknown, s:skip sample")
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

    # # Collect responses for all samples
    # all_data = []
    # for id in id_to_path_gt.keys():
    #     with open(out_dir / f"{id}.pickle", "rb") as handle:
    #         all_data += pickle.load(handle)

    # # Save collected responses
    # with open(out_dir / f"all_data.pickle", "wb") as handle:
    #     pickle.dump(all_data, handle)

def threat_score(seg_gt, seg_pred, iou_threshold):
    assert iou_threshold >= 0.5 and iou_threshold <= 1.0

    gt_labels = np.unique(seg_gt[seg_gt > 0])
    pred_labels = np.unique(seg_pred[seg_pred > 0])

    tps = 0

    for id_gt in gt_labels:
        pred_labels_id = np.unique(seg_pred[seg_gt == id_gt])
        for id_pred in pred_labels_id:
            iou  = np.sum(np.logical_and(seg_gt == id_gt, seg_pred == id_pred)) / np.sum(np.logical_or(seg_gt == id_gt, seg_pred == id_pred))
            if iou >= iou_threshold:
                tps += 1
                break

    return tps, len(pred_labels) - tps, len(gt_labels) - tps
