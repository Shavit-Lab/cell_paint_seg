import subprocess
from tqdm import tqdm
import os
from skimage import io
from pathlib import Path
from cell_paint_seg.utils import (
    get_id_to_path,
)


def apply_ilastik_images(h5_files, ilastik_path, ilastik_project):
    if len(h5_files) > 10:
        h5_files_batches = [h5_files[i : i + 10] for i in range(0, len(h5_files), 10)]
    else:
        h5_files_batches = [h5_files]

    for h5_files_batch in tqdm(h5_files_batches, desc="executing pixel classification"):
        command = [
            ilastik_path,
            "--headless",
            f"--project={ilastik_project}",
        ] + h5_files_batch

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def apply_ilastik_multicut(
    h5_files, ilastik_path, multicut_project, output_dir_path, blank_seg
):
    output_dir_path = Path(output_dir_path)
    export_source = "Multicut Segmentation"
    output_file_suffix = "-ch8sk1fk1fl1.tif"
    output_format = f"{output_dir_path}/{{nickname}}{output_file_suffix}"

    for h5_file in tqdm(h5_files, desc="executing multicut"):
        prob_file = str(h5_file)[:-3] + "_Probabilities_hier.h5"
        command = [
            ilastik_path,
            "--headless",
            f"--project={multicut_project}",
            f"--raw_data={h5_file}",
            f"--probabilities={prob_file}",
            f"--export_source={export_source}",
            f"--output_filename_format={output_format}",
        ]

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        cut_file = output_dir_path / (h5_file.stem + output_file_suffix)

        # assume because whole image was classified as boundary
        if not os.path.isfile(cut_file):
            print(f"No cells detected in {h5_file} - writing blank segmentation...")
            blank_seg.save(cut_file)
        else:
            seg = io.imread(cut_file)
            io.imsave(cut_file, seg.T)


def apply_ilastik_obj_class(
    h5_files,
    segmentation_path,
    ilastik_path,
    multicut_project,
    id_from_name,
):
    export_source = "Object Predictions"

    id_to_path_seg = get_id_to_path(
        segmentation_path, tag=".h5", id_from_name=id_from_name
    )

    for h5_file in tqdm(h5_files, desc="executing object classification"):
        id = str(h5_file.stem)
        seg_file = id_to_path_seg[id]
        command = [
            ilastik_path,
            "--headless",
            f"--project={multicut_project}",
            f"--raw_data={h5_file}",
            f"--segmentation_image={seg_file}",
            f"--export_source={export_source}",
        ]

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
