import subprocess
from tqdm import tqdm
import os
import sys
from pathlib import Path
from cell_paint_seg.utils import get_id_to_path

models_dir_path = Path(os.path.realpath(__file__)).parents[1] / "models"


def apply_cpose(tif_dir, output_dir, id_from_name, nuclei=False):
    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)

    # somas
    run_cpose(tif_dir, 1, 2, project="cyto3")

    id_to_path = get_id_to_path(tif_dir, tag="masks.tif", id_from_name=id_from_name)
    for id, path in tqdm(id_to_path.items()):
        os.rename(path, output_dir / f"{id}c8.tif")

    if nuclei:
        # nuclei
        path_nuclei_model = models_dir_path / "CP_20240911_083848_nuclei"
        run_cpose(tif_dir, 2, 0, project=path_nuclei_model)


def run_cpose(tif_dir, fg_channel, bg_channel, project="cyto3"):
    # nuclei
    command = [
        sys.executable,
        "-m",
        "cellpose",
        f"--dir={tif_dir}",
        f"--pretrained_model={project}",
        f"--chan={fg_channel}",
        f"--chan2={bg_channel}",
        "--diameter=0.",
        "--save_tif",
        "--no_npy",
        "--verbose",
    ]

    subprocess.run(
        command,
    )
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
