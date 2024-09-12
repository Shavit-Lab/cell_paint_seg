import subprocess
from tqdm import tqdm
import os
from pathlib import Path
from cell_paint_seg.utils import get_id_to_path, get_id_from_name_first_us

models_dir_path = Path(os.path.realpath(__file__)).parents[1] / "models"


def apply_cpose(tif_dir, output_dir, nuclei=False):
    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)

    # somas
    run_cpose(tif_dir, 1, 2, project="cyto3")

    id_to_path = get_id_to_path(
        tif_dir, tag="masks.tif", id_from_name=get_id_from_name_first_us
    )
    for id, path in tqdm(id_to_path.items()):
        os.rename(path, output_dir / f"{id}-ch8sk1fk1fl1.tif")

    if nuclei:
        # nuclei
        path_nuclei_model = models_dir_path / "CP_20240911_083848_nuclei"
        run_cpose(tif_dir, 2, 0, project=path_nuclei_model)

        # id_to_path = get_id_to_path(tif_dir, tag="masks.tif", id_from_name=get_id_from_name_first_us)
        # for id, path in tqdm(id_to_path.items()):
        #     os.rename(path, output_dir / f"{id}-ch8sk1fk1fl1.tif")


def run_cpose(tif_dir, fg_channel, bg_channel, project="cyto3"):
    # nuclei
    command = [
        "python",
        "-m",
        "cellpose",
        f"--dir={tif_dir}",
        f"--pretrained_model={project}",
        f"--chan={fg_channel}",
        f"--chan2={bg_channel}",
        "--diameter=0.",
        "--save_tif",
        "--verbose",
    ]

    subprocess.run(
        command,
    )
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
