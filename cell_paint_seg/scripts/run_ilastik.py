from pathlib import Path
import os
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess
import time
from skimage import io

test = False

if test:
    parent_dir = "C:\\Users\\zeiss\\projects\\athey_als\\test-images"
    parent_dir = Path(parent_dir)
    tif_path = parent_dir / "tifs"
    hdf5_path = parent_dir / "hdf5s"
    output_path = parent_dir / "segmentations"
    convert = False
    run_pxl_classification = False

else:
    parent_dir = (
        "D:\\Aneesh\\Assay Dev 20230329\\BR00142687__2024-03-29T18_18_57-Measurement 1"
    )
    parent_dir = Path(parent_dir)
    tif_path = parent_dir / "Images"
    hdf5_path = parent_dir / "hdf5s"
    output_path = parent_dir / "segmentations"
    convert = False
    run_pxl_classification = False

ilastik_path = "C:\\Program Files\\ilastik-1.4.0.post1\\ilastik.exe"
bdry_pxl_path = "C:\\Users\\zeiss\\projects\\athey_als\\cell_paint_seg\\models\\mb-vs-nonmb-pxlclass.ilp"
multicut_path = "C:\\Users\\zeiss\\projects\\athey_als\\cell_paint_seg\\models\\mb-vs-nonmb-multicut.ilp"


files = os.listdir(tif_path)
files = [f for f in files if ".tif" in f]
image_names = [f.split("-")[0] for f in files]

# convert to hdf5
image_names = set(image_names)
n_files = len(image_names)
print(f"Converting the following images to hdf5s: {image_names}")

time_start = time.time()
for image_name in tqdm(image_names, desc="Converting tifs to hdf5s"):
    channels = {}
    for file in files:
        if image_name in file:
            file_path = tif_path / file
            c = int(file_path.stem.split("-")[-1][2])
            im = Image.open(file_path)
            im = np.array(im)
            channels[c] = im

    im_allc = np.stack([channels[i] for i in range(1, 7)], axis=2)

    if convert:
        with h5py.File(hdf5_path / f"{image_name}.h5", "a") as h5:
            h5.create_dataset(f"image", data=im_allc)
    else:
        break

im_shape = im.shape

time_convert = time.time()

# run headless pixel classification
print(f"Running ilastik pixel classification")
files = os.listdir(hdf5_path)

h5_files = [hdf5_path / f for f in files if ".h5" in f]
h5_files = [h for h in h5_files if "_Probabilities.h5" not in str(h)]

if run_pxl_classification:
    h5_files_batches = [h5_files[i : i + 10] for i in range(0, len(h5_files), 10)]

    for h5_files_batch in tqdm(
        h5_files_batches, desc="executing boundary classification"
    ):
        # print(h5_files_batch)
        command = [
            ilastik_path,
            "--headless",
            f"--project={bdry_pxl_path}",
        ] + h5_files_batch

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

time_bdry_pxl = time.time()

# run headless multicut
blank_seg = np.zeros(im_shape, dtype="int32")
blank_seg = Image.fromarray(blank_seg)

export_source = "Multicut Segmentation"
output_format = f"{output_path}/{{nickname}}-ch7sk1fk1fl1.tif"
for h5_file in tqdm(h5_files, desc="executing multicut"):
    prob_file = str(h5_file)[:-3] + "_Probabilities.h5"
    command = [
        ilastik_path,
        "--headless",
        f"--project={multicut_path}",
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

    cut_file = output_path / (h5_file.stem + "-ch7sk1fk1fl1.tif")
    # assume because whole image was classified as boundary
    if not os.path.isfile(cut_file):
        print(f"No cells detected in {h5_file} - writing blank segmentation...")
        blank_seg.save(cut_file)
    else:
        seg = io.imread(cut_file)
        io.imsave(cut_file, seg.T)


time_cut = time.time()
print(
    f"Time for {n_files} image sites w/{len(channels.items())} channels: (Convert, {time_convert-time_start}), (Boundary pred., {time_bdry_pxl-time_convert}), (Multicut, {time_cut-time_bdry_pxl})"
)
