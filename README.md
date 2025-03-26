![example workflow](https://github.com/Shavit-Lab/cell_paint_seg/actions/workflows/.github/workflows/python-app.yml/badge.svg)
![example workflow](https://github.com/Shavit-Lab/cell_paint_seg/actions/workflows/.github/workflows/black.yml/badge.svg)
[![codecov](https://codecov.io/gh/Shavit-Lab/cell_paint_seg/graph/badge.svg?token=0IYX9KSDKF)](https://codecov.io/gh/Shavit-Lab/cell_paint_seg)

# cell_paint_seg

This repository has two main functions
1. Scaffold to evaluate and compare 2D cell segmentation algorithms (e.g. CellProfiler, ilastik, Cellpose, see `eval-segmentation.ipynb`)
2. Code to deploy segmentation algorithms on image datasets. 


# Example Pipeline Usage
Move to project: `cd C:\Users\zeiss\projects\athey_als\`
Activate virtual environment: `venv_als_395\Scripts\activate`
Execute pipeline: `python .\cell_paint_seg\cell_paint_seg\scripts\run_ilastik_hier_cellpose.py --tif_dir <path to directory of tifs> --id_nchar <number of leading characters in tif filenames that define the image field e.g. e1_s001c2.tif is 7>`