from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name="cell_paint_seg",
    version=__version__,
    packages=["cell_paint_seg"],
    author="Thomas Athey",
    author_email="tom.l.athey@gmail.com",
    install_requires=[
        "h5py",
        "pillow",
        "numpy",
        "tqdm",
        "scikit-image",
        "pandas",
        "scipy",
        "matplotlib",
        "ipython",
    ],
    license="MIT",
)
