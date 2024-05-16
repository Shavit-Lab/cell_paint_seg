from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name='ilastik_profiler',
    version=__version__,
    packages=['ilastik_profiler'],  
    author='Thomas Athey',
    author_email='tom.l.athey@gmail.com',

    install_requires=[
        'h5py',
        'pillow',
        'numpy',
        'tqdm'
    ],
    license='MIT'
)