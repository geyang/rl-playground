from os.path import join
from setuptools import setup
import sys

assert sys.version_info > (3, 6, 0), "Only support Python 3.6 and above."

with open(join("firedup", "VERSION")) as f:
    version = f.read()

setup(
    name="playground",
    py_modules=["firedup"],
    version=version,
    install_requires=[
        "cloudpickle",
        "gym[atari,box2d,classic_control]",
        "ipython",
        "joblib",
        "matplotlib",
        "mpi4py",
        "numpy",
        "pandas",
        "pytest",
        "psutil",
        "scipy",
        "seaborn",
        "torch>=1.5.1",
        "tqdm",
        "wandb",
    ],
    description="A collection of clean implementation of reinforcement learning algorithms",
    author="Ge Yang<ge.ike.yang@gmail.com>",
    license="MIT",
)
