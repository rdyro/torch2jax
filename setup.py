from setuptools import setup, find_packages
from pathlib import Path

import toml

config = toml.loads((Path(__file__).parent / "pyproject.toml").read_text())

setup(
    name="torch2jax",
    version=config["project"]["version"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "jax", "torch"],
    package_data={
        "": ["cpp/*.cpp", "cpp/*.cu", "cpp/*.h"],
    },
)
