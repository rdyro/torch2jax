from setuptools import setup, find_packages

setup(
    name="torch2jax",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["cpp/torch_call.cu", "cpp/torch_call.h"],
    },
)
