from setuptools import setup, find_packages

setup(
    name="torch2jax",
    version="0.4.4",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["cpp/*.cpp", "cpp/*.cu", "cpp/*.h"],
    },
)
