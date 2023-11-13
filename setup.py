from setuptools import setup, find_packages

setup(
    name="torch2jax",
    version="0.4.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "jax", "torch"],
    package_data={
        "": ["cpp/*.cpp", "cpp/*.cu", "cpp/*.h"],
    },
)
