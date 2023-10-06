import sys
import platform
import os
from shutil import rmtree
from pathlib import Path
from types import ModuleType
from importlib import import_module

import torch
from torch.utils import cpp_extension
from jax.lib import xla_client

CPP_MODULE_CACHED = None


def _generate_extension_version() -> str:
    py_impl = sys.implementation.name
    py_version = "".join(map(str, sys.version_info[:2]))
    py_abi_tag = sys.abiflags
    py_name_version = f"{py_impl}-{py_version}{py_abi_tag}"
    system_info = f"{platform.system().lower()}-{platform.machine()}"
    return f"{py_name_version}-{system_info}"


def compile_extension(force_recompile: bool = False) -> ModuleType:
    global CPP_MODULE_CACHED
    if not force_recompile and CPP_MODULE_CACHED is not None:
        return CPP_MODULE_CACHED

    mod_version = _generate_extension_version()
    build_dir = Path(f"~/.cache/torch2jax/{mod_version}").expanduser().absolute()
    if force_recompile and build_dir.exists():
        if build_dir.is_dir():
            rmtree(build_dir)
        else:
            os.remove(build_dir)
    build_dir.mkdir(exist_ok=True, parents=True)

    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        mod = import_module("torch2jax_cpp")
        # import torch2jax_cpp as mod  # noqa: F811
    except ImportError:
        print("Cache empty, we will compile the C++ extension component now...")
        source_prefix = Path(__file__).parent.absolute() / "cpp"
        source_list = ["main.cpp", "cpu_impl.cpp", "utils.cpp"]
        extra_cflags = ["-O3"]
        extra_cuda_cflags = None

        if torch.cuda.is_available():
            source_list.extend(["main.cu", "gpu_impl.cu"])
            extra_cflags.append("-DTORCH2JAX_WITH_CUDA")
            extra_cuda_cflags = ["-DTORCH2JAX_WITH_CUDA", "-O3"]
        mod = cpp_extension.load(
            "torch2jax_cpp",
            sources=[source_prefix / fname for fname in source_list],
            build_directory=build_dir,
            verbose=False,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
    for _name, _value in mod.cpu_registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="cpu")
    if hasattr(mod, "gpu_registrations"):
        for _name, _value in mod.gpu_registrations().items():
            xla_client.register_custom_call_target(_name, _value, platform="gpu")
    CPP_MODULE_CACHED = mod
    return mod


def import_extension() -> ModuleType:
    global CPP_MODULE_CACHED
    if CPP_MODULE_CACHED is not None:
        return CPP_MODULE_CACHED
    else:
        mod = import_module("torch2jax_cpp")
        return mod


def compile_and_import_module(force_recompile: bool = False):
    compile_extension(force_recompile)
    return import_extension()
