import sys
from shutil import rmtree
from pathlib import Path
from types import ModuleType

import torch
from torch.utils import cpp_extension
from jax.lib import xla_client

CPP_MODULE_CACHED = None


def compile_and_import_module(force_recompile: bool = False) -> ModuleType:
    global CPP_MODULE_CACHED
    if not force_recompile and CPP_MODULE_CACHED is not None:
        return CPP_MODULE_CACHED

    source_prefix = Path(__file__).parent.absolute() / "cpp"
    source_list = ["main.cpp", "cpu_impl.cpp", "utils.cpp"]
    extra_cflags = ["-O3"]
    extra_cuda_cflags = None

    if torch.cuda.is_available():
        source_list.extend(["main.cu", "gpu_impl.cu"])
        extra_cflags.append("-DTORCH2JAX_WITH_CUDA")
        extra_cuda_cflags = ["-DTORCH2JAX_WITH_CUDA", "-O3"]

    build_dir = Path("~/.cache/torch2jax").expanduser().absolute()
    if force_recompile:
        rmtree(build_dir)
    build_dir.mkdir(exist_ok=True)

    mod = cpp_extension.load(
        "torch2jax_cpp",
        sources=[source_prefix / fname for fname in source_list],
        build_directory=build_dir,
        verbose=False,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    sys.path.append(build_dir)
    import torch2jax_cpp as mod

    for _name, _value in mod.cpu_registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="cpu")
    if hasattr(mod, "gpu_registrations"):
        for _name, _value in mod.gpu_registrations().items():
            xla_client.register_custom_call_target(_name, _value, platform="gpu")
    CPP_MODULE_CACHED = mod
    return mod
