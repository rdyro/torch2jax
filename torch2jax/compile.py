from pathlib import Path
from types import ModuleType

from torch.utils import cpp_extension
from jax.lib import xla_client

CPP_MODULE_CACHED = None


def compile_and_import_module() -> ModuleType:
    global CPP_MODULE_CACHED
    if CPP_MODULE_CACHED is not None:
        return CPP_MODULE_CACHED
    source_file = Path(__file__).absolute().parent / "cpp" / "torch_call.cu"
    mod = cpp_extension.load(name="torch_call", sources=[str(source_file)])
    for _name, _value in mod.cpu_registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="cpu")
    for _name, _value in mod.gpu_registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
    CPP_MODULE_CACHED = mod
    return mod
