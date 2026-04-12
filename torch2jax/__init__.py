from .api import dtype_t2j, torch2jax_without_vjp  # noqa: F401
from .compile import compile_and_import_module  # noqa: F401
from .dlpack_passing import j2t, t2j, tree_j2t, tree_t2j  # noqa: F401
from .gradients import torch2jax, torch2jax_with_vjp  # noqa: F401
from .auto import torch2jax_auto  # noqa: F401

from torch import Size  # noqa: F401

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version
__version__ = version(__name__)

wrap_torch_fn = torch2jax
