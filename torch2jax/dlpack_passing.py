from __future__ import annotations

import numpy as np
import jax.dlpack
import torch
from torch import Tensor
import torch.utils.dlpack
from jax import Array
from jax.core import ConcretizationTypeError
from jax.tree_util import tree_map

JAXDevice = jax.lib.xla_extension.Device

def transfer(x: Array | Tensor, via: str = "dlpack", device: str = "cuda"):
    assert via in ("dlpack", "cpu")
    if isinstance(x, Array):
        if via == "dlpack":
            return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
        else:
            return torch.as_tensor(np.array(x), device=device)
    else:
        if via == "dlpack":
            return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        else:
            device = jax.devices(device)[0] if not isinstance(device, JAXDevice) else device
            return jax.device_put(jax.numpy.array(x.detach().cpu().numpy()), device=device)

def j2t(x: Array) -> Tensor:
    try:
        device = x.device()
    except ConcretizationTypeError:
        msg = "You are attempting to convert a non-convert JAX array to a PyTorch tensor."
        msg += " This is not supported."
        raise RuntimeError(msg)
    return transfer(x, via="dlpack", device=device)

def t2j(x: Tensor) -> Array:
    return transfer(x, via="dlpack", device=x.device)

def tree_j2t(xs: list[Array] | tuple[Array]) -> list[Tensor] | tuple[Tensor]:
    return tree_map(lambda x: j2t(x) if isinstance(x, Array) else x, xs)

def tree_t2j(xs: list[Tensor] | tuple[Array]) -> list[Array] | tuple[Array]:
    return tree_map(lambda x: t2j(x) if isinstance(x, Tensor) else x, xs)
