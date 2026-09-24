from __future__ import annotations

import contextlib

import numpy as np
import jax.dlpack
import torch
from torch import Tensor
import torch.utils.dlpack
from jax import Array

try:
    from jax.errors import ConcretizationTypeError
except ImportError:
    from jax.core import ConcretizationTypeError

JAXDevice = jax.Device if hasattr(jax, "Device") else jax.lib.xla_extension.Device

_ERR_MULTI_DEVICE = (
    "You are attempting to convert a JAX array with multiple devices to a PyTorch tensor. This is not supported"
)
_ERR_NON_CONCRETE = (
    "You are attempting to convert a non-concrete JAX array to a PyTorch tensor."
    " This is not supported, since that JAX array does not contain any numbers."
)


def _transfer(x: Array | Tensor, via: str = "dlpack", device: str = "cuda"):
    """Transfer a JAX array or PyTorch tensor to the other framework. Assume only 1 GPU."""
    assert via in ("dlpack", "cpu", "host")
    if isinstance(x, Array):
        if via == "dlpack":
            return torch.utils.dlpack.from_dlpack(x)
        else:
            if isinstance(device, JAXDevice):
                torch_device = torch.device("cuda" if device.platform == "gpu" else "cpu")
            else:
                torch_device = torch.device(device)
            return torch.as_tensor(np.array(x), device=torch_device)
    else:
        if via == "dlpack":
            # torch only exports a CUDA tensor via DLPack when its device is the current one, e.g., JAX initialization
            # can leave another GPU current
            with torch.cuda.device(x.device) if x.is_cuda else contextlib.nullcontext():
                return jax.dlpack.from_dlpack(x.detach())
        else:
            if isinstance(device, JAXDevice):
                jax_device = device
            else:
                jax_device = jax.devices(device.type if isinstance(device, torch.device) else device)[0]
            return jax.device_put(jax.numpy.array(x.detach().cpu().numpy()), device=jax_device)


def j2t(x: Array, via: str = "dlpack") -> Tensor:
    """Transfer a single jax.Array to a PyTorch tensor."""
    try:
        devices = x.devices()
        if len(devices) > 1:
            raise RuntimeError(_ERR_MULTI_DEVICE)
        device = list(devices)[0]
    except ConcretizationTypeError:
        raise RuntimeError(_ERR_NON_CONCRETE)
    return _transfer(x, via=via, device=device)


def t2j(x: Tensor, via: str = "dlpack") -> Array:
    """Transfer a single PyTorch tensor to a jax.Array."""
    return _transfer(x, via=via, device=x.device)


def tree_j2t(xs: list[Array] | tuple[Array], via: str = "dlpack") -> list[Tensor] | tuple[Tensor]:
    """Transfer a tree of PyTorch tensors to a corresponding tree of jax.Array-s."""
    return jax.tree.map(lambda x: j2t(x, via=via) if isinstance(x, Array) else x, xs)


def tree_t2j(xs: list[Tensor] | tuple[Array], via: str = "dlpack") -> list[Array] | tuple[Array]:
    """Transfer a tree of  jax.Array-s to a corresponding tree of PyTorch tensors."""
    return jax.tree.map(lambda x: t2j(x, via=via) if isinstance(x, Tensor) else x, xs)
