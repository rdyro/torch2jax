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
    """Transfer a JAX array or PyTorch tensor to the other framework. Assume only 1 GPU."""
    assert via in ("dlpack", "cpu")
    if isinstance(x, Array):
        if via == "dlpack":
            return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
        else:
            if isinstance(device, JAXDevice):
                torch_device = torch.device("cuda" if device.platform == "gpu" else "cpu")
            else:
                torch_device = torch.device(device)
            return torch.as_tensor(np.array(x), device=torch_device)
    else:
        if via == "dlpack":
            return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        else:
            if isinstance(device, JAXDevice):
                jax_device = device
            else:
                jax_device = jax.devices(
                    device.type if isinstance(device, torch.device) else device
                )[0]
            return jax.device_put(jax.numpy.array(x.detach().cpu().numpy()), device=jax_device)


def j2t(x: Array, via: str = "dlpack") -> Tensor:
    try:
        device = x.device()
    except ConcretizationTypeError:
        msg = "You are attempting to convert a non-concrete JAX array to a PyTorch tensor."
        msg += " This is not supported, since that JAX array does not contain any numbers."
        raise RuntimeError(msg)
    return transfer(x, via=via, device=device)


def t2j(x: Tensor, via: str = "dlpack") -> Array:
    return transfer(x, via=via, device=x.device)


def tree_j2t(xs: list[Array] | tuple[Array], via: str = "dlpack") -> list[Tensor] | tuple[Tensor]:
    return tree_map(lambda x: j2t(x, via=via) if isinstance(x, Array) else x, xs)


def tree_t2j(xs: list[Tensor] | tuple[Array], via: str = "dlpack") -> list[Array] | tuple[Array]:
    return tree_map(lambda x: t2j(x, via=via) if isinstance(x, Tensor) else x, xs)
