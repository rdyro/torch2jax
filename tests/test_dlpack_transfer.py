import sys
from pathlib import Path

import torch
from torch import Tensor
from jax import numpy as jnp
from jax import Array
import numpy as np


paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils import jax_randn
from torch2jax import j2t, t2j, tree_j2t, tree_t2j

DTYPE_MAP = {torch.float32: jnp.float32, torch.float64: jnp.float64}
DEVICE_MAP = {"gpu": "cuda", "cpu": "cpu", "cuda": "cuda"}


def test_dlpack_transfer():
    device_list = ["cpu", "cuda"]
    shape = (2, 3, 5, 1)
    for device in device_list:
        for dtype in [torch.float32, torch.float64]:
            # jax -> torch
            x1 = jax_randn(shape, device, DTYPE_MAP[dtype])
            x2 = j2t(x1)
            assert DEVICE_MAP[x2.device.type] == DEVICE_MAP[device] and x2.dtype == dtype
            err = np.linalg.norm(x2.cpu().numpy() - np.array(x1)) / np.linalg.norm(np.array(x1))
            assert err < 1e-5

            # torch -> jax
            x1 = torch.randn(shape, device=device, dtype=dtype)
            x2 = t2j(x1)
            assert (
                DEVICE_MAP[x2.device().platform] == DEVICE_MAP[device]
                and x2.dtype == DTYPE_MAP[dtype]
            )
            err = np.linalg.norm(x1.cpu().numpy() - np.array(x2)) / np.linalg.norm(x1.cpu().numpy())
            assert err < 1e-5


def test_tree_dlpack_transfer():
    args = dict(
        a=1,
        b=jax_randn((10, 2), dtype=jnp.float32, device="cuda"),
        c="hello",
        d=(
            1,
            jax_randn((10, 2), dtype=jnp.float32, device="cuda"),
            torch.randn((10, 2), device="cuda"),
        ),
    )
    args2 = tree_j2t(args)
    assert isinstance(args2["a"], int)
    assert DEVICE_MAP[args2["b"].device.type] == "cuda"
    assert isinstance(args2["c"], str)
    assert isinstance(args2["d"][1], Tensor)
    assert DEVICE_MAP[args2["d"][1].device.type] == "cuda"
    assert isinstance(args2["d"][2], Tensor)

    args = dict(
        a=1,
        b=torch.randn((10, 2), device="cuda"),
        c="hello",
        d=(
            1,
            torch.randn((10, 2), device="cuda"),
            jax_randn((10, 2), dtype=jnp.float32, device="cuda"),
        ),
    )
    args2 = tree_t2j(args)
    assert isinstance(args2["a"], int)
    assert DEVICE_MAP[args2["b"].device().platform] == "cuda"
    assert isinstance(args2["c"], str)
    assert isinstance(args2["d"][1], Array)
    assert DEVICE_MAP[args2["d"][1].device().platform] == "cuda"
    assert isinstance(args2["d"][2], Array)
