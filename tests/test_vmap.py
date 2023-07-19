from __future__ import annotations

import sys
from pathlib import Path

import torch
import jax
from jax import numpy as jnp, Array
from jax.nn import softmax
from jax.tree_util import tree_flatten
from jax.scipy.linalg import cho_factor, cho_solve

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax_with_vjp, tree_t2j, tree_j2t  # noqa: E402
from utils import jax_randn  # noqa: E402

####################################################################################################


def test_simple_vmap():
    def torch_fn(A, x):
        # return torch.linalg.solve_triangular(torch.linalg.cholesky(A, upper=True), x, upper=True)
        return torch.linalg.solve(A, x)

    def expected_fn(A, x):
        return cho_solve(cho_factor(A), x)

    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    dtype_list = [jnp.float32, jnp.float64]
    for use_torch_vmap in [True, False]:
        for device in device_list:
            for dtype in dtype_list:
                A = jax_randn((10, 5, 5), dtype=dtype, device=device)
                A = jnp.swapaxes(A, -1, -2) @ A + 1e-2 * jnp.tile(
                    jnp.diag(1.0 + 0.0 * jax_randn((A.shape[-1],), dtype=dtype, device=device)),
                    (A.shape[0], 1, 1),
                )
                x = jax_randn((5, 7), dtype=dtype, device=device)
                jax_fn = torch2jax_with_vjp(
                    torch_fn, A, x, output_shapes=x, depth=2, use_torch_vmap=use_torch_vmap
                )
                sol = jax.vmap(jax_fn, in_axes=(0, None))(A, x)
                sol_expected = jax.vmap(expected_fn, in_axes=(0, None))(A, x)
                err = jnp.linalg.norm(sol - sol_expected) / jnp.linalg.norm(sol_expected)
                assert err < 1e-3


if __name__ == "__main__":
    test_simple_vmap()
