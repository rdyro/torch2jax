from __future__ import annotations

import sys
from pathlib import Path

import torch
import jax
from jax import numpy as jnp, Array

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax_with_vjp  # noqa: E402
from torch2jax import j2t, t2j, tree_j2t, tree_t2j  # noqa: E402
from utils import jax_randn  # noqa: E402


def test_int_args():
    def fn(a, b, c):
        return a + b

    a, b = torch.randn(10), torch.randn(10)
    c = torch.randint(0, 100, size=(10,))

    fn_jax = torch2jax_with_vjp(fn, a, b, c, nondiff_argnums=(2,), depth=2)
    a, b, c = tree_t2j((a, b, c))
    print(fn_jax(a, b, c))


def test_gradient():
    def fn(a, b, c):
        return a + 2 * b

    a, b = torch.randn(10), torch.randn(10)
    c = torch.randint(0, 100, size=(10,))

    fn_jax = torch2jax_with_vjp(fn, a, b, c, nondiff_argnums=(2,), depth=2)
    a, b, c = tree_t2j((a, b, c))
    print(jax.grad(lambda *args: jnp.sum(fn_jax(*args)), argnums=(0, 1))(a, b, c))


def test_jacobian():
    def fn(a, b, c):
        return a + 2 * b

    a, b = torch.randn(10), torch.randn(10)
    c = torch.randint(0, 100, size=(10,))

    fn_jax = torch2jax_with_vjp(fn, a, b, c, nondiff_argnums=(2,), depth=2)
    a, b, c = tree_t2j((a, b, c))
    print(jax.jacobian(fn_jax)(a, b, c))
    print()
    print(jax.jacobian(fn_jax, argnums=(0, 1))(a, b, c))


if __name__ == "__main__":
    #test_int_args()
    #test_gradient()
    test_jacobian()
