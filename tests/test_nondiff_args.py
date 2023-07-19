from __future__ import annotations

import sys
from pathlib import Path

import torch
import jax
from jax import numpy as jnp, Array, ShapeDtypeStruct
from jax.nn import softmax

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax_with_vjp  # noqa: E402
from torch2jax import j2t, t2j, tree_j2t, tree_t2j  # noqa: E402
from torch2jax.utils import dtype_t2j, dtype_j2t  # noqa: E402
from utils import jax_randn  # noqa: E402


def test_int_args():
    def fn(a, b, c):
        return a + b

    a, b = torch.randn(10), torch.randn(10)
    c = torch.randint(0, 100, size=(10,))

    for use_torch_vjp in [True, False]:
        fn_jax = torch2jax_with_vjp(
            fn,
            a,
            b,
            c,
            output_shapes=ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
            nondiff_argnums=(2,),
            depth=2,
            use_torch_vjp=use_torch_vjp,
        )
        a, b, c = tree_t2j((a, b, c))
        f = fn_jax(a, b, c)


def test_gradient():
    def fn(a, b, c):
        return torch.sin(a + 2 * b) * torch.softmax(a - b, dim=0).reshape(-1)[0]

    def jax_fn(a, b, c):
        return jnp.sin(a + 2 * b) * softmax(a - b, axis=0).reshape(-1)[0]

    a, b = torch.randn(10), torch.randn(10)
    c = torch.randint(0, 100, size=(10,))

    for use_torch_vjp in [True, False]:
        fn_jax = torch2jax_with_vjp(
            fn, a, b, c, nondiff_argnums=(2,), depth=2, use_torch_vjp=use_torch_vjp
        )
    a, b, c = tree_t2j((a, b, c))
    g = jax.grad(lambda *args: jnp.sum(fn_jax(*args)), argnums=(0, 1))(a, b, c)

    def fn(a, c, b):
        return torch.sin(a + 2 * b) * torch.softmax(a - b, dim=0).reshape(-1)[0]

    a, b = torch.randn(10), torch.randn(10)
    c = torch.randint(0, 100, size=(10,))

    for use_torch_vjp in [True, False]:
        fn_jax = torch2jax_with_vjp(
            fn, a, c, b, nondiff_argnums=(1,), depth=2, use_torch_vjp=use_torch_vjp
        )
    a, b, c = tree_t2j((a, b, c))
    g = jax.grad(lambda *args: jnp.sum(fn_jax(*args)), argnums=(0, 2))(a, c, b)


def test_jacobian():
    def fn(a, b, c):
        return torch.sin(a + 2 * b) * torch.softmax(a - b, dim=0).reshape(-1)[0]

    def jax_fn(a, b, c):
        return jnp.sin(a + 2 * b) * softmax(a - b, axis=0).reshape(-1)[0]

    at, bt = torch.randn(10), torch.randn(10)
    ct = torch.randint(0, 100, size=(10,))

    for use_torch_vjp in [True, False]:
        fn_jax = torch2jax_with_vjp(
            fn, at, bt, ct, depth=2, use_torch_vjp=use_torch_vjp, use_torch_vmap=use_torch_vjp
        )
        a, b, c = tree_t2j((at, bt, ct))
        f = jax.jacobian(fn_jax)(a, b, c)
        err = jnp.linalg.norm(f - jax.jacobian(jax_fn)(a, b, c))
        assert err < 1e-5
        Ja, Jb = jax.jacobian(fn_jax, argnums=(0, 1))(a, b, c)
        err = jnp.linalg.norm(Ja - jax.jacobian(jax_fn, argnums=0)(a, b, c))
        assert err < 1e-5
        err = jnp.linalg.norm(Jb - jax.jacobian(jax_fn, argnums=1)(a, b, c))
        assert err < 1e-5
        Ja = jax.jacobian(fn_jax, argnums=0)(a, b, c)
        err = jnp.linalg.norm(Ja - jax.jacobian(jax_fn, argnums=0)(a, b, c))
        assert err < 1e-5


if __name__ == "__main__":
    test_int_args()
    test_gradient()
    test_jacobian()
