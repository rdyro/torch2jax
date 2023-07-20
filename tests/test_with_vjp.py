from __future__ import annotations

import sys
import pdb
from pathlib import Path

import torch
import jax
from jax import numpy as jnp, Array
from jax.nn import softmax
from jax.tree_util import tree_flatten

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax_with_vjp, tree_t2j, tree_j2t  # noqa: E402
from utils import jax_randn  # noqa: E402

####################################################################################################


def test_torch2jax_with_vjp():
    shape = (5, 7)

    def torch_fn(x, y):
        return x**2 * y**2

    def expected_f_fn(x, y):
        return x**2 * y**2

    expected_g_fn = jax.grad(lambda *args: jnp.sum(expected_f_fn(*args)), argnums=(0, 1))
    expected_h_fn = jax.grad(
        lambda *args: jnp.sum(expected_g_fn(*args)[0] + expected_g_fn(*args)[1]), argnums=(0, 1)
    )

    xt, yt = torch.randn(shape), torch.randn(shape)
    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    dtype_list = [jnp.float32, jnp.float64]

    for use_torch_vjp in [True, False]:
        wrap_jax_f_fn = torch2jax_with_vjp(torch_fn, xt, yt, depth=2, use_torch_vjp=use_torch_vjp)
        wrap_jax_g_fn = jax.grad(lambda x, y: jnp.sum(wrap_jax_f_fn(x, y)), argnums=(0, 1))
        wrap_jax_h_fn = jax.grad(
            lambda x, y: jnp.sum(wrap_jax_g_fn(x, y)[0] + wrap_jax_g_fn(x, y)[1]), argnums=(0, 1)
        )

        for device in device_list:
            for dtype in dtype_list:
                x = jax_randn(shape, dtype=dtype, device=device)
                y = jax_randn(shape, dtype=dtype, device=device)
                f = wrap_jax_f_fn(x, y)
                g = wrap_jax_g_fn(x, y)
                h = wrap_jax_h_fn(x, y)

                f_expected = expected_f_fn(x, y)
                g_expected = expected_g_fn(x, y)
                h_expected = expected_h_fn(x, y)

                # test output structure #############################
                assert isinstance(f, Array)
                assert (
                    isinstance(g, (tuple, list))
                    and len(g) == 2
                    and isinstance(g[0], Array)
                    and isinstance(g[1], Array)
                )
                assert (
                    isinstance(h, (tuple, list))
                    and len(h) == 2
                    and isinstance(h[0], Array)
                    and isinstance(h[1], Array)
                )

                # test values not under JIT #########################
                err_f = jnp.linalg.norm(f - f_expected)
                err_g = jnp.linalg.norm(g[0] - g_expected[0]) + jnp.linalg.norm(
                    g[1] - g_expected[1]
                )
                err_h = jnp.linalg.norm(h[0] - h_expected[0]) + jnp.linalg.norm(
                    h[1] - h_expected[1]
                )
                print(f"Error in f value is {err_f:.4e}")
                print(f"Error in g value is {err_g:.4e}")
                print(f"Error in h value is {err_h:.4e}")

                try:
                    assert err_f.block_until_ready() < 1e-5, f"Error in f value is {err_f:.4e}"
                    assert err_g.block_until_ready() < 1e-5, f"Error in g value is {err_g:.4e}"
                    assert err_h.block_until_ready() < 1e-5, f"Error in h value is {err_h:.4e}"
                except:
                    pdb.set_trace()

                # test values when under JIT ########################
                f = jax.jit(wrap_jax_f_fn)(x, y)
                g = jax.jit(wrap_jax_g_fn)(x, y)
                h = jax.jit(wrap_jax_h_fn)(x, y)

                err_f = jnp.linalg.norm(f - f_expected)
                err_g = jnp.linalg.norm(g[0] - g_expected[0]) + jnp.linalg.norm(
                    g[1] - g_expected[1]
                )
                err_h = jnp.linalg.norm(h[0] - h_expected[0]) + jnp.linalg.norm(
                    h[1] - h_expected[1]
                )

                try:
                    assert err_f.block_until_ready() < 1e-5, f"Error in f value is {err_f:.4e}"
                    assert err_g.block_until_ready() < 1e-5, f"Error in g value is {err_g:.4e}"
                    assert err_h.block_until_ready() < 1e-5, f"Error in h value is {err_h:.4e}"
                except:
                    pdb.set_trace()


def test_jacobian():
    shape = (2, 3)

    def fn(a, b):
        return torch.sin(a + 2 * b) * torch.softmax(a - b, dim=0).reshape(-1)[0]

    def jax_fn(a, b):
        return jnp.sin(a + 2 * b) * softmax(a - b, axis=0).reshape(-1)[0]

    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    dtype_list = [jnp.float32, jnp.float64]
    for use_torch_vjp in [True, False]:
        for device in device_list:
            for dtype in dtype_list:
                for argnums in [(0,), (1,), (0, 1)]:
                    a = jax_randn(shape, dtype=dtype, device=device)
                    b = jax_randn(shape, dtype=dtype, device=device)
                    fn_jax = torch2jax_with_vjp(
                        fn, *tree_j2t((a, b)), depth=2, use_torch_vjp=use_torch_vjp
                    )

                    # no jit
                    J = jax.jacobian(fn_jax, argnums=argnums)(a, b)
                    J_expected = jax.jacobian(jax_fn, argnums=argnums)(a, b)
                    J_flat, J_expected_flat = tree_flatten(J)[0], tree_flatten(J_expected)[0]
                    err = sum(
                        jnp.linalg.norm(J_flat[i] - J_expected_flat[i]) for i in range(len(J_flat))
                    )
                    msg = f"device:{device} dtype:{dtype} use_torch_vjp:{use_torch_vjp}"
                    assert err < 1e-5, msg

                    # with jit
                    J = jax.jit(jax.jacobian(fn_jax, argnums=argnums))(a, b)
                    J_expected = jax.jit(jax.jacobian(jax_fn, argnums=argnums))(a, b)
                    J_flat, J_expected_flat = tree_flatten(J)[0], tree_flatten(J_expected)[0]
                    err = sum(
                        jnp.linalg.norm(J_flat[i] - J_expected_flat[i]) for i in range(len(J_flat))
                    )
                    msg = f"device:{device} dtype:{dtype} use_torch_vjp:{use_torch_vjp}"
                    assert err < 1e-5, msg


def test_hessian():
    shape = (2, 3)

    def fn(a, b):
        return torch.sin(a + 2 * b) * torch.softmax(a - b, dim=0).reshape(-1)[0]

    def jax_fn(a, b):
        return jnp.sin(a + 2 * b) * softmax(a - b, axis=0).reshape(-1)[0]

    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    dtype_list = [jnp.float32, jnp.float64]
    for use_torch_vjp in [True, False]:
        for device in device_list:
            for dtype in dtype_list:
                for argnums in [(0,), (1,), (0, 1)]:
                    a = jax_randn(shape, dtype=dtype, device=device)
                    b = jax_randn(shape, dtype=dtype, device=device)
                    fn_jax = torch2jax_with_vjp(
                        fn, *tree_j2t((a, b)), depth=2, use_torch_vjp=use_torch_vjp
                    )

                    # no jit
                    H = jax.jacobian(jax.jacobian(fn_jax, argnums=argnums))(a, b)
                    H_expected = jax.jacobian(jax.jacobian(jax_fn, argnums=argnums))(a, b)
                    H_flat, H_expected_flat = tree_flatten(H)[0], tree_flatten(H_expected)[0]
                    err = sum(
                        jnp.linalg.norm(H_flat[i] - H_expected_flat[i]) for i in range(len(H_flat))
                    )
                    msg = f"device:{device} dtype:{dtype} use_torch_vjp:{use_torch_vjp}"
                    assert err < 1e-5, msg

                    # with jit
                    H = jax.jit(jax.jacobian(jax.jacobian(fn_jax, argnums=argnums)))(a, b)
                    H_expected = jax.jit(jax.jacobian(jax.jacobian(jax_fn, argnums=argnums)))(a, b)
                    H_flat, H_expected_flat = tree_flatten(H)[0], tree_flatten(H_expected)[0]
                    err = sum(
                        jnp.linalg.norm(H_flat[i] - H_expected_flat[i]) for i in range(len(H_flat))
                    )
                    msg = f"device:{device} dtype:{dtype} use_torch_vjp:{use_torch_vjp}"
                    assert err < 1e-5, msg


if __name__ == "__main__":
    test_torch2jax_with_vjp()
    test_jacobian()
    test_hessian()
