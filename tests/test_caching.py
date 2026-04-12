import sys
import warnings
from pathlib import Path

from absl.testing import parameterized, absltest
import torch
import jax
from jax import numpy as jnp

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax, torch2jax_with_vjp, tree_t2j  # noqa: E402
from torch2jax.api import _SHAPE_CHANGE_WARN_CONCRETE, _SHAPE_CHANGE_WARN_EXPLICIT  # noqa: E402

####################################################################################################


class TestCaching(parameterized.TestCase):
    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_torch2jax_caching(self, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("Skipping CUDA test when CUDA is not available.")

        def torch_fn(x, y):
            return x @ y

        x = torch.randn(10, 5)
        y = torch.randn(5, 3)

        jax_fn = torch2jax(torch_fn, x, y)

        # Original shape
        xj, yj = tree_t2j((x, y))
        out1 = jax_fn(xj, yj)
        assert out1.shape == (10, 3)

        # New shape
        x2 = torch.randn(20, 5)
        y2 = torch.randn(5, 7)
        xj2, yj2 = tree_t2j((x2, y2))

        out2 = jax_fn(xj2, yj2)
        assert out2.shape == (20, 7)

    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_torch2jax_with_vjp_caching(self, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("Skipping CUDA test when CUDA is not available.")

        def torch_fn(x, y):
            return x @ y

        x = torch.randn(10, 5)
        y = torch.randn(5, 3)

        jax_fn = torch2jax_with_vjp(torch_fn, x, y)

        xj, yj = tree_t2j((x, y))

        @jax.jit
        def f(x, y):
            return jnp.sum(jax_fn(x, y))

        g_fn = jax.jit(jax.grad(f, argnums=(0, 1)))

        g1_x, g1_y = g_fn(xj, yj)
        assert g1_x.shape == (10, 5)
        assert g1_y.shape == (5, 3)

        x2 = torch.randn(20, 5)
        y2 = torch.randn(5, 7)
        xj2, yj2 = tree_t2j((x2, y2))

        g2_x, g2_y = g_fn(xj2, yj2)
        assert g2_x.shape == (20, 5)
        assert g2_y.shape == (5, 7)


class TestCachingWarnings(parameterized.TestCase):
    def test_warns_concrete_inputs(self):
        torch_fn = lambda x, y: x @ y
        jax_fn = torch2jax(torch_fn, torch.randn(4, 3), torch.randn(3, 2))
        xj, yj = tree_t2j((torch.randn(8, 3), torch.randn(3, 5)))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jax_fn(xj, yj)
        msgs = [str(wi.message) for wi in w]
        assert any(_SHAPE_CHANGE_WARN_CONCRETE[:-20] in m for m in msgs), f"Expected concrete warning, got: {msgs}"

    def test_warns_explicit_output_shapes(self):
        torch_fn = lambda x, y: x @ y
        x, y = torch.randn(4, 3), torch.randn(3, 2)
        jax_fn = torch2jax(torch_fn, x, y, output_shapes=jax.ShapeDtypeStruct((4, 2), jnp.float32))
        xj, yj = tree_t2j((torch.randn(8, 3), torch.randn(3, 5)))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jax_fn(xj, yj)
        msgs = [str(wi.message) for wi in w]
        assert any(_SHAPE_CHANGE_WARN_EXPLICIT[:-20] in m for m in msgs), f"Expected explicit warning, got: {msgs}"

    def test_no_warning_on_same_shape(self):
        torch_fn = lambda x, y: x @ y
        x, y = torch.randn(4, 3), torch.randn(3, 2)
        jax_fn = torch2jax(torch_fn, x, y)
        xj, yj = tree_t2j((x, y))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jax_fn(xj, yj)
        msgs = [str(wi.message) for wi in w]
        assert not any("input shapes changed" in m for m in msgs), f"Unexpected warning: {msgs}"


####################################################################################################

if __name__ == "__main__":
    absltest.main()
