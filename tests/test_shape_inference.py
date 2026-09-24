import warnings

from absl.testing import absltest, parameterized
import torch
import jax
from jax import numpy as jnp

from torch2jax import torch2jax, torch2jax_auto, tree_t2j, j2t


CUDA_AVAILABLE = torch.cuda.is_available() and jax.default_backend() == "gpu"


class TestShapeInference(parameterized.TestCase):
    def test_inference_runs_on_meta_without_grad(self):
        seen = []

        def torch_fn(x):
            seen.append((x.device.type, torch.is_grad_enabled()))
            return 2 * x

        torch2jax(torch_fn, torch.randn(10, 3), depth=0)
        torch2jax(torch_fn, torch.randn(10, 3), depth=2)
        self.assertEqual(seen, [("meta", False)] * 2)

    @parameterized.product(depth=[0, 2])
    def test_shape_change_with_device_resident_weights(self, depth):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA not available")
        lin = torch.nn.Linear(3, 2).cuda().requires_grad_(False)
        torch_fn = lambda x: lin(x)
        jax_fn = torch2jax(torch_fn, torch.randn(10, 3, device="cuda"), depth=depth)
        xt = torch.randn(20, 3, device="cuda")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = jax_fn(tree_t2j(xt))
        torch.testing.assert_close(j2t(y), lin(xt))

    def test_auto_does_not_compute_on_zeros(self):
        seen = []

        def torch_fn(x):
            seen.append(x.device.type)
            return x.sum(0)

        y = torch2jax_auto(torch_fn, depth=0)(jnp.ones((4, 3)))
        self.assertEqual(seen, ["meta", "cpu" if jax.default_backend() == "cpu" else "cuda"])
        self.assertEqual(y.tolist(), [4.0] * 3)


if __name__ == "__main__":
    absltest.main()
