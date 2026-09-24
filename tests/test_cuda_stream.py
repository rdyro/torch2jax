from absl.testing import absltest, parameterized
import torch
import jax
from jax import numpy as jnp

from torch2jax import torch2jax


CUDA_AVAILABLE = torch.cuda.is_available() and jax.default_backend() == "gpu"


class TestCudaStreamOrdering(parameterized.TestCase):
    @parameterized.product(depth=[0, 2])
    def test_interleaved_jax_torch_chain(self, depth):
        # torch runs on XLA's stream without device syncs, so producer -> torch -> consumer ordering must still hold
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA not available")
        x = jax.device_put(jax.random.normal(jax.random.key(0), (2048, 2048)), jax.devices("cuda")[0])
        torch_step = torch2jax(lambda a: torch.tanh(a @ a * 1e-3) + a, x, depth=depth)
        jax_step = lambda a: jnp.tanh(a @ a * 1e-3) + a

        def chain(step):
            def f(x):
                for _ in range(6):
                    x = jnp.sin(step(jnp.cos(x) @ x * 1e-3))
                return jnp.sum(x**2)

            return f

        with jax.default_matmul_precision("highest"):  # match torch's full fp32 matmuls
            for _ in range(3):
                y, y_ref = jax.jit(chain(torch_step))(x), jax.jit(chain(jax_step))(x)
                self.assertTrue(jnp.allclose(y, y_ref, rtol=1e-4), (y, y_ref))
            if depth > 0:
                g, g_ref = jax.jit(jax.grad(chain(torch_step)))(x), jax.jit(jax.grad(chain(jax_step)))(x)
                self.assertLess(float(jnp.linalg.norm(g - g_ref) / jnp.linalg.norm(g_ref)), 1e-4)

    def _slow_fn(self):  # returns a fn whose value `v` is available only after a chain of large matmuls
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA not available")
        A = torch.randn(4096, 4096, device="cuda")

        def slow(v):
            B = A
            for _ in range(20):
                B = B @ A * 1e-3
            return B[0, :1024] * 0 + v

        return slow

    def test_sees_prior_torch_work(self):
        # torch work queued on torch's stream before the call (e.g., a weight update) is visible to the torch fn
        slow, W = self._slow_fn(), torch.zeros(1024, device="cuda")
        fn = jax.jit(torch2jax(lambda x: x + W, torch.zeros(1024, device="cuda"), depth=0))
        jax.block_until_ready(fn(jnp.zeros(1024)))
        for i in range(1, 6):
            torch.cuda.synchronize()
            W.copy_(slow(i))
            self.assertEqual(float(fn(jnp.zeros(1024))[0]), i)

    def test_later_torch_work_sees_state(self):
        # state the torch fn modifies is visible to torch work queued after the call, without block_until_ready
        slow, buf = self._slow_fn(), torch.zeros(1024, device="cuda")
        fn = jax.jit(torch2jax(lambda x: (buf.copy_(slow(x[0])), x)[1], torch.zeros(1024, device="cuda"), depth=0))
        jax.block_until_ready(fn(jnp.zeros(1024)))
        for i in range(1, 6):
            torch.cuda.synchronize()
            fn(jnp.full(1024, float(i)))
            self.assertEqual(float((buf * 1).sum().item()), 1024 * i)

if __name__ == "__main__":
    absltest.main()
