import os
import subprocess
import sys
import textwrap
from pathlib import Path

from absl.testing import absltest, parameterized
import torch


class TestX64Disabled(parameterized.TestCase):
    @parameterized.product(device=["cpu", "cuda"])
    def test_int64_example_args_without_x64(self, device):
        # x64 must be off from process start, so run in a fresh interpreter
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        script = textwrap.dedent(f"""
            import warnings
            import jax, torch
            from torch2jax import torch2jax, tree_t2j
            assert not jax.config.jax_enable_x64
            torch_fn = lambda x, y: torch.nn.CrossEntropyLoss()(x, y)
            xt, yt = torch.randn(10, 5, device="{device}"), torch.randint(0, 5, (10,), device="{device}")
            jax_fn = torch2jax(torch_fn, xt, yt)
            x, y = tree_t2j((xt, yt))
            assert y.dtype == jax.numpy.int32
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="torch2jax")
                out, g = jax.value_and_grad(jax_fn)(x, y)
                out_jit = jax.jit(jax_fn)(x, y)
            expected = torch_fn(xt, yt).item()
            assert abs(out - expected) < 1e-5 and abs(out_jit - expected) < 1e-5, (out, out_jit, expected)
            assert g.shape == x.shape
        """)
        root = str(Path(__file__).absolute().parents[1])  # no preallocation, the parent process may hold GPU memory
        env = dict(os.environ, JAX_ENABLE_X64="0", PYTHONPATH=root, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])


if __name__ == "__main__":
    absltest.main()
