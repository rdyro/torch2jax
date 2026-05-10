import subprocess
import sys
import textwrap

from absl.testing import absltest


class FreeThreadingTest(absltest.TestCase):
    def test_extension_load_does_not_reenable_gil(self):
        # sys._is_gil_enabled() only exists on free-threading Python (3.13t+); on a regular GIL
        # build the GIL is always on and there is nothing to assert.
        if not hasattr(sys, "_is_gil_enabled"):
            self.skipTest("free-threading Python required")
        # Check in a fresh subprocess: any other import in this process may have already
        # re-enabled the GIL globally, and `import torch2jax` alone does not load the C++
        # extension (compilation/import is lazy), so an in-process check would be meaningless.
        script = textwrap.dedent("""
            import sys
            from torch2jax import compile_and_import_module
            compile_and_import_module()  # actually loads the C++ extension
            assert not sys._is_gil_enabled(), "loading torch2jax re-enabled the GIL"
        """)
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(
            proc.returncode,
            0,
            "loading the torch2jax C++ extension re-enabled the GIL; ensure it is built with "
            f"py::mod_gil_not_used().\n{proc.stderr}",
        )


if __name__ == "__main__":
    absltest.main()
