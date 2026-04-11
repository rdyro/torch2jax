import time
import sys
from subprocess import check_call

from absl.testing import absltest

from torch2jax import compile_and_import_module  # noqa: E402


class CompilationTest(absltest.TestCase):
    def _test_compilation(self):
        cpp_module = compile_and_import_module()
        assert cpp_module is not None

    def _test_forced_compilation(self):
        print("testing forced compilation")
        cpp_module = compile_and_import_module(force_recompile=True)
        assert cpp_module is not None

    def _test_compilation_caching(self):
        check_call(
            [sys.executable, "-c", "from torch2jax import compile_and_import_module; compile_and_import_module()"]
        )

        t = time.time()
        check_call(
            [sys.executable, "-c", "from torch2jax import compile_and_import_module; compile_and_import_module()"]
        )
        t = time.time() - t
        assert t < 10.0

    def test_ordered(self):
        self._test_compilation()
        self._test_forced_compilation()
        self._test_compilation_caching()


if __name__ == "__main__":
    absltest.main()
