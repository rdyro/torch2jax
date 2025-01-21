import time
import sys
import os
from pathlib import Path
from subprocess import check_call

from absl.testing import absltest

root_path = Path(__file__).parents[1]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

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
        os.chdir(root_path)

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
