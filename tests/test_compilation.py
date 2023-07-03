import time
import sys
import os
from pathlib import Path
from subprocess import check_call

root_path = Path(__file__).parents[1]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from torch2jax import torch2jax, compile_and_import_module


def _test_compilation():
    cpp_module = compile_and_import_module()
    assert cpp_module is not None

def _test_forced_compilation():
    cpp_module = compile_and_import_module(force_recompile=True)
    assert cpp_module is not None


def _test_compilation_caching():
    os.chdir(root_path)

    check_call(
        [
            sys.executable,
            "-c",
            "from torch2jax import compile_and_import_module; compile_and_import_module()",
        ]
    )

    t = time.time()
    check_call(
        [
            sys.executable,
            "-c",
            "from torch2jax import compile_and_import_module; compile_and_import_module()",
        ]
    )
    t = time.time() - t
    assert t < 10.0


def test_ordered():
    _test_forced_compilation()
    _test_compilation()
    _test_compilation_caching()