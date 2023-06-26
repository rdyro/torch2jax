import sys
from pathlib import Path

root_path = Path(__file__).parents[1]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from torch2jax import torch2jax, compile_and_import_module

def test_compilation():
    cpp_module = compile_and_import_module()
    assert cpp_module is not None