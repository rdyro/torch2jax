import jax

from .utils import placeholder_like
from .gradients import torch2jax


def torch2jax_auto(torch_fn, output_shapes=None, **t2j_kwargs):
    """
    Auto-wraps a PyTorch function for JAX without requiring example arguments ahead of time.
    """
    root_fn = None

    def wrapper(*args):
        nonlocal root_fn
        if root_fn is None:
            # on first run, define a root_fn
            # on shape changes, the fn itself will re-create another version
            dummy_args = jax.tree.map(lambda a: placeholder_like(jax.typeof(a)), args)
            root_fn = torch2jax(torch_fn, *dummy_args, output_shapes=output_shapes, **t2j_kwargs)

        return root_fn(*args)

    return wrapper
