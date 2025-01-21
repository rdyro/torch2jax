"""Various utility functions."""

from __future__ import annotations

import random
from typing import Any
from types import ModuleType
from warnings import warn
from functools import lru_cache

import torch
from torch import Tensor
import jax
from jax import ShapeDtypeStruct
from jax import numpy as jnp, Array


def find_unique_id() -> int:
    while True:
        id = random.randint(0, 2**63)
        if not hasattr(torch, f"_torch2jax_fn_{id}") and not hasattr(torch, f"_torch2jax_args_{id}"):
            return id


def dtype_t2j(dtype: torch.dtype) -> jnp.dtype:
    """Translate torch dtype to jax dtype."""
    try:
        return jnp.dtype(dtype)
    except TypeError:
        pass
    return {
        torch.float32: jnp.float32,
        torch.float: jnp.float32,
        torch.float64: jnp.float64,
        torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.uint8: jnp.uint8,
        torch.int8: jnp.int8,
        torch.int16: jnp.int16,
        torch.short: jnp.int16,
        torch.int32: jnp.int32,
        torch.int: jnp.int32,
        torch.int64: jnp.int64,
        torch.long: jnp.int64,
        torch.bool: jnp.bool,
    }[dtype]


def dtype_j2t(dtype: jnp.dtype) -> torch.dtype:
    """Translate jax dtype to torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype == jnp.bool:
        return torch.bool
    elif dtype == jnp.uint8:
        return torch.uint8
    elif dtype == jnp.int8:
        return torch.int8
    elif dtype == jnp.int16:
        return torch.int16
    elif dtype == jnp.int32:
        return torch.int32
    elif dtype == jnp.int64:
        return torch.int64
    elif dtype == jnp.float16:
        return torch.float16
    elif dtype == jnp.bfloat16:
        return torch.bfloat16
    elif dtype == jnp.float32:
        return torch.float32
    elif dtype == jnp.float64:
        return torch.float64
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))


def dtype_j2m(cpp_module: ModuleType, dtype: jnp.dtype) -> int:
    """Translate jax dtype to integer denoting dtype in the torch2jax cpp extension module."""
    if dtype == jnp.bool:
        return cpp_module.DATA_TYPE_BOOL
    elif dtype == jnp.uint8:
        return cpp_module.DATA_TYPE_UINT8
    elif dtype == jnp.int8:
        return cpp_module.DATA_TYPE_INT8
    elif dtype == jnp.int16:
        return cpp_module.DATA_TYPE_INT16
    elif dtype == jnp.int32:
        return cpp_module.DATA_TYPE_INT32
    elif dtype == jnp.int64:
        return cpp_module.DATA_TYPE_INT64
    elif dtype == jnp.float16:
        return cpp_module.DATA_TYPE_FLOAT16
    elif dtype == jnp.bfloat16:
        return cpp_module.DATA_TYPE_BFLOAT16
    elif dtype == jnp.float32:
        return cpp_module.DATA_TYPE_FLOAT32
    elif dtype == jnp.float64:
        return cpp_module.DATA_TYPE_FLOAT64
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))


####################################################################################################


def _is_floating(x: Tensor | Array) -> bool:
    return jnp.issubdtype(dtype_t2j(x.dtype), jnp.floating)


def guess_float_type(args: list[Array | Tensor]) -> jnp.dtype:
    float_type = None
    msg = "You appear to have provided mixed precision arguments to a function. " + "We cannot guess the output dtype."
    for arg in jax.tree.leaves(args):
        if hasattr(arg, "dtype") and _is_floating(arg):
            assert float_type is None or dtype_t2j(arg.dtype) == float_type, msg
            if float_type is None:
                float_type = dtype_t2j(arg.dtype)
    if float_type is None:
        raise ValueError("We cannot guess the output dtype because no inputs are floating point.")
    return float_type


def is_shape_desc(x):
    return isinstance(x, (list, tuple)) and all(isinstance(y, int) for y in x)


def normalize_shapes(shapes: Any, extra_args: Any | None = None) -> Any:
    if not all(hasattr(shape, "dtype") for shape in jax.tree.flatten(shapes)[0]):
        default_dtype = guess_float_type((shapes, extra_args))
    else:
        default_dtype = None
    return jax.tree.map(
        lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype))
        if hasattr(x, "dtype")
        else ShapeDtypeStruct(x, default_dtype),
        shapes,
        is_leaf=is_shape_desc,
    )


@lru_cache
def warn_once(msg, torch_fn):
    del torch_fn  # used for proper hashing of context for lru_cache
    warn(msg)


####################################################################################################
