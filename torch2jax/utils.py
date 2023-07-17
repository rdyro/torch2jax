"""Various utility functions."""

from __future__ import annotations

import random
from typing import Any
from types import ModuleType

import torch
from torch import Tensor
from jax import numpy as jnp, Array
from jax.core import ShapedArray
from jax.tree_util import tree_flatten, tree_map


def find_unique_id() -> int:
    while True:
        id = random.randint(0, 2**63)
        if not hasattr(torch, f"_torch2jax_fn_{id}") and not hasattr(
            torch, f"_torch2jax_args_{id}"
        ):
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
        torch.float16: jnp.float16,
        torch.uint8: jnp.uint8,
        torch.int8: jnp.int8,
        torch.int16: jnp.int16,
        torch.short: jnp.int16,
        torch.int32: jnp.int32,
        torch.int: jnp.int32,
        torch.int64: jnp.int64,
        torch.long: jnp.int64,
        torch.bool: bool,
    }[dtype]


def dtype_j2t(dtype: jnp.dtype) -> torch.dtype:
    """Translate jax dtype to torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype == bool:
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
    elif dtype == jnp.float32:
        return torch.float32
    elif dtype == jnp.float64:
        return torch.float64
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))


def dtype_j2m(cpp_module: ModuleType, dtype: jnp.dtype) -> int:
    """Translate jax dtype to integer denoting dtype in the torch2jax cpp extension module."""
    if dtype == bool:
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
    elif dtype == jnp.float32:
        return cpp_module.DATA_TYPE_FLOAT32
    elif dtype == jnp.float64:
        return cpp_module.DATA_TYPE_FLOAT64
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))


####################################################################################################


def _is_floating_point(x: Tensor | Array) -> bool:
    return x.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        jnp.float16,
        jnp.float32,
        jnp.float64,
    )


def guess_float_type(args: list[Array | Tensor]) -> jnp.dtype:
    float_type = None
    msg = (
        "You appear to have provided mixed precision arguments to a function. "
        + "We cannot guess the output dtype."
    )
    for arg in tree_flatten(args)[0]:
        if has_assoc_dtype(arg) and _is_floating_point(arg):
            assert float_type is None or dtype_t2j(arg.dtype) == float_type, msg
            if float_type is None:
                float_type = dtype_t2j(arg.dtype)
    if float_type is None:
        raise ValueError("We cannot guess the output dtype because no inputs are floating point.")
    return float_type


def has_assoc_dtype(x: Any) -> bool:
    return hasattr(x, "dtype")


def normalize_shapes(shapes: Any, extra_args: Any | None = None) -> Any:
    if not all(has_assoc_dtype(shape) for shape in tree_flatten(shapes)[0]):
        default_dtype = guess_float_type((shapes, extra_args))
    else:
        default_dtype = None
    return tree_map(
        lambda x: ShapedArray(x.shape, dtype_t2j(x.dtype))
        if has_assoc_dtype(x)
        else ShapedArray(x, default_dtype),
        shapes,
    )


####################################################################################################
