"""Various utility functions."""

from __future__ import annotations

import random
from typing import Any
from types import ModuleType
import warnings
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


_T2J = {
    torch.bool: jnp.bool_,
    torch.uint8: jnp.uint8,
    torch.uint16: jnp.uint16,
    torch.uint32: jnp.uint32,
    torch.uint64: jnp.uint64,
    torch.int8: jnp.int8,
    torch.int16: jnp.int16,
    torch.int32: jnp.int32,
    torch.int64: jnp.int64,
    torch.float16: jnp.float16,
    torch.bfloat16: jnp.bfloat16,
    torch.float32: jnp.float32,
    torch.float64: jnp.float64,
    torch.complex64: jnp.complex64,
    torch.complex128: jnp.complex128,
    torch.float8_e4m3fn: jnp.float8_e4m3fn,
    torch.float8_e5m2: jnp.float8_e5m2,
    torch.float8_e4m3fnuz: jnp.float8_e4m3fnuz,
    torch.float8_e5m2fnuz: jnp.float8_e5m2fnuz,
}
_J2T = {jnp.dtype(v): k for k, v in _T2J.items()}


def dtype_t2j(dtype: torch.dtype) -> jnp.dtype:
    """Translate torch dtype to jax dtype."""
    try:
        return jnp.dtype(_T2J[dtype] if isinstance(dtype, torch.dtype) else dtype)
    except (KeyError, TypeError):
        raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_j2t(dtype: jnp.dtype) -> torch.dtype:
    """Translate jax dtype to torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _J2T[jnp.dtype(dtype)]
    except (KeyError, TypeError):
        raise ValueError(f"Unsupported dtype: {dtype}")


def canonical_dtype(dtype) -> jnp.dtype:
    """The dtype JAX actually uses for `dtype` (torch or jax) under the current x64 setting."""
    return jax.dtypes.canonicalize_dtype(dtype_t2j(dtype))


def shape_key(xs: Any) -> tuple:
    """Hashable (shape, canonical dtype) key of all leaves of `xs`, used for shape-change caching."""
    avals = [x if hasattr(x, "shape") and hasattr(x, "dtype") else jax.typeof(x) for x in jax.tree.leaves(xs)]
    return tuple((tuple(x.shape), canonical_dtype(x.dtype)) for x in avals)


def torch_dtype_like(dtype, like: torch.dtype | None = None) -> torch.dtype:
    """Torch dtype for `dtype`, preferring `like` (e.g., int64 for int32 under disabled x64) if they are equivalent."""
    return like if like is not None and canonical_dtype(like) == canonical_dtype(dtype) else dtype_j2t(dtype)


def default_torch_device() -> torch.device:
    return torch.device("cuda") if jax.default_backend() == "gpu" and torch.cuda.is_available() else torch.device("cpu")


def placeholder_like(x: Any, torch_dtype: torch.dtype | None = None) -> Tensor:
    """A meta-device (no memory) torch tensor with the shape and (torch-preferred) dtype of `x`."""
    return torch.empty(tuple(x.shape), dtype=torch_dtype_like(x.dtype, torch_dtype), device="meta")


def infer_outputs(fn, args: Any, kw: dict | None = None) -> Any:
    """Run `fn` to discover its outputs: on meta tensors when possible, otherwise on concrete tensors.

    Concrete example tensors are used as-is in the fallback; placeholders become zeros on the default device.
    """
    is_array = lambda x: not isinstance(x, Tensor) and hasattr(x, "shape") and hasattr(x, "dtype")
    args, kw = jax.tree.map(lambda x: placeholder_like(x) if is_array(x) else x, (args, {} if kw is None else kw))
    is_concrete = lambda x: isinstance(x, Tensor) and not x.is_meta
    with torch.no_grad():
        try:
            meta_args, meta_kw = jax.tree.map(lambda x: x.to("meta") if is_concrete(x) else x, (args, kw))
            return fn(*meta_args, **meta_kw)
        except Exception:  # e.g., the function mixes inputs with device-resident weights or is data-dependent
            device = default_torch_device()
            args, kw = jax.tree.map(lambda x: torch.zeros_like(x, device=device) if x.is_meta else x, (args, kw))
            return fn(*args, **kw)


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

_WARN_MIXED_PRECISION = (
    "You appear to have provided mixed precision arguments to a function. We cannot guess the output dtype."
)


def _is_floating(x: Tensor | Array) -> bool:
    return jnp.issubdtype(dtype_t2j(x.dtype), jnp.floating)


def guess_float_type(args: list[Array | Tensor]) -> jnp.dtype:
    float_type = None
    for arg in jax.tree.leaves(args):
        if hasattr(arg, "dtype") and _is_floating(arg):
            assert float_type is None or dtype_t2j(arg.dtype) == float_type, _WARN_MIXED_PRECISION
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
        lambda x: (
            ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)) if hasattr(x, "dtype") else ShapeDtypeStruct(x, default_dtype)
        ),
        shapes,
        is_leaf=is_shape_desc,
    )


@lru_cache
def warn_once(msg, torch_fn):
    del torch_fn  # used for proper hashing of context for lru_cache
    warnings.warn(msg)


@lru_cache
def warn_always(msg):
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(msg)


####################################################################################################
