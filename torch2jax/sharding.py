"""Sharding support following the JAX explicit-sharding model.

- Explicit mesh axes: the torch function is opaque, so sharded inputs are never implicitly all-gathered. The call is
  instead wrapped in a `jax.shard_map` that is manual only over the explicit axes in use (in_specs are read from the
  input types), and the user states the per-shard semantics by providing `out_specs`.
- Manual mesh axes (inside `shard_map`): inputs are cast to a common set of varying manual axes (vma) and FFI outputs
  are marked as varying, so the call type-checks with `check_vma=True` and replicated inputs get psum-ed cotangents.
- Auto mesh axes are left to the XLA partitioner and are not specially supported, `out_specs` over them is an error.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
from jax.sharding import PartitionSpec as P, AxisType

_ERR_IMPLICIT_ALLGATHER = (
    "torch2jax: inputs are sharded along explicit mesh axes {}, but no `out_specs` was given. A torch function is"
    " opaque to JAX, so either (1) pass `out_specs=` to run the torch function per-shard (inside `jax.shard_map` with"
    " in_specs taken from the input shardings), or (2) replicate the inputs explicitly, e.g., with"
    " `jax.sharding.reshard(x, P())`, to call the torch function on the full arrays."
)
_ERR_AUTO_OUT_SPECS = (
    "torch2jax: `out_specs` uses Auto mesh axes {}, but a per-shard torch call needs Explicit mesh axes (with Auto"
    " axes, XLA would silently all-gather the inputs of the opaque torch function). Create the mesh with explicit axes,"
    " e.g., `jax.make_mesh(..., axis_types=(AxisType.Explicit,) * n)`, or call the function inside `jax.shard_map`."
)


def _vma(x) -> frozenset:
    aval = jax.typeof(x)
    vma = getattr(aval, "vma", None)  # older JAX
    return frozenset(vma if vma is not None else getattr(getattr(aval, "mat", None), "varying", ()))


def _pvary(x, axes: frozenset):
    missing = tuple(sorted(axes - _vma(x)))
    if not missing:
        return x
    return jax.lax.pcast(x, missing, to="varying") if hasattr(jax.lax, "pcast") else jax.lax.pvary(x, missing)


def _axes_of_type(mesh, axis_type: AxisType) -> set:
    return {ax for ax, t in zip(mesh.axis_names, mesh.axis_types) if t == axis_type}


def union_vma(xs: Any) -> frozenset:
    return frozenset().union(*map(_vma, jax.tree.leaves(xs)))


def match_vma(xs: Any, vma: frozenset | None = None) -> Any:
    """Cast all leaves of `xs` to be varying along `vma` (default: the union of their varying manual axes)."""
    vma = union_vma(xs) if vma is None else vma
    return xs if not vma else jax.tree.map(lambda x: _pvary(x, vma), xs)


def _entry_axes(entry) -> tuple:
    return () if entry is None else entry if isinstance(entry, tuple) else (entry,)


def _spec_axes(spec: P) -> set:
    return {ax for entry in spec for ax in _entry_axes(entry)}


def local_shapes(shapes: Any, out_specs: Any, mesh) -> Any:
    """Per-shard shapes of the global `shapes`, partitioned by `out_specs` (a prefix tree of `shapes`) over `mesh`."""

    def local(x, spec):
        n = [math.prod(mesh.shape[ax] for ax in _entry_axes(entry)) for entry in spec]
        n += [1] * (len(x.shape) - len(n))
        if len(n) > len(x.shape) or any(d % k for d, k in zip(x.shape, n)):
            raise ValueError(f"torch2jax: output shape {tuple(x.shape)} cannot be partitioned by {spec} over {mesh}")
        return jax.ShapeDtypeStruct(tuple(d // k for d, k in zip(x.shape, n)), x.dtype)

    return jax.tree.map(local, shapes, jax.tree.broadcast(out_specs, shapes, is_leaf=lambda s: isinstance(s, P)))


def _explicit_sharding(xs: Any) -> tuple[Any, list[P], set]:
    """The mesh, the per-leaf partition specs and the explicit mesh axes the leaves of `xs` are sharded along."""
    mesh, specs, axes = jax.sharding.get_abstract_mesh(), [], set()
    for x in jax.tree.leaves(xs):
        sharding = getattr(jax.typeof(x), "sharding", None)
        spec = getattr(sharding, "spec", None)
        specs.append(spec if spec is not None else P())
        if spec is None or sharding.mesh.empty:
            continue
        explicit = _spec_axes(spec) & _axes_of_type(sharding.mesh, AxisType.Explicit)
        if explicit:
            mesh, axes = sharding.mesh, axes | explicit
    return mesh, specs, axes


def shard_call(call: Callable, args: Any, out_specs: Any | None) -> Any:
    """Call `call(args, to_local)`, in a `shard_map` over the explicit mesh axes that `args` are sharded along.

    `to_local` maps global output shapes to per-shard ones inside the `shard_map` and is None outside of it.
    """
    mesh, specs, axes = _explicit_sharding(args)
    if out_specs is not None and not mesh.empty:
        out_axes = set().union(*map(_spec_axes, jax.tree.leaves(out_specs, is_leaf=lambda s: isinstance(s, P))))
        if auto_axes := out_axes & _axes_of_type(mesh, AxisType.Auto):
            raise ValueError(_ERR_AUTO_OUT_SPECS.format(sorted(auto_axes)))
        axes |= out_axes & _axes_of_type(mesh, AxisType.Explicit)
    if not axes:
        return call(args, None)
    if out_specs is None:
        raise ValueError(_ERR_IMPLICIT_ALLGATHER.format(sorted(axes)))
    leaves, struct = jax.tree.flatten(args)
    to_local = lambda shapes: local_shapes(shapes, out_specs, mesh)
    body = lambda *leaves: call(jax.tree.unflatten(struct, leaves), to_local)
    return jax.shard_map(body, mesh=mesh, in_specs=tuple(specs), out_specs=out_specs, axis_names=axes)(*leaves)
