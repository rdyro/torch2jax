# Changelog

- unreleased
  - **breaking**: sharding follows JAX's explicit sharding model: inputs sharded along
    explicit mesh axes are never implicitly all-gathered, pass `out_specs=` to call the
    torch function per-shard (inside `jax.shard_map`), `out_specs` also works with gradients
    and is an error over Auto mesh axes (the `jax.make_mesh` default in older JAX);
    `output_sharding_spec` is a deprecated alias, `custom_partitioning` (and the global
    switch to the GSPMD partitioner) was removed
  - gradients inside `jax.shard_map` work with `check_vma=True`, cotangents of replicated
    inputs are `psum`-ed automatically
  - **breaking**: torch outputs are validated against `output_shapes`, a shape or dtype
    mismatch is an error (previously silently broadcast/cast), unsupported dtypes raise
    instead of aborting; added complex, uint16/32/64 and float8 dtypes
  - fixed int64 inputs (e.g., class labels) when JAX x64 is disabled
  - output shapes are inferred on the meta device (no compute) with a fallback to real tensors
  - the torch computation is enqueued on XLA's CUDA stream instead of synchronizing the device,
    ordered with torch's own stream by CUDA events (prior torch work, e.g., weight updates, is
    visible to the torch function, and later torch work sees the state it modified)
  - the C++ extension is rebuilt when its sources change
  - with `out_specs`, a global `output_shapes` is split per-shard, the torch function is not
    run to infer per-shard output shapes
  - the `torch.autograd.grad` VJP fallback is used whenever `torch.func.vjp` fails (e.g.,
    `.numpy()` in the function), the original error is raised if the fallback fails too
  - fixed `t2j` of CUDA tensors on multi-GPU hosts when another GPU is the current device
  - on multiple devices the torch function is called concurrently (one thread per device),
    stateful torch code, e.g., `torch.func.functional_call` on a shared module, must be guarded
  - `jax.pmap` works (with recent JAX)
  - `lock=True` (or a lock object) guards torch calls, which run concurrently per device

- version 0.8.0
  - **breaking**: `torch2jax` now defines gradients by default (`depth=2`), unifying
    the old `torch2jax` (forward-only) and `torch2jax_with_vjp` (with gradients)
  - `torch2jax_with_vjp` is deprecated — use `torch2jax` instead
  - use `depth=0` for the old forward-only behavior
  - `torch2jax_without_vjp` is the public API for sharding (`output_sharding_spec`)
    and keyword arguments (`example_kw`)

- version 0.7.2
  - wrapped functions now automatically cache for different input shapes — no need
    to re-wrap when calling with new shapes
  - a warning is emitted on the first shape change to inform the user

- version 0.6.1
  - added `vmap_method=` support for experimental pytorch-side batching support,
    see [https://github.com/rdyro/torch2jax/issues/28](https://github.com/rdyro/torch2jax/issues/28)

- version 0.6.0
  - proper multi-GPU support mostly with `shard_map` but also via `jax.jit` automatic sharding
  - `shard_map` and automatic `jax.jit` device parallelization should work, but `pmap` doesn't work
  - removed (deprecated)
    - torch2jax_flat - use the more flexible torch2jax
  - added input shapes validation - routines

- version 0.5.0
  - updating to the new JAX ffi interface

- version 0.4.11
  - compilation fixes and support for newer JAX versions

- version 0.4.10
  - support for multiple GPUs, currently, all arguments must and the output
    must be on the same GPU (but you can call the wrapped function with
    different GPUs in separate calls)
  - fixed the coming depreciation in JAX deprecating `.device()` for
    `.devices()`

- no version change
  - added helper script `install_package_aliased.py` to automatically install
    the package with a different name (to avoid a name conflict)

- version 0.4.7
  - support for newest JAX (0.4.17) with backwards compatibility maintained
  - compilation now delegated to python version subfolders for multi-python systems

- version 0.4.6
  - bug-fix: cuda stream is now synchronized before and after a torch call explicitly to
    avoid reading unwritten data

- version 0.4.5
  - `torch2jax_with_vjp` now automatically selects `use_torch_vjp=False` if the `True` fails
  - bug-fix: cuda stream is now synchronized after a torch call explicitly to
    avoid reading unwritten data

- version 0.4.4
  - introduced a `use_torch_vjp` (defaulting to True) flag in `torch2jax_with_vjp` which
    can be set to False to use the old `torch.autograd.grad` for taking
    gradients, it is the slower method, but is more compatible

- version 0.4.3
  - added a note in README about specifying input/output structure without instantiating data

- version 0.4.2
  - added `examples/input_output_specification.ipynb` showing how input/output
  structure can be specified

- version 0.4.1
  - bug-fix: in `torch2jax_with_vjp`, nondiff arguments were erroneously memorized

- version 0.4.0
  - added batching (vmap support) using `torch.vmap`, this makes `jax.jacobian` work
  - robustified support for gradients
  - added mixed type arguments, including support for float16, float32, float64 and integer types
  - removed unnecessary torch function calls in defining gradients
  - added an example of wrapping a BERT model in JAX (with weights modified from JAX), `examples/bert_from_jax.ipynb`

- version 0.3.0
  - added a beta-version of a new wrapping method `torch2jax_with_vjp` which
  allows recursively defining reverse-mode gradients for the wrapped torch
  function that works in JAX both normally and under JIT

- version 0.2.0
  - arbitrary input and output structure is now allowed
  - removed the restriction on the number of arguments or their maximum dimension
  - old interface is available via `torch2jax.compat.torch2jax`

- version 0.1.2
  - full CPU only version support, selected via `torch.cuda.is_available()`
  - bug-fix: compilation should now cache properly

- version 0.1.1
  - bug-fix: functions do not get overwritten, manual fn id parameter replaced with automatic id generation
  - compilation caching is now better

- version 0.1.0
  - first working version of the package
