# torch2jax

[Documentation](https://rdyro.github.io/torch2jax/)

<a href="https://rdyro.github.io/torch2jax/">
<p align="center">
<img src="https://raw.githubusercontent.com/rdyro/torch2jax/main/images/torch2jax_logo2.png" width="70%" style="max-width:800px;width:70%;display:block;margin-left:auto;margin-right:auto"/>
</p>
</a>
<br />


This package is designed to facilitate no-copy PyTorch calling from JAX under
both eager execution and JIT. It leverages the JAX C++ extension interface,
enabling operations on both CPU and GPU platforms. Moreover, it allows for
executing arbitrary PyTorch code from JAX under eager execution and JIT.

The intended application is efficiently running existing PyTorch code (like ML
models) in JAX applications with very low overhead.

`torch2jax` also runs PyTorch code on **multiple devices**: on sharded JAX arrays
across multiple GPUs, under `jax.jit` and with gradients. The torch function runs
concurrently on each device's shard, without hidden all-gathers or device
synchronization, see [Multi-device (multi-GPU) support](#multi-device-multi-gpu-support).

This project was inspired by the jax2torch repository
[https://github.com/lucidrains/jax2torch](https://github.com/lucidrains/jax2torch)
and has been made possible due to an amazing tutorial on extending JAX
[https://github.com/dfm/extending-jax](https://github.com/dfm/extending-jax).
Comprehensive JAX documentation
[https://github.com/google/jax](https://github.com/google/jax) also
significantly contributed to making this work easier.

Although I am unsure this functionality could be achieved without C++/CUDA, the
C++ compilation is efficiently done using PyTorch's portable CUDA & C++
compilation features, requiring minimal configuration.

# Install

```bash
$ pip install git+https://github.com/rdyro/torch2jax.git
```

`torch2jax` is now available on PyPI under the alias `wrap_torch2jax`:

```bash
$ pip install wrap-torch2jax
$ # then
$ python3
$ >>> from wrap_torch2jax import torch2jax
```

# Usage

`torch2jax` is the main entry point. By default it defines gradients (VJP rules
up to `depth=2`), so `jax.grad` works out of the box.

```python
import torch
import jax
from jax import numpy as jnp
from torch2jax import torch2jax
from torch2jax import Size, dtype_t2j

def torch_fn(a, b):
    return a + b

shape = (10, 2)
a, b = torch.randn(shape), torch.randn(shape)

# without output_shapes, torch_fn **will be evaluated once** to infer outputs
jax_fn = torch2jax(torch_fn, a, b)

# with output_shapes, torch_fn will NOT be evaluated
jax_fn = torch2jax(torch_fn, a, b, output_shapes=Size(a.shape))

# you can specify the whole input and output structure without instantiating the tensors
jax_fn = torch2jax(
    torch_fn,
    jax.ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
    jax.ShapeDtypeStruct(b.shape, dtype_t2j(b.dtype)),
    output_shapes=jax.ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
)

key = jax.random.key(0)
device = jax.devices("cuda")[0]  # both CPU and CUDA are supported
a = jax.device_put(jax.random.normal(key, shape), device)
b = jax.device_put(jax.random.normal(key, shape), device)

# call the no-copy torch function
out = jax_fn(a, b)

# call the no-copy torch function **under JIT**
out = jax.jit(jax_fn)(a, b)

# gradients work!
g_fn = jax.grad(lambda a, b: jnp.sum(jax_fn(a, b)), argnums=(0, 1))
ga, gb = g_fn(a, b)
```

With multiple outputs

```python
def torch_fn(a, b):
    layer = torch.nn.Linear(2, 20).to(a)
    return a + b, torch.norm(a), layer(a * b)

shape = (10, 2)
a, b = torch.randn(shape), torch.randn(shape)
jax_fn = torch2jax(torch_fn, a, b)

key = jax.random.key(0)
device = jax.devices("cuda")[0]
a = jax.device_put(jax.random.normal(key, shape), device)
b = jax.device_put(jax.random.normal(key, shape), device)

x, y, z = jax_fn(a, b)
x, y, z = jax.jit(jax_fn)(a, b)
```

For a more advanced discussion on different ways of specifying input/output
specification of the wrapped function, take a look at:
[input_output_specification.ipynb](./examples/input_output_specification.ipynb)
notebook in the `examples` folder.

# Multi-device (multi-GPU) support

`torch2jax` runs PyTorch code on sharded JAX arrays, across multiple GPUs, under
`jax.jit` and with gradients. The recommended way is JAX's **explicit
sharding**: the sharding of an array is part of its type, so `torch2jax` reads
the input shardings from the arrays and you only state how the outputs are
sharded with `out_specs=`. A PyTorch function is opaque to JAX, so `torch2jax`
**never implicitly all-gathers sharded inputs**, every collective in your
program is one you asked for, and a missing or unsupported `out_specs` is an
error rather than a silent slowdown.

- **explicit sharding** (recommended), with explicit mesh axes (the
  `jax.make_mesh` default in recent JAX, otherwise pass
  `axis_types=(AxisType.Explicit,) * n`) &mdash; pass `out_specs=` and the
  torch function is called per-shard, inside a `jax.shard_map` that is
  manual only over the mesh axes the inputs are sharded along (`in_specs` are
  read from the input types). `output_shapes`, if given, are global and are split
  per-shard by `out_specs`. Without `out_specs`, sharded inputs raise an error;
  replicate them explicitly (`jax.sharding.reshard(x, P())`) to call the torch
  function on the full arrays instead. `out_specs` over Auto mesh axes raises an
  error, since XLA would silently all-gather the inputs.
- **inside `jax.shard_map`** (manual axes), if you already write per-shard code
  &mdash; call `torch2jax` as usual, the torch function sees the local shards. Gradients type-check with the default
  `check_vma=True`, and cotangents of replicated inputs (e.g., parameters) are
  `psum`-ed automatically.

Gradients work in both cases.

On multiple devices, the torch function is called **concurrently**, once per
device, from different threads. Pure tensor code is fine, but stateful torch
code, e.g., `torch.func.functional_call` (it temporarily swaps the parameters of
a shared module), can silently produce wrong results. Pass `lock=True` to run
all torch calls under a process-wide lock (or `lock=my_lock` for your own lock,
e.g., one per model). The GPUs still compute in parallel, since torch only
enqueues work, unless the torch function synchronizes with the host (`.item()`,
`.cpu()`, data-dependent shapes like `x[mask]`), in which case the devices run
one after another.

```python
import torch
import jax
from jax.sharding import PartitionSpec as P, NamedSharding
from torch2jax import torch2jax

model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 16))
params = {k: jax.numpy.asarray(v.detach().numpy()) for k, v in model.named_parameters()}
call_model = lambda x, params: torch.func.functional_call(model, params, x)

mesh = jax.make_mesh((jax.device_count(),), ("x",))  # explicit axes by default
params = jax.device_put(params, NamedSharding(mesh, P()))  # replicated
x = jax.device_put(jax.numpy.ones((128, 1024)), NamedSharding(mesh, P("x")))  # sharded along the batch

# 1. explicit sharding: the torch function runs per-shard, the output is sharded along "x"
fwd_fn = torch2jax(call_model, x, params, out_specs=P("x"), lock=True)  # functional_call mutates `model`
with jax.set_mesh(mesh):
    y = jax.jit(fwd_fn)(x, params)
    grads = jax.jit(jax.grad(lambda params: jax.numpy.sum(fwd_fn(x, params) ** 2)))(params)

# 2. or inside shard_map, where the torch function sees the local shards
@jax.jit
@jax.shard_map(mesh=mesh, in_specs=(P("x"), P()), out_specs=P("x"))
def fwd_fn_shard_map(x, params):
    return torch2jax(call_model, x, params, lock=True)(x, params)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/rdyro/torch2jax/main/images/data_parallel.png" width="100%" style="width: 100%; max-width: 1000px; height: auto; max-height: 1000px;">
  <p align="center">Fig: Overlapping torch calls on multiple devices (RTX A4000 x 4)</p>
</p>

> Note: `jax.vmap`'s semantics might indicate that it can compute on sharded
arrays, it can work, but it is not recommend, and because of `torch2jax`'s
implementation will likely be executed sequentially (and likely be slow).

For more on explicit sharding, `lock=` and CUDA streams, see the
[multi-device guide](https://rdyro.github.io/torch2jax/multi_device/).

# Automatically defining gradients

`torch2jax` defines reverse-mode gradients (VJP rules) by default (`depth=2`).
The `depth` parameter controls how many times the function can be differentiated.

```python
import torch
import jax
from jax import numpy as jnp
import numpy as np
from torch2jax import torch2jax

def torch_fn(a, b):
  return torch.nn.MSELoss()(a, b)

shape = (6,)
xt, yt = torch.randn(shape), torch.randn(shape)

# depth=2 is the default, allowing up to 2nd-order differentiation
jax_fn = torch2jax(torch_fn, xt, yt)

# derivatives are taken using PyTorch autodiff
g_fn = jax.grad(jax_fn, argnums=(0, 1))
x, y = jnp.array(np.random.randn(*shape)), jnp.array(np.random.randn(*shape))

print(g_fn(x, y))

# JIT works too
print(jax.jit(g_fn)(x, y))
```

Use `depth=0` to skip gradient definitions (forward-only):

```python
jax_fn = torch2jax(torch_fn, xt, yt, depth=0)  # no VJP, forward-only
```

> **Note**: `torch2jax_with_vjp` is deprecated. Use `torch2jax` (which has
> `depth=2` by default) instead.

Caveats:

- `jax.hessian(f)` will not work since `torch2jax` uses forward differentiation, but
  the same functionality can be achieved using `jax.jacobian(jax.jacobian(f))`
- in line with JAX philosophy, PyTorch functions must be non-mutable,
  [torch.func](https://pytorch.org/docs/master/func.html) has a good description
  of how to convert e.g., PyTorch models, to non-mutable formulation

# Dealing with Changing Shapes

Wrapped functions now **automatically cache** for different input shapes. When
called with new shapes, the wrapper re-creates itself transparently (a warning
is emitted on the first shape change).

```python
jax_fn = torch2jax(torch_fn, xt_10, yt_10)  # wrapped for shape (10,)

# calling with shape (20,) works — the wrapper is automatically re-created and cached
jax_fn(x_20, y_20)

# subsequent calls with shape (20,) reuse the cached wrapper
jax_fn(x_20, y_20)
```

You can also still manually call `torch2jax` inside JIT for full control:

```python
@jax.jit
def compute(a, b, c):
    d = torch2jax(
        torch_fn,
        jax.ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
        jax.ShapeDtypeStruct(b.shape, dtype_t2j(b.dtype)),
        output_shapes=jax.ShapeDtypeStruct(a.shape, dtype_t2j(a.dtype)),
    )(a, b)
    return d - c

print(compute(a, b, a))
```

# Timing Comparison vs `pure_callback`

This package achieves a much better performance when calling PyTorch code from
JAX because it does not copy its input arguments and does not move CUDA data off
the GPU.

<img src="https://raw.githubusercontent.com/rdyro/torch2jax/main/images/time_difference.png" width="100%">


# Current Limitations of `torch2jax`

- compilation happens on module import and can take 1-2 minutes (it will be cached afterwards)
- in the PyTorch function all arguments must be tensors, all outputs must be tensors
- all arguments of a single torch call must be on the same device (sharded
  arrays are called per device, see [Multi-device (multi-GPU) support](#multi-device-multi-gpu-support))
- an input/output shape (e.g. `output_shapes=` kw argument) representations (for
  flexibility in input and output structure) must be wrapped in `torch.Size` or
  `jax.ShapeDtypeStruct`

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


# Roadmap

- [x] call PyTorch functions on JAX data without input data copy
- [x] call PyTorch functions on JAX data without input data copy under jit
- [x] support both GPU and CPU
- [x] (feature) support partial CPU building on systems without CUDA
- [x] (user-friendly) support functions with a single output (return a single output, not a tuple)
- [x] (user-friendly) support arbitrary argument input and output structure (use pytrees on the
      Python side)
- [x] (feature) support batching (e.g., support for `jax.vmap`)
- [x] (feature) support integer input/output types
- [x] (feature) support mixed-precision arguments in inputs/outputs
- [x] (feature) support defining VJP for the wrapped function (now on by default via `depth=2`)
- [x] (tests) test how well device mapping works on multiple GPUs
- [x] (feature) multi-device support: explicit sharding (`out_specs`) and `shard_map`, with gradients
- [x] (tests) setup automatic tests for multiple versions of Python, PyTorch and JAX
- [ ] (feature) look into supporting in-place functions (support for output without copy)
- [ ] (feature) support TPU

# Related Work

Our Python package wraps PyTorch code as-is (so custom code and mutating code
will work!), but if you're looking for an automatic way to transcribe a
supported subset of PyTorch code to JAX, take a look at
[https://github.com/samuela/torch2jax/tree/main](https://github.com/samuela/torch2jax/tree/main).

We realize that two packages named the same is not ideal. As we work towards a
solution, here's a stop-gap solution. We offer a helper script to install the
package with an alias name, installing our package using pip under a different
name.

1. `$ git clone https://github.com/rdyro/torch2jax.git` - clone this repo
2. `$ python3 install_package_aliased.py new_name_torch2jax --install --test` - install and test this package under the name `new_name_torch2jax`
3. you can now use this package under the name `new_name_torch2jax`
