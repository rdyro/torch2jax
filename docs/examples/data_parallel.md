```python
import functools
import copy
from pathlib import Path

import torch
import torch.nn as nn
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding
from torch2jax import torch2jax, torch2jax_with_vjp, tree_j2t, tree_t2j


def _setattr(mod, key, delim: str = "."):
    if delim not in key:
        setattr(mod, key, None)
    else:
        key, key_remaining = key.split(delim, 1)
        _setattr(getattr(mod, key), key_remaining, delim=delim)


def _strip_model(model):
    for key in dict(model.named_parameters()).keys():
        _setattr(model, key, delim=".")


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(1024 * 1024, 1024), nn.SiLU(), nn.Linear(1024, 16)).to("cuda:0")
    params = dict(model.named_parameters())
    [p.requires_grad_(False) for p in params.values()]
    _strip_model(model)  # remove params from the model, leaving only a skeleton

    def call_model_torch(x, params):
        ys = []
        for _ in range(30):
            # functional_call uses the model in-place, we need a local copy
            local_model_skeleton = copy.deepcopy(model)
            ys.append(torch.func.functional_call(local_model_skeleton, params, x))
        return sum(ys)

    # jax init
    devices = jax.devices("cuda")
    mesh = jax.make_mesh((len(devices),), P("x"), devices=devices)
    params_sharding = NamedSharding(mesh, P())  # fully replicated
    batch_sharding = NamedSharding(mesh, P("x", None))  # sharded along batch

    x = jax.jit(
        lambda: jax.random.normal(jax.random.key(0), (128, 1024 * 1024)),
        out_shardings=batch_sharding,
    )()

    params = jax.tree.map(lambda p: jax.device_put(p, params_sharding), tree_t2j(params))
    params_spec = jax.tree.map(lambda _: params_sharding.spec, params)

    @jax.jit
    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(batch_sharding.spec, params_spec),
        out_specs=batch_sharding.spec,
        check_rep=False,
    )
    def fwd_fn(x, params):
        return torch2jax_with_vjp(call_model_torch, x, params, output_shapes=x[:, :16])(x, params)

    y = fwd_fn(x, params)

    # OR using JIT (but without gradients)
    fwd_fn = jax.jit(
        torch2jax(
            call_model_torch, x, params, output_shapes=x[:, :16], output_sharding_spec=P("x", None)
        )
    )

    y = fwd_fn(x, params)

    # profile the computation
    _ = fwd_fn(x, params)
    path = Path("/tmp/profiles/data_parallel")
    path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(path)):
       for _ in range(10):
           fwd_fn(x, params).block_until_ready()

```