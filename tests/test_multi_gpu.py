#!/usr/bin/env python3

import os
import functools
from pathlib import Path
import sys
import math

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from absl.testing import absltest, parameterized
import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import NamedSharding as NS, PartitionSpec as P, Mesh, SingleDeviceSharding as SDS
from jax.experimental.shard_map import shard_map
import numpy as np
import torch
from torch import Tensor

from torch2jax import torch2jax
from torch2jax import Size, dtype_t2j, tree_t2j, tree_j2t, t2j, j2t

WRITE_PROFILE = False

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))


def torch_fn(a, b, simulate_compute: bool = False):
    ret = a + b
    if simulate_compute:
        ret = ret + 0 * torch.linalg.svd(a)[0][0, 0]
    return ret


def _generate_data(seed: int, size0: int, size1: int, devices):
    size0 = math.floor(size0 / len(devices)) * len(devices)
    size1 = math.floor(size1 / len(devices)) * len(devices)
    shape = (size0, size1)
    dtype = jnp.float32
    mesh = jax.make_mesh((len(devices),), P("x"), devices=devices)
    sharding = NS(mesh, P("x", None))

    @functools.partial(jax.jit, out_shardings=(sharding, sharding))
    def _gen_data():
        keyit = iter(random.split(random.key(seed), 1024))
        a = jax.random.normal(next(keyit), shape)
        b = jax.random.normal(next(keyit), shape)
        return a, b

    return mesh, sharding, jax.ShapeDtypeStruct(shape, dtype), _gen_data()


def _to_device0(x, devices=None):
    device0 = np.array(devices).reshape(-1)[0] if devices is not None else jax.devices()[0]
    return jax.device_put(x, SDS(device0))


class MultiDeviceTest(parameterized.TestCase):
    def setUp(self):
        self.profiled_svd_compute_shard_map = False
        self.profiled_svd_compute_jax_jit = False

    @parameterized.product(
        seed=[0, 1, 2], device=["cpu", "cuda"], size0=[256], size1=[8, 16], simulate_compute=[True, False]
    )
    def test_shard_map(self, seed: int, size0, size1, device, simulate_compute):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        if simulate_compute and device == "cuda":
            size0, size1 = 1024 * len(jax.devices(device)), 1024
        mesh, sharding, shape, (a, b) = _generate_data(seed, size0=size0, size1=size1, devices=jax.devices(device))

        spec = sharding.spec

        @jax.jit
        @functools.partial(shard_map, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_rep=False)
        def fn_(a, b):
            jax_fn = torch2jax(functools.partial(torch_fn, simulate_compute=simulate_compute), a, b, output_shapes=a)
            return jax_fn(a, b)

        c = fn_(a, b)

        if WRITE_PROFILE and simulate_compute and device == "cuda" and not self.profiled_svd_compute_shard_map:
            with jax.profiler.trace(str(Path("/tmp/torch2jax-profiles/shard-map"))):
                for _ in range(3):
                    c = fn_(a, b).block_until_ready()
            self.profiled_svd_compute_shard_map = True

        c_torch = tree_t2j(torch_fn(*jax.tree.map(lambda x: j2t(_to_device0(x, mesh.devices)), (a, b))))
        print(f"c = {c}\nc_torch = {c_torch}")

        print("result sharding =")
        jax.debug.visualize_array_sharding(c)
        self.assertEqual(c.sharding, sharding)
        c = _to_device0(c, mesh.devices)
        np.testing.assert_allclose(np.array(c_torch), np.array(c))

    def test_pmap(self):
        self.skipTest("`pmap` doesn't work (just hangs), TODO(rdyro): more debugging needed")

        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        mesh, sharding, shape, (a, b) = _generate_data(seed, devices=jax.devices(device))

        fn_ = jax.jit(
            jax.pmap(
                torch2jax(torch_fn, a[0, ...], b[0, ...], output_shapes=jax.ShapeDtypeStruct(a.shape[1:], a.dtype)),
                in_axes=(0, 0),
                out_axes=0,
                devices=mesh.devices.reshape(-1),
            )
        )

        c = fn_(a, b).block_until_ready()
        print("done with c")
        c_torch = tree_t2j(torch_fn(*jax.tree.map(lambda x: j2t(_to_device0(x, mesh.devices)), (a, b))))
        print("done with torch")
        c = _to_device0(c, mesh.devices)
        print(f"c = {c}\nc_torch = {c_torch}")

        print("result sharding =")
        jax.debug.visualize_array_sharding(c)
        c = _to_device0(c, mesh.devices)
        np.testing.assert_allclose(np.array(c_torch), np.array(c))

    @parameterized.product(
        seed=[0, 1, 2], device=["cpu", "cuda"], size0=[1024, 16], size1=[1024, 16], simulate_compute=[True, False]
    )
    def test_auto_partitioning(self, seed, size0, size1, device, simulate_compute):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        if simulate_compute and device == "cuda":
            size0, size1 = 1024 * len(jax.devices(device)), 1024
        mesh, sharding, shape, (a, b) = _generate_data(seed, size0, size1, devices=jax.devices(device))

        jax_fn = torch2jax(
            functools.partial(torch_fn, simulate_compute=simulate_compute),
            a,
            b,
            output_shapes=a,
            output_sharding_spec=sharding.spec,
        )
        fn_ = jax.jit(jax_fn)

        c = fn_(a, b)
        if WRITE_PROFILE and simulate_compute and device == "cuda" and not self.profiled_svd_compute_jax_jit:
            with jax.profiler.trace(str(Path("/tmp/torch2jax-profiles/jax-jit"))):
                for _ in range(3):
                    c = fn_(a, b).block_until_ready()
            self.profiled_svd_compute_jax_jit = True

        c_torch = tree_t2j(torch_fn(*jax.tree.map(lambda x: j2t(_to_device0(x, mesh.devices)), (a, b))))
        print(f"c = {c}\nc_torch = {c_torch}")

        print("result sharding =")
        jax.debug.visualize_array_sharding(c)
        self.assertEqual(c.sharding, sharding)
        c = _to_device0(c)
        np.testing.assert_allclose(np.array(c_torch), np.array(c))


########################################################################################################################


class OldMultiDeviceTest(parameterized.TestCase):
    def test_multi_gpu_call(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Not enough devices")

        def torch_device_11_fn(x, y):
            assert x.device.index == 1 == y.device.index, "Device must be index 1"
            z = x + y
            return z

        def torch_device_00_fn(x: Tensor, y: Tensor):
            assert x.device.index == 0 == y.device.index, "Device must be index 0"
            z = x + y
            return z

        def torch_device_xx_fn(x: Tensor, y: Tensor):
            z = x + y
            return z

        device0, device1 = jax.devices("cuda")[0], jax.devices("cuda")[1]

        x = jax.device_put(jnp.zeros(10), device1)
        y = jax.device_put(jnp.zeros(10), device1)

        torchfn = torch2jax(torch_device_11_fn, x, y, output_shapes=Size(x.shape))
        z = torchfn(x, y)
        assert len(z.devices()) == 1 and list(z.devices())[0] == device1
        assert jnp.linalg.norm(z - (x + y)) < 1e-6

        ################################################################################################
        x = jax.device_put(jnp.zeros(10), device0)
        y = jax.device_put(jnp.zeros(10), device0)

        torchfn = torch2jax(torch_device_00_fn, x, y, output_shapes=Size(x.shape))
        z = torchfn(x, y)
        assert len(z.devices()) == 1 and list(z.devices())[0] == device0
        assert jnp.linalg.norm(z - (x + y)) < 1e-6

        ################################################################################################
        x = jax.device_put(jnp.zeros(10), device0)
        y = jax.device_put(jnp.zeros(10), device0)

        torchfn = torch2jax(torch_device_xx_fn, x, y, output_shapes=Size(x.shape))
        z = torchfn(x, y)
        assert len(z.devices()) == 1 and list(z.devices())[0] == device0
        assert jnp.linalg.norm(z - (x + y)) < 1e-6

        x = jax.device_put(jnp.zeros(10), device1)
        y = jax.device_put(jnp.zeros(10), device1)

        z = torchfn(x, y)
        assert len(z.devices()) == 1 and list(z.devices())[0] == device1
        assert jnp.linalg.norm(z - (x + y)) < 1e-6


if __name__ == "__main__":
    absltest.main()
