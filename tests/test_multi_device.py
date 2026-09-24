"""Multi-device stress tests: real GPUs when >= 2 are attached, otherwise (the 4 forced) CPU devices."""

import re
import time
import threading

from absl.testing import absltest, parameterized
import numpy as np
import torch
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

from torch2jax import torch2jax, t2j

CUDA_AVAILABLE = torch.cuda.is_available() and jax.default_backend() == "gpu"
_COLLECTIVE = re.compile(r"\s(all-gather|all-reduce|reduce-scatter|all-to-all|collective-permute)(?:-start)?\(")


def _devices():
    gpus = jax.devices("gpu") if jax.default_backend() == "gpu" else []
    devices = gpus if len(gpus) >= 2 else jax.devices("cpu")
    if len(devices) < 2:
        raise absltest.SkipTest("needs >= 2 devices")
    return devices[: 4 if len(devices) >= 4 else 2]


def _mesh(shape=None, names=("x",)):
    devices = _devices()
    shape = (len(devices),) if shape is None else shape
    if int(np.prod(shape)) != len(devices):
        raise absltest.SkipTest(f"mesh {shape} needs {int(np.prod(shape))} devices")
    return jax.make_mesh(shape, names, axis_types=(AxisType.Explicit,) * len(names), devices=devices)


def _on_gpu(mesh):
    return mesh.devices.flat[0].platform == "gpu"


def _put(x, mesh, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))


def _collectives(fn, *args, mesh):
    with jax.set_mesh(mesh):
        hlo = jax.jit(fn).trace(*args).lower().compile().as_text()
    return sorted(m.group(1) for m in _COLLECTIVE.finditer(hlo))


def _randn(key, shape):
    return jax.random.normal(jax.random.key(key), shape, dtype=jnp.float32)


class TestShardPlacement(parameterized.TestCase):
    def test_each_device_gets_its_own_shard(self):
        # every shard must be seen by torch on the device JAX placed it on, with the right block of rows
        mesh = _mesh()
        n = mesh.devices.size
        x = _put(jnp.arange(8.0 * n)[:, None] * jnp.ones((1, 4)), mesh, P("x"))
        seen, lock = [], threading.Lock()

        def torch_fn(a):
            if not a.is_meta:  # skip the shape inference
                with lock:
                    seen.append((a.device.index if a.is_cuda else None, a[0, 0].item()))
            return 2 * a

        f = torch2jax(torch_fn, x, depth=0, out_specs=P("x"))
        with jax.set_mesh(mesh):
            y = jax.block_until_ready(jax.jit(f)(x))
        np.testing.assert_allclose(np.asarray(y), 2 * np.asarray(x))
        expected = {
            (d.id if _on_gpu(mesh) else None, float(idx[0].start or 0))
            for d, idx in x.sharding.devices_indices_map(x.shape).items()
        }
        self.assertEqual(set(seen), expected)
        self.assertLen(seen, n)


class TestTransfers(parameterized.TestCase):
    def test_t2j_when_another_gpu_is_current(self):
        if not CUDA_AVAILABLE or torch.cuda.device_count() < 2:
            self.skipTest("needs >= 2 GPUs")
        with torch.cuda.device(1):
            x = t2j(torch.arange(4.0, device="cuda:0"))
        self.assertEqual(x.devices(), {jax.devices("gpu")[0]})
        np.testing.assert_allclose(np.asarray(x), np.arange(4.0))


class TestShardingFuzz(parameterized.TestCase):
    @parameterized.parameters(range(12))
    def test_random_specs_elementwise(self, seed):
        # per-shard semantics of an elementwise function equal global semantics, with zero communication
        rng = np.random.default_rng(seed)
        n = len(_devices())
        mesh_shape, names = [((n,), ("x",)), ((1, n), ("x", "y")), ((n, 1), ("x", "y"))][seed % 3]
        if n == 4 and seed % 4 == 3:
            mesh_shape, names = (2, 2), ("x", "y")
        mesh = _mesh(mesh_shape, names)
        ndim = int(rng.integers(1, 4))
        shape = tuple(int(rng.choice([4, 8, 12])) * (4 if i == 0 else 1) for i in range(ndim))
        dims = [None] * ndim
        for ax in names:  # each mesh axis shards a random dim (or nothing)
            d = int(rng.integers(-1, ndim))
            if d >= 0 and shape[d] % (mesh.shape[ax] * (1 if dims[d] is None else 2)) == 0:
                dims[d] = ax if dims[d] is None else (dims[d], ax)
        spec = P(*dims)
        a, b, c = (_put(_randn(seed * 3 + i, shape), mesh, spec) for i in range(3))
        torch_fn = lambda a, b, c: (a * torch.sin(b) + c**2, torch.tanh(a - c))
        jax_fn = lambda a, b, c: (a * jnp.sin(b) + c**2, jnp.tanh(a - c))
        f = torch2jax(torch_fn, a, b, c, out_specs=spec)
        with jax.set_mesh(mesh):
            out, ref = jax.jit(f)(a, b, c), jax.jit(jax_fn)(a, b, c)
            g = jax.jit(jax.grad(lambda *xs: sum(jnp.sum(o) for o in f(*xs)), argnums=(0, 1, 2)))(a, b, c)
            g_ref = jax.grad(lambda *xs: sum(jnp.sum(o) for o in jax_fn(*xs)), argnums=(0, 1, 2))(a, b, c)
        for o, r in zip(jax.tree.leaves((out, g)), jax.tree.leaves((ref, g_ref))):
            np.testing.assert_allclose(np.asarray(o), np.asarray(r), rtol=1e-5, atol=1e-5)
            self.assertEqual(jax.typeof(o).sharding.spec, jax.typeof(r).sharding.spec)
        grad_fn = jax.grad(lambda *xs: sum(jnp.sum(o) for o in f(*xs)), argnums=(0, 1, 2))
        self.assertEqual(_collectives(f, a, b, c, mesh=mesh), [], f"{mesh_shape=} {spec=}")
        # the scalar loss is a sum over shards: at most one all-reduce per output for the loss itself
        self.assertNotIn("all-gather", _collectives(grad_fn, a, b, c, mesh=mesh), f"{mesh_shape=} {spec=}")


class TestDataParallelTraining(parameterized.TestCase):
    def test_sgd_matches_jax_and_only_all_reduces(self):
        mesh = _mesh()
        n = mesh.devices.size
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        params = {k: jnp.asarray(v.detach().numpy()) for k, v in model.named_parameters()}
        params = jax.device_put(params, NamedSharding(mesh, P()))
        x, y = _put(_randn(0, (8 * n, 16)), mesh, P("x")), _put(_randn(1, (8 * n, 1)), mesh, P("x"))
        lock = threading.Lock()  # functional_call swaps the shared module's params, devices call torch concurrently

        def torch_model(x, params):
            with lock:
                return torch.func.functional_call(model, params, x)

        jax_model = lambda x, p: jnp.tanh(x @ p["0.weight"].T + p["0.bias"]) @ p["2.weight"].T + p["2.bias"]
        f = torch2jax(torch_model, x, params, out_specs=P("x"))

        def step(model_fn):
            @jax.jit
            def _step(params, x, y):
                loss, g = jax.value_and_grad(lambda p: jnp.mean((model_fn(x, p) - y) ** 2))(params)
                return loss, jax.tree.map(lambda p, g: p - 0.1 * g, params, g)

            return _step

        with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):  # torch uses full fp32 matmuls
            p_torch, p_jax = params, params
            for _ in range(5):
                loss_torch, p_torch = step(f)(p_torch, x, y)
                loss_jax, p_jax = step(jax_model)(p_jax, x, y)
            collectives = _collectives(lambda p: step(f)(p, x, y), params, mesh=mesh)
        np.testing.assert_allclose(float(loss_torch), float(loss_jax), rtol=1e-4)
        for k in params:
            np.testing.assert_allclose(np.asarray(p_torch[k]), np.asarray(p_jax[k]), rtol=1e-4, atol=1e-5)
            self.assertEqual(jax.typeof(p_torch[k]).sharding.spec, P(*([None] * p_torch[k].ndim)))
        self.assertEqual(set(collectives), {"all-reduce"})  # gradient (and loss) reductions only


class TestConcurrentCalls(parameterized.TestCase):
    def test_torch_fn_is_called_concurrently_per_device(self):
        # documents the execution model: one call per device, from different threads, possibly at the same time
        mesh = _mesh()
        threads, lock = set(), threading.Lock()

        def torch_fn(a):
            if not a.is_meta:
                with lock:
                    threads.add(threading.get_ident())
            return a + 1

        x = _put(jnp.ones((8 * mesh.devices.size, 4)), mesh, P("x"))
        with jax.set_mesh(mesh):
            jax.block_until_ready(jax.jit(torch2jax(torch_fn, x, depth=0, out_specs=P("x")))(x))
        self.assertLen(threads, mesh.devices.size)


class TestTransformsUnderSharding(parameterized.TestCase):
    def test_scan_over_layers(self):
        mesh = _mesh()
        x = _put(_randn(0, (8 * mesh.devices.size, 16)), mesh, P("x"))
        ws = _put(_randn(1, (4, 16, 16)) * 0.3, mesh, P())
        layer = torch2jax(lambda x, w: torch.tanh(x @ w), x, ws[0], out_specs=P("x"))
        jax_layer = lambda x, w: jnp.tanh(x @ w)
        net = lambda layer: lambda x, ws: jnp.sum(jax.lax.scan(lambda x, w: (layer(x, w), None), x, ws)[0] ** 2)
        with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
            vg = lambda fn: jax.jit(jax.value_and_grad(net(fn), argnums=1))(x, ws)
            (v, g), (v_ref, g_ref) = vg(layer), vg(jax_layer)
            collectives = _collectives(jax.grad(net(layer), argnums=1), x, ws, mesh=mesh)
        np.testing.assert_allclose(float(v), float(v_ref), rtol=1e-4)
        np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-3, atol=1e-4)
        self.assertNotIn("all-gather", collectives)

    def test_remat_recomputes_per_shard(self):
        mesh = _mesh()
        x = _put(_randn(0, (8 * mesh.devices.size, 16)), mesh, P("x"))
        calls, lock = [], threading.Lock()

        def torch_fn(a):
            if not a.is_meta:
                with lock:
                    calls.append(tuple(a.shape))
            return torch.sin(a)

        f = jax.checkpoint(torch2jax(torch_fn, x, out_specs=P("x")))
        with jax.set_mesh(mesh):
            g = jax.block_until_ready(jax.jit(jax.grad(lambda x: jnp.sum(f(x) ** 2)))(x))
        np.testing.assert_allclose(np.asarray(g), np.asarray(2 * jnp.sin(x) * jnp.cos(x)), rtol=1e-5, atol=1e-6)
        self.assertTrue(calls and all(s == (8, 16) for s in calls), calls)

    def test_same_function_on_different_meshes(self):
        n = len(_devices())
        x = _randn(0, (8 * n, 4))
        f = torch2jax(lambda a: a * 3, x, depth=0, out_specs=P("x"))
        for mesh in [_mesh(), _mesh((1, n), ("x", "y")), _mesh((n, 1), ("x", "y"))]:
            with jax.set_mesh(mesh):
                y = jax.jit(f)(_put(x, mesh, P("x")))
            np.testing.assert_allclose(np.asarray(y), 3 * np.asarray(x))
        np.testing.assert_allclose(np.asarray(f(x)), 3 * np.asarray(x))  # unsharded still works

    def test_per_shard_output_shape_differs(self):
        mesh = _mesh()
        x = _randn(0, (8 * mesh.devices.size, 4))
        f = torch2jax(lambda a: a[:1], x, depth=0, out_specs=P("x"))  # per-shard output of 1 row
        with jax.set_mesh(mesh):
            y = jax.jit(f)(_put(x, mesh, P("x")))
        self.assertEqual(y.shape, (mesh.devices.size, 4))  # 1 row per shard, concatenated


class TestConcurrencyAndStreams(parameterized.TestCase):
    def test_devices_run_concurrently(self):
        # each shard's torch call busy-waits on its GPU: with n GPUs the sharded call should take ~1x, not ~nx
        mesh = _mesh()
        if not _on_gpu(mesh):
            self.skipTest("needs GPUs")
        n = mesh.devices.size
        cycles = int(3e8)  # ~0.1-0.2 s on a modern GPU
        x = _put(jnp.ones((8 * n, 4)), mesh, P("x"))
        x1 = jax.device_put(jnp.ones((8, 4)), mesh.devices.flat[0])

        def torch_fn(a):
            torch.cuda._sleep(cycles)
            return a + 1

        fs = jax.jit(torch2jax(torch_fn, x, depth=0, out_specs=P("x")))
        f1 = jax.jit(torch2jax(torch_fn, x1, depth=0))

        def timeit(f, a):
            jax.block_until_ready(f(a))
            t = time.perf_counter()
            for _ in range(3):
                jax.block_until_ready(f(a))
            return (time.perf_counter() - t) / 3

        with jax.set_mesh(mesh):
            t_sharded = timeit(fs, x)
        t_single = timeit(f1, x1)
        print(f"single device {t_single * 1e3:.1f} ms, {n} devices sharded {t_sharded * 1e3:.1f} ms")
        self.assertLess(t_sharded, 1.5 * t_single)

    def test_interleaved_chain_on_all_devices(self):
        mesh = _mesh()
        n = mesh.devices.size
        x = _put(_randn(0, (512 * n, 512)), mesh, P("x"))
        w = _put(_randn(1, (512, 512)) * 0.05, mesh, P())
        torch_step = torch2jax(lambda a, w: torch.tanh(a @ w) + a, x, w, out_specs=P("x"))
        jax_step = lambda a, w: jnp.tanh(a @ w) + a

        def chain(step):
            def f(x, w):
                for _ in range(6):
                    x = jnp.sin(step(jnp.cos(x) * 1.5, w))
                return jnp.sum(x**2)

            return f

        with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
            for _ in range(3):
                vg = lambda step: jax.jit(jax.value_and_grad(chain(step), argnums=1))(x, w)
                (v, g), (v_ref, g_ref) = vg(torch_step), vg(jax_step)
                np.testing.assert_allclose(float(v), float(v_ref), rtol=1e-4)
                np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-3, atol=1e-3)

    def test_memory_is_stable_over_many_calls(self):
        mesh = _mesh()
        if not _on_gpu(mesh):
            self.skipTest("needs GPUs")
        x = _put(_randn(0, (1024 * mesh.devices.size, 1024)), mesh, P("x"))
        f = jax.jit(torch2jax(lambda a: torch.relu(a @ a.T[:, :1024]), x, depth=0, out_specs=P("x")))
        # memory_stats() is None with XLA_PYTHON_CLIENT_ALLOCATOR=platform, then only torch's allocator is checked
        usage = lambda: [s["bytes_in_use"] for d in mesh.devices.flat if (s := d.memory_stats()) is not None] + [
            torch.cuda.memory_allocated(i) for i in range(mesh.devices.size)
        ]
        with jax.set_mesh(mesh):
            for _ in range(20):
                y = jax.block_until_ready(f(x))
            before = usage()
            for _ in range(200):
                y = jax.block_until_ready(f(x))
            after = usage()
        del y
        self.assertEqual(before, after)


if __name__ == "__main__":
    absltest.main()
