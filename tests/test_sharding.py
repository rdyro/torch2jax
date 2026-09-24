import warnings
from functools import partial

from absl.testing import absltest, parameterized
import torch
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

from torch2jax import torch2jax, torch2jax_without_vjp


def _mesh(shape=(4,), names=("x",)):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        raise absltest.SkipTest("needs 4 CPU devices (set XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return jax.make_mesh(shape, names, axis_types=(AxisType.Explicit,) * len(names), devices=devices[:4])


def record(seen, x):
    if not x.is_meta:  # skip the meta-device shape inference, only record the actual calls
        seen.append(tuple(x.shape))


def _data(mesh, x_spec=P("x"), w_spec=P()):
    keys = iter(jax.random.split(jax.random.key(0), 1024))
    x = jax.device_put(jax.random.normal(next(keys), (16, 8)), NamedSharding(mesh, x_spec))
    w = jax.device_put(jax.random.normal(next(keys), (8, 3)), NamedSharding(mesh, w_spec))
    return x, w


class TestExplicitSharding(parameterized.TestCase):
    def test_sharded_input_without_out_specs_errors(self):
        mesh = _mesh()
        x, w = _data(mesh)
        f = torch2jax(lambda x, w: x @ w, x, w, depth=0)
        with jax.set_mesh(mesh), self.assertRaisesRegex(ValueError, "out_specs"):
            jax.jit(f)(x, w)

    @parameterized.product(depth=[0, 2])
    def test_replicated_inputs_global_semantics(self, depth):
        mesh = _mesh()
        x, w = _data(mesh, x_spec=P())
        seen = []
        f = torch2jax(lambda x, w: record(seen, x) or x @ w, x, w, depth=depth)
        with jax.set_mesh(mesh):
            y = jax.block_until_ready(jax.jit(f)(x, w))
        self.assertIn((16, 8), seen)
        self.assertNotIn((4, 8), seen)
        self.assertEqual(jax.typeof(y).sharding.spec, P(None, None))
        self.assertTrue(jnp.allclose(y, x @ w, atol=1e-5))

    @parameterized.product(depth=[0, 2], use_jit=[True, False])
    def test_out_specs_per_shard(self, depth, use_jit):
        mesh = _mesh()
        x, w = _data(mesh)
        seen = []
        f = torch2jax(lambda x, w: record(seen, x) or x @ w, x, w, depth=depth, out_specs=P("x"))
        with jax.set_mesh(mesh), warnings.catch_warnings():
            warnings.filterwarnings("error", message="torch2jax")
            y = jax.block_until_ready((jax.jit(f) if use_jit else f)(x, w))
        self.assertIn((4, 8), seen)
        self.assertNotIn((16, 8), seen)
        self.assertEqual(jax.typeof(y).sharding.spec, P("x", None))
        self.assertTrue(jnp.allclose(y, x @ w, atol=1e-5))

    def test_out_specs_grad(self):
        mesh = _mesh()
        x, w = _data(mesh)
        f = torch2jax(lambda x, w: torch.sin(x @ w), x, w, out_specs=P("x"))
        loss = lambda f: lambda x, w: jnp.sum(f(x, w) ** 2)
        with jax.set_mesh(mesh):
            gx, gw = jax.jit(jax.grad(loss(f), argnums=(0, 1)))(x, w)
            gx_ref, gw_ref = jax.grad(loss(lambda x, w: jnp.sin(x @ w)), argnums=(0, 1))(x, w)
        self.assertEqual(jax.typeof(gx).sharding.spec, jax.typeof(x).sharding.spec)
        self.assertEqual(jax.typeof(gw).sharding.spec, jax.typeof(w).sharding.spec)
        self.assertTrue(jnp.allclose(gx, gx_ref, atol=1e-4) and jnp.allclose(gw, gw_ref, atol=1e-4))

    def test_out_specs_prefix_for_multiple_outputs(self):
        mesh = _mesh()
        x, w = _data(mesh)
        f = torch2jax(lambda x, w: (x @ w, 2 * x), x, w, depth=0, out_specs=P("x"))
        with jax.set_mesh(mesh):
            y, z = jax.jit(f)(x, w)
        self.assertTrue(jnp.allclose(y, x @ w, atol=1e-5) and jnp.allclose(z, 2 * x))

    @parameterized.product(depth=[0, 2])
    def test_out_specs_splits_output_shapes(self, depth):
        # global `output_shapes` are partitioned by `out_specs`, the torch fn is never run for shape inference
        mesh = _mesh()
        x, w = _data(mesh)
        devices = []
        fn = lambda x, w: devices.append(x.device.type) or (x @ w, 2 * x)
        shapes = (jax.ShapeDtypeStruct((16, 3), x.dtype), jax.ShapeDtypeStruct((16, 8), x.dtype))
        f = torch2jax(fn, x, w, depth=depth, output_shapes=shapes, out_specs=(P("x"), P("x")))
        with jax.set_mesh(mesh):
            y, z = jax.jit(f)(x, w)
            if depth > 0:
                gw = jax.jit(jax.grad(lambda x, w: jnp.sum(f(x, w)[0] ** 2), argnums=1))(x, w)
                self.assertTrue(jnp.allclose(gw, jax.grad(lambda w: jnp.sum((x @ w) ** 2))(w), atol=1e-4))
        self.assertNotIn("meta", devices)
        self.assertTrue(jnp.allclose(y, x @ w, atol=1e-5) and jnp.allclose(z, 2 * x))

    def test_out_specs_over_auto_axes_errors(self):
        # with Auto axes, XLA would silently all-gather the inputs, so a per-shard call must use Explicit axes
        mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Auto,), devices=list(_mesh().devices.flat))
        x, w = _data(mesh)
        f = torch2jax(lambda x, w: x @ w, x, w, depth=0, out_specs=P("x"))
        with jax.set_mesh(mesh), self.assertRaisesRegex(ValueError, "Auto mesh axes"):
            jax.jit(f)(x, w)

    def test_out_specs_indivisible_output_shapes_errors(self):
        mesh = _mesh()
        x, w = _data(mesh)
        f = torch2jax(lambda x, w: x @ w, x, w, depth=0, output_shapes=jax.ShapeDtypeStruct((3,), x.dtype),
                      out_specs=P("x"))
        with jax.set_mesh(mesh), self.assertRaisesRegex(ValueError, "cannot be partitioned"):
            jax.jit(f)(x, w)

    def test_partially_manual(self):
        mesh = _mesh((2, 2), ("x", "y"))
        x, w = _data(mesh)
        seen = []
        f = torch2jax(lambda x, w: record(seen, x) or x @ w, x, w, depth=0, out_specs=P("x"))
        with jax.set_mesh(mesh):
            y = jax.block_until_ready(jax.jit(f)(x, w))
        self.assertEqual(seen, [(8, 8)] * 4)  # manual along "x" only, replicated along "y"
        self.assertTrue(jnp.allclose(y, x @ w, atol=1e-5))

    def test_output_sharding_spec_is_deprecated_alias_and_keeps_shardy(self):
        mesh = _mesh()
        x, w = _data(mesh)
        with self.assertWarnsRegex(UserWarning, "deprecated"):
            f = torch2jax_without_vjp(lambda x, w: x @ w, x, w, output_sharding_spec=P("x"))
        with jax.set_mesh(mesh):
            y = jax.jit(f)(x, w)
        self.assertTrue(jnp.allclose(y, x @ w, atol=1e-5))
        self.assertTrue(jax.config.jax_use_shardy_partitioner)


class TestManualSharding(parameterized.TestCase):
    @parameterized.product(depth=[0, 2])
    def test_shard_map_mixed_replicated_and_sharded(self, depth):
        mesh = _mesh()
        x, w = _data(mesh)

        @jax.jit
        @partial(jax.shard_map, mesh=mesh, in_specs=(P("x"), P()), out_specs=P("x"))
        def fwd(x, w):
            return torch2jax(lambda x, w: x @ w, x, w, depth=depth)(x, w)

        self.assertTrue(jnp.allclose(fwd(x, w), x @ w, atol=1e-5))

    def test_shard_map_grad_with_check_vma(self):
        mesh = _mesh()
        x, w = _data(mesh)

        @jax.jit
        @partial(jax.shard_map, mesh=mesh, in_specs=(P("x"), P()), out_specs=(P("x"), P()))
        def grads(x, w):
            f = torch2jax(lambda x, w: torch.sin(x @ w), x, w)
            return jax.grad(lambda x, w: jnp.sum(f(x, w) ** 2), argnums=(0, 1))(x, w)

        with jax.set_mesh(mesh):
            gx, gw = grads(x, w)  # the replicated w cotangent must be psum-ed automatically, no manual psum
            gx_ref, gw_ref = jax.grad(lambda x, w: jnp.sum(jnp.sin(x @ w) ** 2), argnums=(0, 1))(x, w)
        self.assertTrue(jnp.allclose(gx, gx_ref, atol=1e-4))
        self.assertTrue(jnp.allclose(gw, gw_ref, atol=1e-4))

    def test_shard_map_output_is_varying(self):
        mesh = _mesh()
        x, w = _data(mesh)

        @partial(jax.shard_map, mesh=mesh, in_specs=(P("x"), P()), out_specs=P())
        def fwd(x, w):
            return torch2jax(lambda x, w: x @ w, x, w, depth=0)(x, w)

        with self.assertRaises(Exception):  # per-device outputs must not type-check as replicated
            jax.jit(fwd)(x, w)


if __name__ == "__main__":
    absltest.main()
