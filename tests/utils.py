import jax
import time
from jax.lib import xla_extension

Device = xla_extension.Device

def jax_randn(shape, device, dtype):
    rand_seed = round((time.time() % 1) * 1e12)
    device = jax.devices(device)[0] if not isinstance(device, Device) else device
    return jax.device_put(
        jax.random.normal(jax.random.PRNGKey(rand_seed), shape, dtype=dtype), device=device
    )
