import os

# must be set before JAX initializes its backends, so that multi-device (sharding) tests have 4 CPU devices,
# first so that a user-provided device count takes precedence
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4 " + os.environ.get("XLA_FLAGS", "")
