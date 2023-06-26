import sys
from pathlib import Path
import time

import torch
import numpy as np
from tqdm import tqdm
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[1].absolute() / "tests"))

from pure_callback_alternative import wrap_torch_fn
from utils import jax_randn
from torch2jax import torch2jax, j2t


def torch_fn(a, b):
    return a + b


cpu_times_pc, cpu_times_tj, cuda_times_pc, cuda_times_tj = [], [], [], []
cpu_times_torch, cuda_times_torch = [], []

numels = np.logspace(3, 8, 20, dtype=int)
for numel in tqdm(numels):
    shape = [numel]
    jax_fn = jax.jit(torch2jax(torch_fn, output_shapes=[shape]))
    jax2_fn = jax.jit(wrap_torch_fn(torch_fn, output_shapes=[shape]))
    trials = 50

    for device in ["cpu", "cuda"]:
        dtype = jnp.float64 if device == "cpu" else jnp.float32
        a = jax_randn(shape, device=device, dtype=dtype)
        b = jax_randn(shape, device=device, dtype=dtype)

        # compile
        jax_fn(a, b)[0].block_until_ready()
        jax2_fn(a, b)[0].block_until_ready()

        ts = [time.time()]
        for _ in range(trials):
            jax_fn(a, b)[0].block_until_ready()
            ts.append(time.time())
        ts = np.diff(ts)
        if device == "cpu":
            cpu_times_tj.append(ts[1:])
        else:
            cuda_times_tj.append(ts[1:])

        ts = [time.time()]
        for _ in range(trials):
            jax2_fn(a, b)[0].block_until_ready()
            ts.append(time.time())
        ts = np.diff(ts)
        if device == "cpu":
            cpu_times_pc.append(ts[1:])
        else:
            cuda_times_pc.append(ts[1:])

        a, b = j2t(a), j2t(b)
        ts = [time.time()]
        for _ in range(trials):
            c = torch_fn(a, b)
            torch.cuda.synchronize()
            ts.append(time.time())
        ts = np.diff(ts)
        if device == "cpu":
            cpu_times_torch.append(ts[1:])
        else:
            cuda_times_torch.append(ts[1:])


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
mu, std = [np.mean(ts) for ts in cpu_times_pc], [np.std(ts) for ts in cpu_times_pc]
mu, std = np.array(mu), np.array(std)
ax[0].loglog(numels, mu, label="pure_callback", marker=".", color="C0")
ax[0].fill_between(numels, mu - std, mu + std, color="C0", alpha=0.3)
mu, std = [np.mean(ts) for ts in cpu_times_tj], [np.std(ts) for ts in cpu_times_tj]
mu, std = np.array(mu), np.array(std)
ax[0].loglog(numels, mu, label="torch2jax (this package)", marker=".", color="C1")
ax[0].fill_between(numels, mu - std, mu + std, color="C1", alpha=0.3)
mu, std = [np.mean(ts) for ts in cpu_times_torch], [np.std(ts) for ts in cpu_times_torch]
mu, std = np.array(mu), np.array(std)
ax[0].loglog(numels, mu, label="native torch", marker=".", color="C2")
ax[0].fill_between(numels, mu - std, mu + std, color="C2", alpha=0.3)
ax[0].set_xlabel("Data size")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("CPU")
ax[0].grid(True, which="both")
ylim1 = ax[0].get_ylim()

mu, std = [np.mean(ts) for ts in cuda_times_pc], [np.std(ts) for ts in cuda_times_pc]
mu, std = np.array(mu), np.array(std)
ax[1].loglog(numels, mu, label="pure_callback", marker=".", color="C0")
ax[1].fill_between(numels, mu - std, mu + std, color="C0", alpha=0.3)
mu, std = [np.mean(ts) for ts in cuda_times_tj], [np.std(ts) for ts in cuda_times_tj]
mu, std = np.array(mu), np.array(std)
ax[1].loglog(numels, mu, label="torch2jax (this package)", marker=".", color="C1")
ax[1].fill_between(numels, mu - std, mu + std, color="C1", alpha=0.3)
mu, std = [np.mean(ts) for ts in cuda_times_torch], [np.std(ts) for ts in cuda_times_torch]
mu, std = np.array(mu), np.array(std)
ax[1].loglog(numels, mu, label="native torch", marker=".", color="C2")
ax[1].fill_between(numels, mu - std, mu + std, color="C2", alpha=0.3)
ax[1].set_xlabel("Data size")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("CUDA")
ax[1].legend()
ylim2 = ax[1].get_ylim()

ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
ax[0].set_ylim(ylim)
ax[1].set_ylim(ylim)
ax[1].grid(True, which="both")

plt.tight_layout()
plt.savefig("time_difference.png", dpi=200, bbox_inches="tight", pad_inches=0.0)
plt.savefig("time_difference.pdf", dpi=200, bbox_inches="tight", pad_inches=0.0)
plt.show()
