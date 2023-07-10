# ResNet 50 example

```python
from __future__ import annotations

from pprint import pprint

from tqdm import tqdm
from datasets import load_dataset
import torch
from torchvision.models.resnet import resnet18
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import jax
from jax import numpy as jnp
import optax

from torch2jax import tree_t2j, torch2jax_with_vjp, tree_j2t, t2j, j2t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_jax = jax.devices(device.type)[0]
```

### Loading the dataset and the model (in PyTorch)

```python
dataset = load_dataset("mnist", split="train")

def collate_torch_fn(batch):
    imgs = torch.stack([ToTensor()(x["image"]).repeat((3, 1, 1)) for x in batch]).to(device)
    labels = torch.tensor([x["label"] for x in batch]).to(device)
    return imgs, labels

collate_jax_fn = lambda batch: tree_t2j(collate_torch_fn(batch))
```

```python
model = nn.Sequential(resnet18(), nn.Linear(1000, 10))
model.to(device)
model.eval()

opts = dict(batch_size=32, shuffle=True, num_workers=0)
dl = DataLoader(dataset, **opts)
dl_jax = DataLoader(dataset, **dict(opts, collate_fn=collate_jax_fn))
dl_torch = DataLoader(dataset, **dict(opts, collate_fn=collate_torch_fn))
```

### Let's convert the torch model to a function, using `torch.func.functional_call`

```python
params, buffers = dict(model.named_parameters()), dict(model.named_buffers())


def torch_fwd_fn(params, buffers, input):
    buffers = {k: torch.clone(v) for k, v in buffers.items()}
    return functional_call(model, (params, buffers), args=input)


Xt, yt = next(iter(dl_torch))
nondiff_argnums = (1, 2)  # buffers, input
jax_fwd_fn = jax.jit(
    torch2jax_with_vjp(torch_fwd_fn, params, buffers, Xt, nondiff_argnums=nondiff_argnums)
)
params_jax, buffers_jax = tree_t2j(params), tree_t2j(buffers)
```

### Let's use torch's CrossEntropyLoss

```python
Xt, yt = next(iter(dl_torch))
torch_ce_fn = lambda yp, y: nn.CrossEntropyLoss()(yp, y)
jax_ce_fn = torch2jax_with_vjp(torch_ce_fn, model(Xt), yt)

jax_l_fn = jax.jit(
    lambda params_jax, X, y: jnp.mean(jax_ce_fn(jax_fwd_fn(params_jax, buffers_jax, X), y))
)
jax_g_fn = jax.jit(jax.grad(jax_l_fn))
torch_g_fn = torch.func.grad(
    lambda params, Xt, yt: torch_ce_fn(torch_fwd_fn(params, buffers, Xt), yt)
)

```

```python
X, y = next(iter(dl_jax))
gs_jax = jax_g_fn(params_jax, X, y)
gs_torch = torch_g_fn(params, *tree_j2t((X, y)))

# let's compute error in gradients between JAX and Torch (the errors are 0!)
errors = {k: float(jnp.linalg.norm(v - t2j(gs_torch[k]))) for k, v in gs_jax.items()}
pprint(errors)
```

<p>
{'0.bn1.bias': 6.606649449736324e-09,<br>
 '0.bn1.weight': 1.0237145575686668e-09,<br>
 '0.conv1.weight': 1.9232666659263487e-07,<br>
 '0.fc.bias': 0.0,<br>
 '0.fc.weight': 0.0,<br>
 '0.layer1.0.bn1.bias': 4.424356436771859e-09,<br>
 '0.layer1.0.bn1.weight': 5.933196711715993e-10,<br>
 '0.layer1.0.bn2.bias': 2.3588471176339e-09,<br>
 '0.layer1.0.bn2.weight': 4.533372566228877e-10,<br>
 '0.layer1.0.conv1.weight': 1.4028480599392879e-08,<br>
 '0.layer1.0.conv2.weight': 1.1964990775936712e-08,<br>
 '0.layer1.1.bn1.bias': 8.75052974524948e-10,<br>
 '0.layer1.1.bn1.weight': 2.0072446482721773e-10,<br>
 '0.layer1.1.bn2.bias': 5.820766091346741e-11,<br>
 '0.layer1.1.bn2.weight': 2.9103830456733704e-11,<br>
 '0.layer1.1.conv1.weight': 1.1259264631746646e-08,<br>
 '0.layer1.1.conv2.weight': 1.1262083710050774e-08,<br>
 '0.layer2.0.bn1.bias': 0.0,<br>
 '0.layer2.0.bn1.weight': 0.0,<br>
 '0.layer2.0.bn2.bias': 0.0,<br>
 '0.layer2.0.bn2.weight': 0.0,<br>
 '0.layer2.0.conv1.weight': 0.0,<br>
 '0.layer2.0.conv2.weight': 0.0,<br>
 '0.layer2.0.downsample.0.weight': 6.819701248161891e-09,<br>
 '0.layer2.0.downsample.1.bias': 0.0,<br>
 '0.layer2.0.downsample.1.weight': 0.0,<br>
 '0.layer2.1.bn1.bias': 0.0,<br>
 '0.layer2.1.bn1.weight': 0.0,<br>
 '0.layer2.1.bn2.bias': 0.0,<br>
 '0.layer2.1.bn2.weight': 5.820766091346741e-11,<br>
 '0.layer2.1.conv1.weight': 0.0,<br>
 '0.layer2.1.conv2.weight': 0.0,<br>
 '0.layer3.0.bn1.bias': 0.0,<br>
 '0.layer3.0.bn1.weight': 0.0,<br>
 '0.layer3.0.bn2.bias': 0.0,<br>
 '0.layer3.0.bn2.weight': 0.0,<br>
 '0.layer3.0.conv1.weight': 0.0,<br>
 '0.layer3.0.conv2.weight': 0.0,<br>
 '0.layer3.0.downsample.0.weight': 0.0,<br>
 '0.layer3.0.downsample.1.bias': 0.0,<br>
 '0.layer3.0.downsample.1.weight': 0.0,<br>
 '0.layer3.1.bn1.bias': 0.0,<br>
 '0.layer3.1.bn1.weight': 0.0,<br>
 '0.layer3.1.bn2.bias': 0.0,<br>
 '0.layer3.1.bn2.weight': 0.0,<br>
 '0.layer3.1.conv1.weight': 0.0,<br>
 '0.layer3.1.conv2.weight': 0.0,<br>
 '0.layer4.0.bn1.bias': 0.0,<br>
 '0.layer4.0.bn1.weight': 0.0,<br>
 '0.layer4.0.bn2.bias': 0.0,<br>
 '0.layer4.0.bn2.weight': 0.0,<br>
 '0.layer4.0.conv1.weight': 0.0,<br>
 '0.layer4.0.conv2.weight': 0.0,<br>
 '0.layer4.0.downsample.0.weight': 0.0,<br>
 '0.layer4.0.downsample.1.bias': 0.0,<br>
 '0.layer4.0.downsample.1.weight': 0.0,<br>
 '0.layer4.1.bn1.bias': 0.0,<br>
 '0.layer4.1.bn1.weight': 0.0,<br>
 '0.layer4.1.bn2.bias': 0.0,<br>
 '0.layer4.1.bn2.weight': 0.0,<br>
 '0.layer4.1.conv1.weight': 0.0,<br>
 '0.layer4.1.conv2.weight': 0.0,<br>
 '1.bias': 0.0,<br>
 '1.weight': 0.0}
</p>

### Train loop 

This isn't very efficient because torch synchronizes for every batch when called
from JAX. Train in PyTorch, but you can do inference in JAX fast.

```python
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params_jax)
update_fn, apply_updates = jax.jit(optimizer.update), jax.jit(optax.apply_updates)
for i, (X, y) in enumerate(tqdm(dl_jax, total=len(dl_jax))):
    gs = jax_g_fn(params_jax, X, y)
    updates, opt_state = update_fn(gs, opt_state)
    params_jax2 = apply_updates(params_jax, updates)
    if i > 10:
        break
```
