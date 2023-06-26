from __future__ import annotations

import random

import torch
import numpy as np


def _find_unique_id() -> int:
    while True:
        id = random.randint(0, 2**63)
        if not hasattr(torch, f"_torch2jax_fn_{id}") and not hasattr(
            torch, f"_torch2jax_args_{id}"
        ):
            return id


def _shapes_to_ir_constants(
    in_shapes: list[list[int]], out_shapes: list[list[int]]
) -> list[np.int64]:
    vals = [len(in_shapes)]
    for shape in in_shapes:
        vals.append(len(shape))
        vals.extend(shape)
    vals.append(len(out_shapes))
    for shape in out_shapes:
        vals.append(len(shape))
        vals.extend(shape)
    return [np.int64(val) for val in vals]
