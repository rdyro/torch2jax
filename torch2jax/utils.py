from __future__ import annotations

import random

import torch


def _find_unique_id() -> int:
    while True:
        id = random.randint(0, 2**63)
        if not hasattr(torch, f"_torch2jax_fn_{id}") and not hasattr(
            torch, f"_torch2jax_args_{id}"
        ):
            return id
