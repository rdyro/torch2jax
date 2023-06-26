from __future__ import annotations

import numpy as np

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
