"""Hook the layers and hand back the raw attention.

See CONTRACT.md section 5.6 for the signatures.
"""

from __future__ import annotations

import numpy as np

from src.models.base import AttentionPredictor


def attention_maps(
    model: AttentionPredictor, hist: np.ndarray
) -> dict[str, np.ndarray]:
    """Return module -> the raw token map. No collapse happens here.

    The encoder map is (N*8, N*8). The decoder map is (N*12, N*12) and the
    cross map is (N*12, N*8), because the decoder reads 12 future steps.
    CONTRACT.md section 5.6 names only the encoder shape, and the encoder is
    config.PRIMARY_MODULE. Read the shape from the array.
    """
    maps = model.attention(hist)
    for name, array in maps.items():
        if np.asarray(array).ndim != 2:
            raise ValueError(f"the {name} map must be 2 dimensional")
    return maps
