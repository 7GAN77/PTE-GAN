# -*- coding: utf-8 -*-
"""Generate the small synthetic seven-class volume used by quick_test.py."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def build_volume(shape=(32, 32, 32)) -> np.ndarray:
    z, y, x = shape
    zz, yy, xx = np.meshgrid(
        np.arange(z, dtype=np.float32),
        np.arange(y, dtype=np.float32),
        np.arange(x, dtype=np.float32),
        indexing="ij",
    )
    # Smoothly folded/deformed stratigraphic coordinate.  This is generated data,
    # not one of the manuscript's experimental datasets.
    surface = (
        zz
        + 2.2 * np.sin(2.0 * np.pi * xx / max(x, 1))
        + 1.6 * np.cos(2.0 * np.pi * yy / max(y, 1))
        + 0.8 * np.sin(2.0 * np.pi * (xx + yy) / max(x + y, 1))
    )
    smin, smax = float(surface.min()), float(surface.max())
    u = (surface - smin) / max(smax - smin, 1e-8)
    labels = np.floor(u * 7.0).astype(np.int16) + 1
    labels = np.clip(labels, 1, 7).astype(np.int16)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./example/example_data.npy")
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vol = build_volume((args.size, args.size, args.size))
    np.save(out, vol)
    print(f"[OK] saved {out} shape={vol.shape} labels={np.unique(vol).tolist()}")


if __name__ == "__main__":
    main()
