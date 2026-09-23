# -*- coding: utf-8 -*-
"""Prepare sparse conditioning profiles for the seven-class PTE-GAN example.

The reference volume must be stored as (Z, Y, X) with categorical values 1..7.
Unknown cells in the conditioning volume are encoded as 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    from tvtk.api import tvtk, write_data  # type: ignore
    HAS_TVTK = True
except Exception:
    tvtk = write_data = None
    HAS_TVTK = False


def save_vtk_from_zyx(volume_zyx: np.ndarray, vtk_path: Path, scalar_name: str) -> None:
    if not HAS_TVTK:
        print(f"[WARN] tvtk/Mayavi is not installed; skip VTK: {vtk_path}")
        return
    data_xyz = np.asarray(volume_zyx).transpose(2, 1, 0)
    grid = tvtk.ImageData(
        spacing=(1, 1, -1),
        origin=(0, 0, 0),
        dimensions=data_xyz.shape,
    )
    grid.point_data.scalars = np.ravel(data_xyz, order="F")
    grid.point_data.scalars.name = scalar_name
    write_data(grid, str(vtk_path))
    print(f"[VTK] saved: {vtk_path}")


def validate_positions(values: list[int], n: int, axis_name: str) -> None:
    bad = [p for p in values if p < 0 or p >= n]
    if bad:
        raise ValueError(f"{axis_name} positions out of range [0,{n-1}]: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build sparse X/Y conditioning profiles for PTE-GAN.")
    ap.add_argument("--input", default="./dataset/diceng_228_228_228_zyx_change_xiangsu.npy")
    ap.add_argument("--out-dir", default="./Ti")
    ap.add_argument("--xs", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200])
    ap.add_argument("--ys", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200])
    ap.add_argument("--unknown", type=int, default=0)
    ap.add_argument("--write-vtk", action="store_true", help="Also write VTK files when tvtk/Mayavi is available.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(in_path)
    if ref.ndim != 3:
        raise ValueError(f"Expected a 3-D array (Z,Y,X), got shape={ref.shape}")
    if np.any(ref == args.unknown):
        raise ValueError(f"Reference data contain unknown value {args.unknown}; expected categorical labels 1..K only.")

    z, y, x = ref.shape
    validate_positions(args.xs, x, "X")
    validate_positions(args.ys, y, "Y")

    vol = np.full(ref.shape, args.unknown, dtype=ref.dtype)
    mask = np.zeros(ref.shape, dtype=np.uint8)

    for px in args.xs:
        vol[:, :, px] = ref[:, :, px]
        mask[:, :, px] = 1
    for py in args.ys:
        vol[:, py, :] = ref[:, py, :]
        mask[:, py, :] = 1

    base = in_path.stem
    out_npy = out_dir / f"{base}_insert{len(args.xs)+len(args.ys)}_UNKNOWN{args.unknown}.npy"
    out_mask = out_dir / f"{base}_insert{len(args.xs)+len(args.ys)}_mask.npy"
    np.save(out_npy, vol)
    np.save(out_mask, mask)
    print(f"[NPY] saved: {out_npy} | shape={vol.shape}")
    print(f"[NPY] saved: {out_mask} | known voxels={int(mask.sum())}")

    if args.write_vtk:
        save_vtk_from_zyx(vol, out_npy.with_suffix('.vtk'), "lithology")
        save_vtk_from_zyx(mask.astype(np.float32), out_mask.with_suffix('.vtk'), "mask")

    print("[DONE] conditioning volume prepared.")


if __name__ == "__main__":
    main()
