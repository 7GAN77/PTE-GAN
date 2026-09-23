# -*- coding: utf-8 -*-
# @FileName: 0generate_cascade_insert10_1to3_until_23to25_in120_150_150.py
# @Time : 2026/5/10 下午5:18
# @Software: PyCharm
3

"""
【功能】
在 (Z,Y,X)=(120,150,150) 的体数据上，逐级调用训练好的 GAN，完成：
1) 对 split_json 中全部测试剖面（fixed_test_pairs + extra_test_pairs）的“单剖面独立级联”：
   每个测试剖面都单独从厚度=1 的中心剖面开始，依次经过 1→3、3→5、...、21→23、23→25，
   最终得到“仅对应这一条剖面”的厚度=25 单独体。
2) 仅对 fixed_test_pairs（通常是 x/y 各 5 条，共 10 条）做“累计回填大模型”：
   也就是保留原来 insert10 的思路，把 fixed 的 10 条剖面依次扩厚到 25 后，再共同插入同一个大模型。
3) 记录每个剖面从“单个中心剖面 → 厚度 25 单独体”的总耗时与分阶段耗时。

【这版脚本的核心逻辑】
A. 所有测试对（fixed + extra）都要跑单剖面的 1→25 级联，并各自保存结果。
B. 只有 fixed_test_pairs 参与最终大模型的累计回填；extra_test_pairs 不参与 final 大模型。
C. 单剖面结果不再建立在“10 条剖面同时存在的 input_volume”之上，而是为每个测试对单独构造
   一个“只含该中心剖面，其余全为 unknown=0”的基础体，然后再把厚度=25 的结果回填进去。
D. final 大模型仍然沿用 insert10 的思路：以 input_volume_npy 为底，只累计 fixed_test_pairs。

【默认输入】
- input_volume_npy : ./Ti/fold_categorical_180x150x120_zyx_change_xiangsu_insert10_UNKNOWN0.npy
- reference_volume_npy : ./dataset/fold_categorical_180x150x120_zyx_change_xiangsu.npy
- split_json : ./Ti/split_pairs_fixed_test_xy_25_50_75_100_125_seed1234.json
- best.pt : ./checkpoints_1to3_cgan_3scales/best.pt 到 ./checkpoints_23to25_cgan_3scales/best.pt
- train_pairs_*.py : 用于动态导入 Generator 类

【输出目录】
./output/
  stage1_1to3/<tag>/
  stage2_3to5/<tag>/
  ...
  stage10_19to21/<tag>/
  stage11_21to23/<tag>/
  stage12_23to25/<tag>/
  untili_23to25/<tag>/
    <tag>_vol_backfilled.npy
    <tag>_vol_backfilled.vtk
    <tag>_vol_backfilled_reference.npy
    <tag>_vol_backfilled_reference.vtk
    <tag>_meta.json
  untili_23to25/
    runtime_summary_all_test_pairs.json
    runtime_summary_all_test_pairs.csv
  untili_23to25/final/
    final_volume_backfilled.npy
    final_volume_backfilled.vtk
    final_volume_backfilled_reference.npy
    final_volume_backfilled_reference.vtk
    final_meta.json
    runtime_summary_fixed_pairs.json
    runtime_summary_fixed_pairs.csv

注意：
- 体数据按 (Z,Y,X) 存储：volume[z, y, x]
- unknown 约定为 0；已知类别为 1..K
- split_json 中 axis=0 表示 X-slab，axis=1 表示 Y-slab
"""

import os
import csv
import json
import time
import argparse
import sys
import shlex
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch


# ============================================================
# VTK writer：输入 volume_zyx (Z,Y,X)，写出 STRUCTURED_POINTS
# scalar_name 固定 "lithology"
# ============================================================
try:
    from tvtk.api import tvtk, write_data  # type: ignore
    _HAS_TVTK = True
except Exception:
    tvtk, write_data = None, None
    _HAS_TVTK = False


STAGE_SPECS: List[Dict[str, Any]] = [
    {"idx": 1,  "name": "1to3",   "dout": 3,  "model_arg": "model_1to3",   "train_arg": "train_1to3_py",   "base_arg": "base_1to3",   "module_name": "train_pairs_1to3_mod",   "stage_dir": "stage1_1to3"},
    {"idx": 2,  "name": "3to5",   "dout": 5,  "model_arg": "model_3to5",   "train_arg": "train_3to5_py",   "base_arg": "base_3to5",   "module_name": "train_pairs_3to5_mod",   "stage_dir": "stage2_3to5"},
    {"idx": 3,  "name": "5to7",   "dout": 7,  "model_arg": "model_5to7",   "train_arg": "train_5to7_py",   "base_arg": "base_5to7",   "module_name": "train_pairs_5to7_mod",   "stage_dir": "stage3_5to7"},
    {"idx": 4,  "name": "7to9",   "dout": 9,  "model_arg": "model_7to9",   "train_arg": "train_7to9_py",   "base_arg": "base_7to9",   "module_name": "train_pairs_7to9_mod",   "stage_dir": "stage4_7to9"},
    {"idx": 5,  "name": "9to11",  "dout": 11, "model_arg": "model_9to11",  "train_arg": "train_9to11_py",  "base_arg": "base_9to11",  "module_name": "train_pairs_9to11_mod",  "stage_dir": "stage5_9to11"},
    {"idx": 6,  "name": "11to13", "dout": 13, "model_arg": "model_11to13", "train_arg": "train_11to13_py", "base_arg": "base_11to13", "module_name": "train_pairs_11to13_mod", "stage_dir": "stage6_11to13"},
    {"idx": 7,  "name": "13to15", "dout": 15, "model_arg": "model_13to15", "train_arg": "train_13to15_py", "base_arg": "base_13to15", "module_name": "train_pairs_13to15_mod", "stage_dir": "stage7_13to15"},
    {"idx": 8,  "name": "15to17", "dout": 17, "model_arg": "model_15to17", "train_arg": "train_15to17_py", "base_arg": "base_15to17", "module_name": "train_pairs_15to17_mod", "stage_dir": "stage8_15to17"},
    {"idx": 9,  "name": "17to19", "dout": 19, "model_arg": "model_17to19", "train_arg": "train_17to19_py", "base_arg": "base_17to19", "module_name": "train_pairs_17to19_mod", "stage_dir": "stage9_17to19"},
    {"idx": 10, "name": "19to21", "dout": 21, "model_arg": "model_19to21", "train_arg": "train_19to21_py", "base_arg": "base_19to21", "module_name": "train_pairs_19to21_mod", "stage_dir": "stage10_19to21"},
    {"idx": 11, "name": "21to23", "dout": 23, "model_arg": "model_21to23", "train_arg": "train_21to23_py", "base_arg": "base_21to23", "module_name": "train_pairs_21to23_mod", "stage_dir": "stage11_21to23"},
    {"idx": 12, "name": "23to25", "dout": 25, "model_arg": "model_23to25", "train_arg": "train_23to25_py", "base_arg": "base_23to25", "module_name": "train_pairs_23to25_mod", "stage_dir": "stage12_23to25"},
]

# 最终阶段统一由 STAGE_SPECS 的最后一项决定；当前为 23→25。
# 后续若继续扩展到 25→27，只需要在 STAGE_SPECS 末尾追加阶段即可。
FINAL_STAGE_NAME = STAGE_SPECS[-1]["name"]
FINAL_THICKNESS = int(STAGE_SPECS[-1]["dout"])
UNTILI_DIR_NAME = f"untili_{FINAL_STAGE_NAME}"


def save_vtk_legacy_from_zyx(
    volume_zyx: np.ndarray,
    vtk_path: str,
    scalar_name: str = "lithology",
    spacing=(1, 1, -1),
    origin=(0, 0, 0),
) -> None:
    os.makedirs(os.path.dirname(vtk_path) or ".", exist_ok=True)
    vol_zyx = np.asarray(volume_zyx)
    assert vol_zyx.ndim == 3

    data_xyz = vol_zyx.transpose(2, 1, 0)  # (X,Y,Z)
    nx, ny, nz = data_xyz.shape
    flat = np.ravel(data_xyz, order="F")

    vtk_dtype = "int" if np.issubdtype(data_xyz.dtype, np.integer) else "float"
    flat = flat.astype(np.int32 if vtk_dtype == "int" else np.float32, copy=False)

    header = [
        "# vtk DataFile Version 3.0",
        "generated by generate_cascade_insert10_1to3_until_23to25_in120_150_150.py",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx} {ny} {nz}",
        f"ORIGIN {origin[0]} {origin[1]} {origin[2]}",
        f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}",
        f"POINT_DATA {nx * ny * nz}",
        f"SCALARS {scalar_name} {vtk_dtype} 1",
        "LOOKUP_TABLE default",
    ]
    with open(vtk_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if vtk_dtype == "int":
            for v in flat:
                f.write(f"{int(v)}\n")
        else:
            for v in flat:
                f.write(f"{float(v)}\n")



def save_vtk_tvtk_from_zyx(
    volume_zyx: np.ndarray,
    vtk_path: str,
    scalar_name: str = "lithology",
    spacing=(1, 1, -1),
    origin=(0, 0, 0),
) -> None:
    if not _HAS_TVTK:
        raise RuntimeError("tvtk is not available.")
    os.makedirs(os.path.dirname(vtk_path) or ".", exist_ok=True)

    vol_zyx = np.asarray(volume_zyx)
    assert vol_zyx.ndim == 3

    data_xyz = vol_zyx.transpose(2, 1, 0)  # (X,Y,Z)
    flat = np.ravel(
        data_xyz.astype(np.int32 if np.issubdtype(data_xyz.dtype, np.integer) else np.float32, copy=False),
        order="F",
    )
    grid = tvtk.ImageData(spacing=spacing, origin=origin, dimensions=data_xyz.shape)
    grid.point_data.scalars = flat
    grid.point_data.scalars.name = scalar_name
    write_data(grid, vtk_path)



def save_vtk_from_zyx(
    volume_zyx: np.ndarray,
    vtk_path: str,
    scalar_name: str = "lithology",
    spacing=(1, 1, -1),
    origin=(0, 0, 0),
    prefer_tvtk: bool = True,
) -> None:
    if prefer_tvtk and _HAS_TVTK:
        save_vtk_tvtk_from_zyx(volume_zyx, vtk_path, scalar_name, spacing=spacing, origin=origin)
    else:
        save_vtk_legacy_from_zyx(volume_zyx, vtk_path, scalar_name, spacing=spacing, origin=origin)


# ============================================================
# 小工具
# ============================================================
def ensure_int16(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(np.int16, copy=False)



def ensure_uint8(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(np.uint8, copy=False)



def slab_dhw_to_zyx(slab_dhw: np.ndarray, axis: int) -> np.ndarray:
    """把 axis=0/1 的薄体 slab(D,H,W) 转成 volume_zyx 形式，便于保存 vtk。"""
    slab = np.asarray(slab_dhw)
    assert slab.ndim == 3
    if axis == 0:
        return slab.transpose(1, 2, 0)  # (D,Z,Y)->(Z,Y,D)
    if axis == 1:
        return slab.transpose(1, 0, 2)  # (D,Z,X)->(Z,D,X)
    raise ValueError("axis must be 0 or 1")



def make_offsets(dout: int) -> np.ndarray:
    """例如 dout=21 -> [-10, ..., 0, ..., 10]。"""
    if dout % 2 != 1:
        raise ValueError(f"dout must be odd, got {dout}")
    r = (dout - 1) // 2
    return np.arange(-r, r + 1, dtype=np.float32)



def pair_tag(axis: int, pos: int) -> str:
    return f"x_{pos}" if axis == 0 else f"y_{pos}"



def extract_center_slice(volume_zyx: np.ndarray, axis: int, pos: int) -> np.ndarray:
    vol = ensure_int16(volume_zyx)
    if axis == 0:
        return vol[:, :, pos]
    if axis == 1:
        return vol[:, pos, :]
    raise ValueError("axis must be 0 or 1")



def center_slice_nonempty(center_hw: np.ndarray, min_known_ratio: float = 0.95) -> bool:
    """
    判断一张中心剖面是否“足够完整”，而不是只要出现过非零值就算可用。
    这样可以避免 extra 剖面仅因和固定剖面相交，留下几条非零线，
    就被误判为“完整中心剖面已存在于 input 中”。

    参数
    ----
    min_known_ratio : float
        已知像元(>0)占整张剖面的最小比例阈值。
        对 insert10 / insert16 这类“固定剖面完整插入”的场景，0.95 较稳妥。
    """
    arr = np.asarray(center_hw)
    if arr.size == 0:
        return False

    known_ratio = float((arr > 0).sum()) / float(arr.size)
    return known_ratio >= float(min_known_ratio)



def choose_center_slice(
    base_input_zyx: np.ndarray,
    reference_zyx: np.ndarray,
    axis: int,
    pos: int,
    center_source: str,
) -> Tuple[np.ndarray, str]:
    """
    选择中心已知剖面的来源：
    - input: 强制用 input_volume
    - reference: 强制用 reference_volume
    - auto: 仅当 input 中该中心剖面“足够完整”时才使用 input，否则退回 reference
    """
    source = center_source.lower().strip()
    center_in = extract_center_slice(base_input_zyx, axis, pos)
    center_ref = extract_center_slice(reference_zyx, axis, pos)

    if source == "input":
        return ensure_int16(center_in), "input"
    if source == "reference":
        return ensure_int16(center_ref), "reference"
    if source == "auto":
        if center_slice_nonempty(center_in):
            return ensure_int16(center_in), "input"
        return ensure_int16(center_ref), "reference"
    raise ValueError(f"Unsupported center_source: {center_source}")



def build_single_section_base_volume(
    shape_zyx: Tuple[int, int, int],
    center_hw: np.ndarray,
    axis: int,
    pos: int,
    unknown_val: int = 0,
) -> np.ndarray:
    """
    为“单个测试剖面独立生成厚度=25 结果”构造基础体：
    - 只保留当前这一条中心剖面
    - 其余全部置为 unknown_val
    """
    Z, Y, X = [int(v) for v in shape_zyx]
    base = np.full((Z, Y, X), int(unknown_val), dtype=np.int16)
    center = ensure_int16(center_hw)

    if axis == 0:
        if center.shape != (Z, Y):
            raise ValueError(f"center shape mismatch for X-slab: expect {(Z, Y)}, got {center.shape}")
        base[:, :, pos] = center
        return base

    if axis == 1:
        if center.shape != (Z, X):
            raise ValueError(f"center shape mismatch for Y-slab: expect {(Z, X)}, got {center.shape}")
        base[:, pos, :] = center
        return base

    raise ValueError("axis must be 0 or 1")


# ============================================================
# 构造 2.5D 输入：onehot(K)+mask+offset+axis -> (1, D*(K+3), H, W)
# ============================================================
def build_input_tensor_D(
    cond_dhw: np.ndarray,
    mask_dhw: np.ndarray,
    K: int,
    axis_flag: float,
    offsets: np.ndarray,
) -> torch.Tensor:
    cond = np.asarray(cond_dhw).astype(np.int64)
    mask = np.asarray(mask_dhw).astype(np.uint8)
    if cond.shape != mask.shape:
        raise ValueError("cond/mask shape mismatch")
    D, H, W = cond.shape
    if offsets.shape[0] != D:
        raise ValueError("offset length mismatch D")

    onehot = np.zeros((K, D, H, W), dtype=np.float32)
    known = (mask == 1)
    if known.any():
        lab = cond[known].astype(np.int64) - 1
        lab = np.clip(lab, 0, K - 1)
        d_idx, h_idx, w_idx = np.where(known)
        onehot[lab, d_idx, h_idx, w_idx] = 1.0

    mask_f = mask.astype(np.float32)
    off = offsets.astype(np.float32)[:, None, None]
    off_dhw = np.broadcast_to(off, (D, H, W)).copy()
    axis_dhw = np.full((D, H, W), float(axis_flag), dtype=np.float32)

    x_4d = np.concatenate(
        [onehot, mask_f[None, ...], off_dhw[None, ...], axis_dhw[None, ...]],
        axis=0,
    )  # (K+3, D, H, W)

    x_2d = x_4d.transpose(1, 0, 2, 3).reshape(D * (K + 3), H, W)
    return torch.from_numpy(x_2d[None, ...]).float()


# ============================================================
# 动态导入训练脚本，拿到 Generator 类，并加载 best.pt
# ============================================================
def import_train_module(train_py: str, module_name: str) -> Any:
    from importlib.util import spec_from_file_location, module_from_spec
    p = Path(train_py).resolve()
    if not p.exists():
        raise FileNotFoundError(f"train_py not found: {p}")
    spec = spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {p}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod



def pick_generator_class(train_mod: Any) -> Any:
    for name in ["UNet2D", "Generator", "NetG", "GNet", "Gen", "UNet"]:
        if hasattr(train_mod, name):
            cls = getattr(train_mod, name)
            if isinstance(cls, type):
                return cls
    raise RuntimeError("Cannot find generator class in training script.")



def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")
    if "G_ema" in ckpt and isinstance(ckpt["G_ema"], dict):
        return ckpt["G_ema"]
    if "G" in ckpt and isinstance(ckpt["G"], dict):
        return ckpt["G"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt



def _infer_base(sd: Dict[str, torch.Tensor], fallback: int = 48) -> int:
    for k in ["enc1.conv1.weight", "module.enc1.conv1.weight"]:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) >= 1:
            return int(sd[k].shape[0])
    return int(fallback)



def instantiate_generator(gen_cls: Any, in_ch: int, out_ch: int, base: int) -> torch.nn.Module:
    for try_fn in (
        lambda: gen_cls(in_ch=in_ch, out_ch=out_ch, base=base),
        lambda: gen_cls(in_ch, out_ch, base=base),
        lambda: gen_cls(in_ch, out_ch, base),
        lambda: gen_cls(in_ch, out_ch),
        lambda: gen_cls(),
    ):
        try:
            return try_fn()
        except Exception:
            pass
    raise RuntimeError("Failed to instantiate generator; check train script signature.")



def load_generator(
    model_pt: str,
    train_py: str,
    device: torch.device,
    in_ch: int,
    out_ch: int,
    base_override: Optional[int],
    module_name: str,
) -> torch.nn.Module:
    ckpt = torch.load(model_pt, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    train_mod = import_train_module(train_py, module_name=module_name)
    gen_cls = pick_generator_class(train_mod)

    base = _infer_base(sd, fallback=48)
    if base_override is not None:
        base = int(base_override)

    net = instantiate_generator(gen_cls, in_ch=in_ch, out_ch=out_ch, base=base)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict strict=False; keys mismatch:")
        if missing:
            print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    net.to(device).eval()
    return net


# ============================================================
# 条件构造与推理
# ============================================================
def make_1to3_cond_mask_from_center_slice(center_hw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1→3：仅中间层已知。"""
    sl = ensure_int16(center_hw)
    H, W = sl.shape
    cond3 = np.zeros((3, H, W), dtype=np.int16)
    mask3 = np.zeros((3, H, W), dtype=np.uint8)
    cond3[1] = sl
    mask3[1][sl > 0] = 1
    return cond3, mask3



def make_next_cond_mask_from_full(prev_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用规律：Din→Dout=Din+2。
    把前一阶段的完整 slab 放到新 slab 的中间 Din 层，两侧各留 1 层未知。
    """
    prev = ensure_int16(prev_full)
    Din, H, W = prev.shape
    Dout = Din + 2
    cond = np.zeros((Dout, H, W), dtype=np.int16)
    mask = np.zeros((Dout, H, W), dtype=np.uint8)
    cond[1:1 + Din] = prev
    mask[1:1 + Din] = 1
    return cond, mask


@torch.no_grad()
def infer_D(
    G: torch.nn.Module,
    cond_dhw: np.ndarray,
    mask_dhw: np.ndarray,
    K: int,
    axis_flag: float,
    offsets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    cond = ensure_int16(cond_dhw)
    mask = ensure_uint8(mask_dhw)
    D, H, W = cond.shape

    x = build_input_tensor_D(cond, mask, K=K, axis_flag=axis_flag, offsets=offsets).to(device)
    logits = G(x)  # (1, K*D, H, W)
    if logits.ndim != 4 or logits.shape[1] != K * D:
        raise RuntimeError(f"logits shape unexpected: {tuple(logits.shape)}")

    logits_kdhw = logits.view(1, K, D, H, W)
    pred0 = torch.argmax(logits_kdhw, dim=1).squeeze(0).cpu().numpy().astype(np.int16)  # 0..K-1
    pred = (pred0 + 1).astype(np.int16)  # 1..K

    known = (mask == 1)
    if known.any():
        pred[known] = cond[known]
    return pred


# ============================================================
# 参考薄体：从 reference volume 切出对应厚度块
# ============================================================
def cut_reference_slab(reference_vol_zyx: np.ndarray, axis: int, pos: int, offsets: List[int]) -> np.ndarray:
    ref = ensure_int16(reference_vol_zyx)
    Z, Y, X = ref.shape

    if axis == 0:
        slab = np.zeros((len(offsets), Z, Y), dtype=np.int16)
        for i, off in enumerate(offsets):
            xx = pos + off
            if not (0 <= xx < X):
                raise ValueError(f"reference x index out of range: x={xx}")
            slab[i] = ref[:, :, xx]
        return slab

    if axis == 1:
        slab = np.zeros((len(offsets), Z, X), dtype=np.int16)
        for i, off in enumerate(offsets):
            yy = pos + off
            if not (0 <= yy < Y):
                raise ValueError(f"reference y index out of range: y={yy}")
            slab[i] = ref[:, yy, :]
        return slab

    raise ValueError("axis must be 0 or 1")


# ============================================================
# split_json：读取 fixed / extra / all 测试集对 (axis,pos)
# ============================================================
def normalize_pairs(raw_pairs: List[Any]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for item in raw_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid pair in split_json: {item}")
        a = int(item[0])
        p = int(item[1])
        if a not in (0, 1):
            raise ValueError(f"axis must be 0/1, got {a}")
        key = (a, p)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out



def load_split_pair_groups(split_json: str) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, Any]]:
    """
    读取 split_json，并同时返回：
    - fixed_test_pairs
    - extra_test_pairs
    - all_test_pairs
    """
    p = Path(split_json)
    if not p.exists():
        raise FileNotFoundError(f"split_json not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        info = json.load(f)

    fixed_pairs = normalize_pairs(info.get("fixed_test_pairs", []))
    extra_pairs = normalize_pairs(info.get("extra_test_pairs", []))

    if "test_pairs_all" in info:
        all_pairs = normalize_pairs(info["test_pairs_all"])
    else:
        all_pairs = normalize_pairs(fixed_pairs + extra_pairs)

    fixed_pairs = sorted(fixed_pairs, key=lambda x: (x[0], x[1]))
    extra_pairs = sorted(extra_pairs, key=lambda x: (x[0], x[1]))
    all_pairs = sorted(all_pairs, key=lambda x: (x[0], x[1]))

    pair_groups = {
        "fixed_test_pairs": fixed_pairs,
        "extra_test_pairs": extra_pairs,
        "all_test_pairs": all_pairs,
    }
    return pair_groups, info



def build_pairs_from_xy_lists(xs: List[int], ys: List[int]) -> List[Tuple[int, int]]:
    pairs = [(0, int(p)) for p in xs] + [(1, int(p)) for p in ys]
    pairs = sorted(list(set(pairs)), key=lambda x: (x[0], x[1]))
    return pairs



def filter_valid_pairs(pairs: List[Tuple[int, int]], x_len: int, y_len: int, note: str) -> List[Tuple[int, int]]:
    valid_pairs: List[Tuple[int, int]] = []
    for axis, pos in pairs:
        if axis == 0 and 0 <= pos < x_len:
            valid_pairs.append((axis, pos))
        elif axis == 1 and 0 <= pos < y_len:
            valid_pairs.append((axis, pos))
        else:
            print(f"[WARN] skip out-of-range pair from {note}: axis={axis}, pos={pos}")
    valid_pairs = sorted(list(dict.fromkeys(valid_pairs)), key=lambda x: (x[0], x[1]))
    return valid_pairs


# ============================================================
# stage 结果保存
# ============================================================
def save_stage_per_tag(
    stage_dir: Path,
    tag: str,
    axis: int,
    stage_name: str,
    gan: np.ndarray,
    cond: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray,
    prefer_tvtk: bool,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    sub = stage_dir / tag
    sub.mkdir(parents=True, exist_ok=True)

    np.save(sub / f"{tag}_gan.npy", ensure_int16(gan))
    np.save(sub / f"{tag}_cond.npy", ensure_int16(cond))
    np.save(sub / f"{tag}_mask.npy", ensure_uint8(mask))
    np.save(sub / f"{tag}_reference.npy", ensure_int16(reference))

    save_vtk_from_zyx(slab_dhw_to_zyx(gan, axis), str(sub / f"{tag}_gan.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)
    save_vtk_from_zyx(slab_dhw_to_zyx(cond, axis), str(sub / f"{tag}_cond.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)
    save_vtk_from_zyx(slab_dhw_to_zyx(mask, axis), str(sub / f"{tag}_mask.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)
    save_vtk_from_zyx(slab_dhw_to_zyx(reference, axis), str(sub / f"{tag}_reference.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)

    meta = {
        "axis": int(axis),
        "stage": stage_name,
        "shape_gan": list(np.asarray(gan).shape),
        "shape_cond": list(np.asarray(cond).shape),
        "shape_mask": list(np.asarray(mask).shape),
        "shape_reference": list(np.asarray(reference).shape),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(sub / f"{tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============================================================
# 回填：把单个薄体插入到体数据副本中，只覆盖 unknown=0
# ============================================================
def backfill_single_section_into_volume(
    base_volume_zyx: np.ndarray,
    slab_dhw: np.ndarray,
    axis: int,
    pos: int,
    unknown_val: int = 0,
) -> np.ndarray:
    base = ensure_int16(base_volume_zyx)
    vol_new = base.copy()

    Z, Y, X = vol_new.shape
    slab = ensure_int16(slab_dhw)
    D = int(slab.shape[0])
    assert D % 2 == 1, f"D must be odd, got D={D}"
    r = (D - 1) // 2

    if axis == 0:
        slab_zyx = slab_dhw_to_zyx(slab, axis=0)  # (Z,Y,D)
        for k in range(D):
            xx = pos - r + k
            if not (0 <= xx < X):
                continue
            unk = (vol_new[:, :, xx] == unknown_val)
            if unk.any():
                vol_new[:, :, xx][unk] = slab_zyx[:, :, k][unk]
    elif axis == 1:
        slab_zyx = slab_dhw_to_zyx(slab, axis=1)  # (Z,D,X)
        for k in range(D):
            yy = pos - r + k
            if not (0 <= yy < Y):
                continue
            unk = (vol_new[:, yy, :] == unknown_val)
            if unk.any():
                vol_new[:, yy, :][unk] = slab_zyx[:, k, :][unk]
    else:
        raise ValueError("axis must be 0 or 1")

    return vol_new



def save_untili_tag_outputs(
    untili_dir: Path,
    tag: str,
    axis: int,
    pos: int,
    vol_gan_zyx: np.ndarray,
    vol_ref_zyx: np.ndarray,
    prefer_tvtk: bool,
    runtime_info: Dict[str, Any],
) -> None:
    sub = untili_dir / tag
    sub.mkdir(parents=True, exist_ok=True)

    np.save(sub / f"{tag}_vol_backfilled.npy", ensure_int16(vol_gan_zyx))
    save_vtk_from_zyx(ensure_int16(vol_gan_zyx), str(sub / f"{tag}_vol_backfilled.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)

    np.save(sub / f"{tag}_vol_backfilled_reference.npy", ensure_int16(vol_ref_zyx))
    save_vtk_from_zyx(ensure_int16(vol_ref_zyx), str(sub / f"{tag}_vol_backfilled_reference.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)

    meta = {
        "axis": int(axis),
        "pos": int(pos),
        "tag": tag,
        "note": "SINGLE-section backfill (NOT cumulative across different test pairs). This output volume is built on a base volume that contains only the current center section as hard data; then the thickness=25 slab is backfilled only where voxel==0.",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta.update(runtime_info)

    with open(sub / f"{tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)



def save_runtime_summary(out_dir: Path, rows: List[Dict[str, Any]], stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if not rows:
        return

    fieldnames: List[str] = []
    common_keys = [
        "tag", "axis", "pos", "pair_index", "pair_group", "used_in_fixed_final",
        "center_source", "final_thickness", "cascade_runtime_sec", "cascade_runtime_min"
    ]
    for k in common_keys:
        if any(k in row for row in rows):
            fieldnames.append(k)

    stage_keys: List[str] = []
    for spec in STAGE_SPECS:
        stage_keys.extend([
            f"{spec['name']}_sec",
            f"{spec['name']}_min",
        ])
    for k in stage_keys:
        if any(k in row for row in rows):
            fieldnames.append(k)

    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})



def save_untili_final(
    untili_dir: Path,
    final_gan_zyx: np.ndarray,
    final_ref_zyx: np.ndarray,
    prefer_tvtk: bool,
    final_meta_extra: Dict[str, Any],
) -> None:
    sub = untili_dir / "final"
    sub.mkdir(parents=True, exist_ok=True)

    np.save(sub / "final_volume_backfilled.npy", ensure_int16(final_gan_zyx))
    save_vtk_from_zyx(ensure_int16(final_gan_zyx), str(sub / "final_volume_backfilled.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)

    np.save(sub / "final_volume_backfilled_reference.npy", ensure_int16(final_ref_zyx))
    save_vtk_from_zyx(ensure_int16(final_ref_zyx), str(sub / "final_volume_backfilled_reference.vtk"),
                      scalar_name="lithology", prefer_tvtk=prefer_tvtk)

    meta = {
        "note": "CUMULATIVE final volume for fixed_test_pairs only (GAN + reference). The script starts from the original insert10 input volume and then backfills only the fixed sections into the same base volume copy.",
        "shape": list(np.asarray(final_gan_zyx).shape),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta.update(final_meta_extra)
    with open(sub / "final_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============================================================
# 主流程
# ============================================================
def run_cascade(
    input_volume_npy: str,
    reference_volume_npy: str,
    split_json: str,
    out_dir: str,
    device: str,
    K: int,
    center_source: str,
    prefer_tvtk: bool,
    ignore_split_json: bool,
    xs: List[int],
    ys: List[int],
    **kwargs: Any,
) -> None:
    out_root = Path(out_dir).resolve()
    stage_dirs: Dict[str, Path] = {}
    for spec in STAGE_SPECS:
        stage_dirs[spec["name"]] = out_root / spec["stage_dir"]
        stage_dirs[spec["name"]].mkdir(parents=True, exist_ok=True)
    untili_dir = out_root / UNTILI_DIR_NAME
    untili_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device = {device_obj}")

    base_input = ensure_int16(np.load(input_volume_npy))
    ref_vol = ensure_int16(np.load(reference_volume_npy))
    if base_input.shape != ref_vol.shape:
        raise ValueError(f"input/reference shape mismatch: {base_input.shape} vs {ref_vol.shape}")

    Z, Y, X = base_input.shape
    print(f"[INFO] base input shape (Z,Y,X) = {base_input.shape}  (unknown=0)")
    print(f"[INFO] reference shape (Z,Y,X) = {ref_vol.shape}  (all known)")
    print(f"[INFO] output dir = {out_root}")

    split_info: Dict[str, Any] = {}
    if ignore_split_json:
        fixed_pairs = build_pairs_from_xy_lists(xs, ys)
        extra_pairs: List[Tuple[int, int]] = []
        all_pairs = fixed_pairs.copy()
        pair_source_note = "manual_xs_ys_as_fixed_pairs"
        print(f"[INFO] ignore_split_json=True -> use xs/ys as fixed pairs only, total fixed={len(fixed_pairs)}")
    else:
        pair_groups, split_info = load_split_pair_groups(split_json)
        fixed_pairs = pair_groups["fixed_test_pairs"]
        extra_pairs = pair_groups["extra_test_pairs"]
        all_pairs = pair_groups["all_test_pairs"]
        pair_source_note = f"split_json ({Path(split_json).resolve()})"
        print(f"[INFO] split_json loaded: {split_json}")
        print(f"[INFO] fixed_test_pairs = {len(fixed_pairs)}")
        print(f"[INFO] extra_test_pairs = {len(extra_pairs)}")
        print(f"[INFO] all_test_pairs = {len(all_pairs)}")

    fixed_pairs = filter_valid_pairs(fixed_pairs, X, Y, note="fixed_test_pairs")
    extra_pairs = filter_valid_pairs(extra_pairs, X, Y, note="extra_test_pairs")
    all_pairs = filter_valid_pairs(all_pairs, X, Y, note="all_test_pairs")

    if not all_pairs:
        raise RuntimeError("No valid test pairs to process.")

    fixed_set = set(fixed_pairs)
    extra_set = set(extra_pairs)
    all_set = set(all_pairs)

    if not ignore_split_json and fixed_set.union(extra_set) and all_set != fixed_set.union(extra_set):
        print("[WARN] all_test_pairs differs from fixed_test_pairs ∪ extra_test_pairs. The script will process all_test_pairs, but only fixed_test_pairs will be used for final cumulative model.")

    loaded_gens: Dict[str, torch.nn.Module] = {}
    for spec in STAGE_SPECS:
        model_pt = kwargs[spec["model_arg"]]
        train_py = kwargs[spec["train_arg"]]
        base_override = kwargs.get(spec["base_arg"], None)
        dout = int(spec["dout"])

        G = load_generator(
            model_pt=model_pt,
            train_py=train_py,
            device=device_obj,
            in_ch=dout * (K + 3),
            out_ch=K * dout,
            base_override=base_override,
            module_name=spec["module_name"],
        )
        loaded_gens[spec["name"]] = G
        print(f"[OK] {spec['name']} generator loaded.")

    offsets_map: Dict[str, np.ndarray] = {spec["name"]: make_offsets(int(spec["dout"])) for spec in STAGE_SPECS}

    final_gan_fixed = base_input.copy()
    final_ref_fixed = base_input.copy()
    runtime_rows_all: List[Dict[str, Any]] = []
    runtime_rows_fixed: List[Dict[str, Any]] = []

    def process_one(pair_index: int, axis: int, pos: int) -> None:
        nonlocal final_gan_fixed, final_ref_fixed

        tag = pair_tag(axis, pos)
        axis_flag = 0.0 if axis == 0 else 1.0
        is_fixed = (axis, pos) in fixed_set
        is_extra = (axis, pos) in extra_set
        pair_group = "fixed" if is_fixed else ("extra" if is_extra else "test")

        # 关键修复：
        # fixed 剖面本来就完整插入在 input_volume 中，优先使用 input；
        # extra 剖面在 input_volume 中往往只有与 fixed 剖面的交线，
        # 若继续用 auto，会把“少量非零交线”误判为完整中心剖面。
        # 因此 extra 强制改用 reference 中的完整中心剖面。
        if is_fixed:
            local_center_source = "input"
        elif is_extra:
            local_center_source = "reference"
        else:
            local_center_source = center_source

        center_hw_gan, center_used = choose_center_slice(
            base_input, ref_vol, axis, pos, center_source=local_center_source
        )
        center_hw_ref = extract_center_slice(ref_vol, axis, pos)

        single_base_gan = build_single_section_base_volume(
            base_input.shape, center_hw_gan, axis, pos, unknown_val=0
        )
        single_base_ref = build_single_section_base_volume(
            base_input.shape, center_hw_ref, axis, pos, unknown_val=0
        )

        print("-" * 90)
        print(f"[PROCESS] #{pair_index + 1}/{len(all_pairs)}  {tag}  axis={axis} pos={pos}  group={pair_group}  center_source={center_used}")

        section_t0 = time.perf_counter()
        stage_times_sec: Dict[str, float] = {}

        st0 = time.perf_counter()
        cond, mask = make_1to3_cond_mask_from_center_slice(center_hw_gan)
        prev_gan = infer_D(loaded_gens["1to3"], cond, mask, K, axis_flag, offsets_map["1to3"], device_obj)
        ref_slab = cut_reference_slab(ref_vol, axis, pos, offsets_map["1to3"].astype(np.int32).tolist())
        stage_times_sec["1to3"] = time.perf_counter() - st0
        save_stage_per_tag(
            stage_dirs["1to3"], tag, axis, "1to3", prev_gan, cond, mask, ref_slab, prefer_tvtk,
            extra_meta={
                "pos": int(pos),
                "pair_group": pair_group,
                "used_in_fixed_final": bool(is_fixed),
                "center_source": center_used,
                "runtime_sec": float(stage_times_sec["1to3"]),
                "runtime_min": float(stage_times_sec["1to3"] / 60.0),
            }
        )

        for spec in STAGE_SPECS[1:]:
            stage_name = spec["name"]
            st0 = time.perf_counter()
            cond, mask = make_next_cond_mask_from_full(prev_gan)
            prev_gan = infer_D(loaded_gens[stage_name], cond, mask, K, axis_flag, offsets_map[stage_name], device_obj)
            ref_slab = cut_reference_slab(ref_vol, axis, pos, offsets_map[stage_name].astype(np.int32).tolist())
            stage_times_sec[stage_name] = time.perf_counter() - st0
            save_stage_per_tag(
                stage_dirs[stage_name], tag, axis, stage_name, prev_gan, cond, mask, ref_slab, prefer_tvtk,
                extra_meta={
                    "pos": int(pos),
                    "pair_group": pair_group,
                    "used_in_fixed_final": bool(is_fixed),
                    "center_source": center_used,
                    "runtime_sec": float(stage_times_sec[stage_name]),
                    "runtime_min": float(stage_times_sec[stage_name] / 60.0),
                }
            )

        final_slab_gan = prev_gan
        final_slab_ref = cut_reference_slab(ref_vol, axis, pos, offsets_map[FINAL_STAGE_NAME].astype(np.int32).tolist())

        vol_gan_single = backfill_single_section_into_volume(single_base_gan, final_slab_gan, axis, pos, unknown_val=0)
        vol_ref_single = backfill_single_section_into_volume(single_base_ref, final_slab_ref, axis, pos, unknown_val=0)

        if is_fixed:
            final_gan_fixed = backfill_single_section_into_volume(final_gan_fixed, final_slab_gan, axis, pos, unknown_val=0)
            final_ref_fixed = backfill_single_section_into_volume(final_ref_fixed, final_slab_ref, axis, pos, unknown_val=0)

        total_sec = time.perf_counter() - section_t0
        runtime_row: Dict[str, Any] = {
            "tag": tag,
            "axis": int(axis),
            "pos": int(pos),
            "pair_index": int(pair_index),
            "pair_group": pair_group,
            "used_in_fixed_final": int(is_fixed),
            "center_source": center_used,
            "final_thickness": int(final_slab_gan.shape[0]),
            "cascade_runtime_sec": float(total_sec),
            "cascade_runtime_min": float(total_sec / 60.0),
        }
        for spec_i in STAGE_SPECS:
            name = spec_i["name"]
            sec = float(stage_times_sec.get(name, 0.0))
            runtime_row[f"{name}_sec"] = sec
            runtime_row[f"{name}_min"] = sec / 60.0
        runtime_rows_all.append(runtime_row)
        if is_fixed:
            runtime_rows_fixed.append(dict(runtime_row))

        save_untili_tag_outputs(
            untili_dir=untili_dir,
            tag=tag,
            axis=axis,
            pos=pos,
            vol_gan_zyx=vol_gan_single,
            vol_ref_zyx=vol_ref_single,
            prefer_tvtk=prefer_tvtk,
            runtime_info={
                "pair_index": int(pair_index),
                "pair_group": pair_group,
                "used_in_fixed_final": bool(is_fixed),
                "center_source": center_used,
                "pair_source": pair_source_note,
                "final_thickness": int(final_slab_gan.shape[0]),
                "single_base_note": "This single output volume contains only the current center section as hard data before backfilling thickness=25.",
                "cascade_runtime_sec": float(total_sec),
                "cascade_runtime_min": float(total_sec / 60.0),
                "stage_runtime_sec": {k: float(v) for k, v in stage_times_sec.items()},
                "stage_runtime_min": {k: float(v / 60.0) for k, v in stage_times_sec.items()},
            },
        )

        print(f"[OK] {tag} done. group={pair_group}, used_in_fixed_final={is_fixed}, final_thickness={FINAL_THICKNESS}, total={total_sec:.3f} s ({total_sec/60.0:.3f} min)")

    print(f"[TASK] all_test_pairs = {len(all_pairs)} | fixed_test_pairs = {len(fixed_pairs)} | extra_test_pairs = {len(extra_pairs)}")
    for i, (axis, pos) in enumerate(all_pairs):
        process_one(i, int(axis), int(pos))

    save_runtime_summary(untili_dir, runtime_rows_all, stem="runtime_summary_all_test_pairs")

    total_runtime_sec_all = float(sum(row["cascade_runtime_sec"] for row in runtime_rows_all))
    total_runtime_sec_fixed = float(sum(row["cascade_runtime_sec"] for row in runtime_rows_fixed))

    if fixed_pairs:
        save_runtime_summary(untili_dir / "final", runtime_rows_fixed, stem="runtime_summary_fixed_pairs")

        unk_left = int((final_gan_fixed == 0).sum())
        total_voxels = int(final_gan_fixed.size)
        final_meta_extra = {
            "pair_source": pair_source_note,
            "all_test_pairs": [list(p) for p in all_pairs],
            "fixed_test_pairs_used_for_final": [list(p) for p in fixed_pairs],
            "extra_test_pairs_only_single_outputs": [list(p) for p in extra_pairs],
            "all_test_pair_count": int(len(all_pairs)),
            "fixed_test_pair_count": int(len(fixed_pairs)),
            "extra_test_pair_count": int(len(extra_pairs)),
            "center_source_mode": center_source,
            "total_sections_runtime_sec_all_test_pairs": total_runtime_sec_all,
            "total_sections_runtime_min_all_test_pairs": total_runtime_sec_all / 60.0,
            "total_sections_runtime_sec_fixed_pairs": total_runtime_sec_fixed,
            "total_sections_runtime_min_fixed_pairs": total_runtime_sec_fixed / 60.0,
            "unknown_left": unk_left,
            "total_voxels": total_voxels,
            "unknown_ratio": float(unk_left / total_voxels),
            "final_note": "The final cumulative model only includes fixed_test_pairs. extra_test_pairs are processed and saved only as single-section outputs.",
        }
        if split_info:
            final_meta_extra["split_json"] = str(Path(split_json).resolve())
            if "fixed_test_pairs" in split_info:
                final_meta_extra["fixed_test_pairs_count_in_json"] = int(len(split_info["fixed_test_pairs"]))
            if "extra_test_pairs" in split_info:
                final_meta_extra["extra_test_pairs_count_in_json"] = int(len(split_info["extra_test_pairs"]))

        save_untili_final(untili_dir, final_gan_fixed, final_ref_fixed, prefer_tvtk, final_meta_extra)

        print("-" * 90)
        print(f"[FINAL] untili_23to25/final/ saved (FIXED pairs only). unknown left = {unk_left}/{total_voxels} ({unk_left/total_voxels:.6f})")
        print(f"[FINAL] runtime summary all pairs -> {untili_dir / 'runtime_summary_all_test_pairs.json'}")
        print(f"[FINAL] runtime summary fixed pairs -> {untili_dir / 'final' / 'runtime_summary_fixed_pairs.json'}")
    else:
        print("-" * 90)
        print("[FINAL] no fixed_test_pairs available, so cumulative final volume is skipped. Single-section outputs have still been saved for all test pairs.")
        print(f"[FINAL] runtime summary all pairs -> {untili_dir / 'runtime_summary_all_test_pairs.json'}")

    print(f"[DONE] out_root = {out_root}")


# ============================================================
# CLI
# ============================================================
def _expand_misquoted_argv(raw_argv: List[str]) -> List[str]:
    """
    兼容一种很常见的 IDE/命令行传参错误：
    本来应传多个参数，但实际被作为“一个大字符串”传进来。
    """
    expanded: List[str] = []
    for tok in raw_argv:
        if not isinstance(tok, str):
            expanded.append(tok)
            continue

        tok_norm = tok.replace("\\r", " ").replace("\\n", " ").strip()

        if (" --" in tok_norm) or (tok_norm.startswith("--") and " " in tok_norm):
            expanded.extend(shlex.split(tok_norm))
        else:
            expanded.append(tok)
    return expanded


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_volume_npy", type=str,
                    default="./Ti/diceng_228_228_228_zyx_change_xiangsu_insert16_UNKNOWN0.npy")
    ap.add_argument("--reference_volume_npy", type=str,
                    default="./dataset/diceng_228_228_228_zyx_change_xiangsu.npy")
    ap.add_argument("--split_json", type=str,
                    default="./Ti/split_pairs_fixed_test_xy_25_50_75_100_125_150_175_200_seed1234.json")

    ap.add_argument("--model_1to3", type=str, default="./checkpoints_1to3_cgan/best.pt")
    ap.add_argument("--model_3to5", type=str, default="./checkpoints_3to5_cgan/best.pt")
    ap.add_argument("--model_5to7", type=str, default="./checkpoints_5to7_cgan/best.pt")
    ap.add_argument("--model_7to9", type=str, default="./checkpoints_7to9_cgan/best.pt")
    ap.add_argument("--model_9to11", type=str, default="./checkpoints_9to11_cgan/best.pt")
    ap.add_argument("--model_11to13", type=str, default="./checkpoints_11to13_cgan/best.pt")
    ap.add_argument("--model_13to15", type=str, default="./checkpoints_13to15_cgan/best.pt")
    ap.add_argument("--model_15to17", type=str, default="./checkpoints_15to17_cgan/best.pt")
    ap.add_argument("--model_17to19", type=str, default="./checkpoints_17to19_cgan/best.pt")
    ap.add_argument("--model_19to21", type=str, default="./checkpoints_19to21_cgan/best.pt")
    ap.add_argument("--model_21to23", type=str, default="./checkpoints_21to23_cgan/best.pt")
    ap.add_argument("--model_23to25", type=str, default="./checkpoints_23to25_cgan/best.pt")

    ap.add_argument("--train_1to3_py", type=str, default="./train_pairs_1to3.py")
    ap.add_argument("--train_3to5_py", type=str, default="./train_pairs_3to5.py")
    ap.add_argument("--train_5to7_py", type=str, default="./train_pairs_5to7.py")
    ap.add_argument("--train_7to9_py", type=str, default="./train_pairs_7to9.py")
    ap.add_argument("--train_9to11_py", type=str, default="./train_pairs_9to11.py")
    ap.add_argument("--train_11to13_py", type=str, default="./train_pairs_11to13.py")
    ap.add_argument("--train_13to15_py", type=str, default="./train_pairs_13to15.py")
    ap.add_argument("--train_15to17_py", type=str, default="./train_pairs_15to17.py")
    ap.add_argument("--train_17to19_py", type=str, default="./train_pairs_17to19.py")
    ap.add_argument("--train_19to21_py", type=str, default="./train_pairs_19to21.py")
    ap.add_argument("--train_21to23_py", type=str, default="./train_pairs_21to23.py")
    ap.add_argument("--train_23to25_py", type=str, default="./train_pairs_23to25.py")

    ap.add_argument("--out_dir", type=str, default="./output")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--K", type=int, default=7)

    ap.add_argument("--base_1to3", type=int, default=None)
    ap.add_argument("--base_3to5", type=int, default=None)
    ap.add_argument("--base_5to7", type=int, default=None)
    ap.add_argument("--base_7to9", type=int, default=None)
    ap.add_argument("--base_9to11", type=int, default=None)
    ap.add_argument("--base_11to13", type=int, default=None)
    ap.add_argument("--base_13to15", type=int, default=None)
    ap.add_argument("--base_15to17", type=int, default=None)
    ap.add_argument("--base_17to19", type=int, default=None)
    ap.add_argument("--base_19to21", type=int, default=None)
    ap.add_argument("--base_21to23", type=int, default=None)
    ap.add_argument("--base_23to25", type=int, default=None)

    ap.add_argument(
        "--center_source",
        type=str,
        default="auto",
        choices=["auto", "input", "reference"],
        help=(
            "中心已知剖面的默认来源。注意：在 run_cascade/process_one 中，"
            "fixed_test_pairs 会强制使用 input，extra_test_pairs 会强制使用 reference；"
            "该参数主要作为其他兜底情形的默认来源。"
        ),
    )
    ap.add_argument("--prefer_tvtk", action="store_true")
    ap.add_argument("--ignore_split_json", action="store_true",
                    help="若指定，则忽略 split_json，改用 --xs / --ys。默认不指定，即优先跑 split_json 测试集。")
    ap.add_argument("--xs", type=int, nargs="+", default= [25, 50, 75,100, 125, 150, 175, 200])
    ap.add_argument("--ys", type=int, nargs="+", default= [25, 50, 75,100, 125, 150, 175, 200])

    raw_argv = sys.argv[1:]
    argv = _expand_misquoted_argv(raw_argv)
    if argv != raw_argv:
        print(f"[INFO] detected misquoted CLI arguments, expanded argv -> {argv}")

    args = ap.parse_args(argv)
    print(f"[ARGS] {vars(args)}")

    t0 = time.time()
    run_cascade(**vars(args))
    dt = time.time() - t0
    print(f"[TIME] total script runtime = {dt / 60.0:.2f} min")


if __name__ == "__main__":
    main()