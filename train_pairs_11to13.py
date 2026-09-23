# -*- coding: utf-8 -*-
# @FileName: train_pairs_11to13_diceng_v12_stableGAN.py
# @Time : 2026/1/3 下午8:35
# @Software: PyCharm

"""
【用途】
针对 228×228×228 的 7 类沉积地层数据，训练 11→13 厚度扩展 cGAN：
- GT: ./dataset/diceng_228_228_228_zyx_change_xiangsu.npy
- 属性值: 1,2,3,4,5,6,7；0 只作为条件未知值 unknown_val 使用，GT 中不能包含 0。
- 输入: 中间 11 层已知，前后两侧为待生成未知层；输出: 13 层 slab。
- 本脚本与 1→3、3→5、5→7、7→9、9→11 的 v12_stableGAN 版本保持同一套超参数、split_json 和 checkpoint 命名规则，
  便于后续 1→3、3→5、...、23→25 级联调用。

【核心训练策略】
【v12_stableGAN 根据 1→3 日志后的统一迁移】
- 1→3 日志显示几何误差已经很低，后期增强 GAN 反而容易使 dReal 偏负；
- 因此本阶段沿用 v12 的弱 GAN、延后启动、低频 D 更新策略：GAN 只做轻量真实性校形，CE+gDice 仍主导层位学习；
- best.pt 继续按最小 val_uCE 保存；best_gan.pt 增加 dReal 下限，避免后期过负判别器状态覆盖稳定模型。

1) GAN 是前提：保留 hinge adversarial loss + Feature Matching，但将对抗项延后到第 16 轮，并采用长 ramp 和更低 lrD，
   避免 7 类层状地层的层界被判别器过早拉偏。
2) 监督项仍以 unknown 区 CE + unknown-only generalized Dice 为主；known 区 CE 强约束中间已知层，保证条件剖面不被破坏。
3) 保留轻量 MC Dropout 和 bottleneck latent noise：训练期不过度扰动，后续级联生成 20 组 realization 时可在 generate 脚本中
   开启 mc_dropout_inference 与 sample 采样来保留不确定性。
4) 多尺度改为 228 数据适配版：主尺度 128×128，辅助尺度 192×192，full 尺度默认不作为主尺度，减少显存压力和级联接缝漂移。
5) 固定测试剖面采用 x/y=[25,50,75,100,125,150,175,200]，并通过 SAFE_PAD=12 保证后续直到 23→25 都可复用同一个 split_json。

【推荐参数】
K=7, epochs=40, lr_g=1.60e-4, lr_d=7.50e-6,
lambda_known=5.8, lambda_gdice=0.78, lambda_fm=0.026, lambda_adv=0.00125,
adv_start_epoch=16, fm_start_epoch=24, adv_ramp_epochs=44, fm_ramp_epochs=40,
d_update_every=40, mc_dropout_p=0.015, latent_noise_std=0.0020,
gan_uce_tolerance=0.0010, gan_dreal_min=-0.40。

【运行】
python train_pairs_11to13_diceng_v12_stableGAN.py

【输出】
- base npz: ./Ti/train_pairs_11to13_diceng_228_base.npz
- split json: ./Ti/split_pairs_fixed_test_xy_25_50_75_100_125_150_175_200_seed1234.json
- checkpoint: ./checkpoints_11to13_diceng_cgan_v12_stableGAN/best.pt 与 ./checkpoints_11to13_diceng_cgan_v12_stableGAN/best_gan.pt
"""

import os
import time
import json
import numpy as np
from typing import List, Tuple, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- optional tvtk writer (preferred) ----
try:
    from tvtk.api import tvtk, write_data
    _HAS_TVTK = True
except Exception:
    tvtk, write_data = None, None
    _HAS_TVTK = False


# ============================================================
# 1) 从 GT 构造 11→13 slab 对（X + Y）并保存 npz
# ============================================================
def build_pairs_11to13_base_npz(gt_path: str, out_npz_path: str, unknown_val: int = 0):
    """
    GT: (Z,Y,X)
    X-slab target: gt[:,:,p-4:p+5] -> (9,Z,Y)
    Y-slab target: gt[:,p-4:p+5,:] -> (9,Z,X)

    cond_label: 中间 11 层(索引1,2,3,4,5,6,7,8,9,10,11)已知，两侧未知层为 unknown_val
    cond_mask : 中间 11 层为 1，两侧为 0
    """
    gt = np.load(gt_path)
    assert gt.ndim == 3, f"GT 应为 3D (Z,Y,X)，实际 {gt.shape}"
    Z, Y, X = gt.shape
    print(f"[GT] load: {gt_path}  shape(Z,Y,X)={gt.shape}")

    # unknown_val=0 仅用于“条件未知”，GT 必须是 1..K，不允许包含 0
    if np.any(gt == unknown_val):
        raise ValueError(f"GT 中含 unknown_val={unknown_val}，请确认你已做像素映射到 1..K。")

    x_positions = range(6, X - 6)  # 6..X-7
    y_positions = range(6, Y - 6)  # 6..Y-7

    cond_label_list, cond_mask_list, target_list = [], [], []
    meta_axis, meta_pos = [], []

    # X slab: (Z,Y,13)->(13,Z,Y)
    for p in x_positions:
        slab_zyd = gt[:, :, p - 6:p + 7]
        target_dhw = np.transpose(slab_zyd, (2, 0, 1))  # (13,Z,Y)

        cond_label = np.full_like(target_dhw, unknown_val)
        cond_label[1:12] = np.transpose(gt[:, :, p - 5:p + 6], (2, 0, 1))  # middle 11 known slices

        cond_mask = np.zeros_like(target_dhw, dtype=np.uint8)
        cond_mask[1:12] = 1

        cond_label_list.append(cond_label.astype(np.int16))
        cond_mask_list.append(cond_mask)
        target_list.append(target_dhw.astype(np.int16))
        meta_axis.append(0)  # 0 means X-slab
        meta_pos.append(p)

    # Y slab: (Z,13,X)->(13,Z,X)
    for p in y_positions:
        slab_zdx = gt[:, p - 6:p + 7, :]
        target_dhw = np.transpose(slab_zdx, (1, 0, 2))  # (13,Z,X)

        cond_label = np.full_like(target_dhw, unknown_val)
        cond_label[1:12] = np.transpose(gt[:, p - 5:p + 6, :], (1, 0, 2))  # middle 11 known slices

        cond_mask = np.zeros_like(target_dhw, dtype=np.uint8)
        cond_mask[1:12] = 1

        cond_label_list.append(cond_label.astype(np.int16))
        cond_mask_list.append(cond_mask)
        target_list.append(target_dhw.astype(np.int16))
        meta_axis.append(1)  # 1 means Y-slab
        meta_pos.append(p)

    cond_label = np.stack(cond_label_list, axis=0)  # (N,13,H,W)
    cond_mask  = np.stack(cond_mask_list, axis=0)
    target     = np.stack(target_list, axis=0)
    meta_axis  = np.asarray(meta_axis, dtype=np.int8)
    meta_pos   = np.asarray(meta_pos, dtype=np.int16)

    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        cond_label=cond_label,
        cond_mask=cond_mask,
        target=target,
        meta_axis=meta_axis,
        meta_pos=meta_pos,
    )

    print("[OK] base npz saved:", out_npz_path,
          "N=", cond_label.shape[0],
          f"cond_label shape={cond_label.shape} (N,13,H,W)")


# ============================================================
# 2) 固定测试集索引：根据 (axis,pos) 映射到 global_idx
# ============================================================
def pick_fixed_indices(meta_axis: np.ndarray, meta_pos: np.ndarray,
                       fixed_pairs: List[Tuple[int, int]]) -> List[int]:
    key2idx = {}
    for i, (a, p) in enumerate(zip(meta_axis.tolist(), meta_pos.tolist())):
        key2idx[(int(a), int(p))] = i

    fixed, missing = [], []
    for kp in fixed_pairs:
        if kp in key2idx:
            fixed.append(key2idx[kp])
        else:
            missing.append(kp)
    if missing:
        raise RuntimeError(f"这些 fixed_pairs 在 base_npz 里找不到：{missing}。"
                           f"请确认 pos 是否在 1..(dim-2) 范围内。")
    return sorted(list(set(fixed)))


# ============================================================
# 2.1) 固定测试剖面 + 可复用 split_json（✅跨 stage 保持一致）
#     目标：固定 test 总数、固定 16 条指定剖面，并且保证对未来最大 PAD 也合法
# ============================================================
from typing import Dict, Any

def make_fixed_test_pairs(xs: List[int], ys: List[int]) -> List[Tuple[int, int]]:
    """固定测试剖面：(axis,pos)。axis=0 表示 X-slab，axis=1 表示 Y-slab。"""
    pairs = [(0, int(p)) for p in xs] + [(1, int(p)) for p in ys]
    return sorted(list(set(pairs)))

def save_split_pairs_json(split_info: Dict[str, Any], json_path: str):
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"[SPLIT JSON] saved -> {json_path}")

def load_split_pairs_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_dim_from_meta(meta_axis: np.ndarray, meta_pos: np.ndarray, axis_val: int, pad_of_this_stage: int) -> int:
    """由 meta_pos 反推该轴的 dim：positions = [PAD .. dim-PAD-1] => dim = max_pos + PAD + 1"""
    m = meta_pos[meta_axis == axis_val]
    if m.size == 0:
        raise RuntimeError(f"meta_axis 中找不到 axis={axis_val} 的样本，无法推断 dim")
    return int(m.max()) + int(pad_of_this_stage) + 1

def build_allowed_pairs_allstages(
    meta_axis: np.ndarray,
    meta_pos: np.ndarray,
    pad_of_this_stage: int,
    safe_pad_for_all_stages: int,
) -> set:
    """返回“对所有 stage 都安全”的 (axis,pos) 集合：pos ∈ [SAFE_PAD, dim - SAFE_PAD - 1]"""
    SAFE = int(max(safe_pad_for_all_stages, 0))
    dim_x = infer_dim_from_meta(meta_axis, meta_pos, axis_val=0, pad_of_this_stage=pad_of_this_stage)
    dim_y = infer_dim_from_meta(meta_axis, meta_pos, axis_val=1, pad_of_this_stage=pad_of_this_stage)

    allowed = set()
    for a, p in zip(meta_axis.tolist(), meta_pos.tolist()):
        a = int(a); p = int(p)
        if a == 0:
            if SAFE <= p <= (dim_x - SAFE - 1):
                allowed.add((a, p))
        else:
            if SAFE <= p <= (dim_y - SAFE - 1):
                allowed.add((a, p))
    return allowed

def build_consistent_split_pairs(
    meta_axis: np.ndarray,
    meta_pos: np.ndarray,
    fixed_test_pairs: List[Tuple[int, int]],
    test_ratio: float,
    seed: int,
    safe_pad_for_all_stages: int,
    pad_of_this_stage: int,
) -> Dict[str, Any]:
    """
    生成并固化 split：
      - 总 test 数 = round(test_ratio * N_base)
      - test 必含 fixed_test_pairs（通常 16 条）
      - extra_test 从 “对所有 stage 都安全”的 allowed 集合中随机抽取（按 seed 固定）
    """
    N_base = int(len(meta_axis))
    fixed_set = set([tuple(x) for x in fixed_test_pairs])

    allowed_set = build_allowed_pairs_allstages(
        meta_axis=meta_axis, meta_pos=meta_pos,
        pad_of_this_stage=pad_of_this_stage,
        safe_pad_for_all_stages=safe_pad_for_all_stages
    )

    missing_fixed = sorted(list(fixed_set - allowed_set))
    if missing_fixed:
        raise RuntimeError(
            f"固定测试剖面不满足 SAFE_PAD={safe_pad_for_all_stages}：{missing_fixed}\n"
            f"请降低 SAFE_PAD 或调整 fixed xs/ys。"
        )

    n_test_total = int(np.round(test_ratio * N_base))
    if n_test_total < len(fixed_set):
        n_test_total = len(fixed_set)
    n_extra = n_test_total - len(fixed_set)

    all_pairs_set = set([(int(a), int(p)) for a, p in zip(meta_axis.tolist(), meta_pos.tolist())])
    # extra 候选：allowed - fixed
    cand = sorted(list((allowed_set & all_pairs_set) - fixed_set))
    if n_extra > len(cand):
        raise RuntimeError(
            f"SAFE_PAD={safe_pad_for_all_stages} 下候选不足：need extra={n_extra}, cand={len(cand)}"
        )

    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(cand), size=n_extra, replace=False) if n_extra > 0 else []
    extra_test_pairs = [cand[i] for i in sorted(pick_idx.tolist())] if n_extra > 0 else []

    test_pairs_all = sorted(list(fixed_set.union(extra_test_pairs)))
    train_pairs_all = sorted(list(all_pairs_set - set(test_pairs_all)))

    return {
        "seed": int(seed),
        "N_base": int(N_base),
        "test_ratio": float(test_ratio),
        "n_test_total": int(n_test_total),
        "n_fixed_test": int(len(fixed_set)),
        "n_extra_test": int(n_extra),
        "fixed_test_pairs": sorted(list(fixed_set)),
        "extra_test_pairs": extra_test_pairs,
        "test_pairs_all": test_pairs_all,
        "train_pairs_all": train_pairs_all,
        "SAFE_PAD_FOR_ALL_STAGES": int(safe_pad_for_all_stages),
    }

def sanitize_split_info_inplace(
    split_info: Dict[str, Any],
    meta_axis: np.ndarray,
    meta_pos: np.ndarray,
    fixed_test_pairs: List[Tuple[int, int]],
    test_ratio: float,
    seed: int,
    pad_of_this_stage: int,
    safe_pad_for_all_stages: int,
) -> Dict[str, Any]:
    """若 split_json 中存在对未来 stage 不安全的 pair，则删除并按 seed 补齐，保证 test 总数不变。"""
    fixed_set = set([tuple(x) for x in fixed_test_pairs])

    allowed_set = build_allowed_pairs_allstages(
        meta_axis=meta_axis, meta_pos=meta_pos,
        pad_of_this_stage=pad_of_this_stage,
        safe_pad_for_all_stages=safe_pad_for_all_stages
    )

    missing_fixed = sorted(list(fixed_set - allowed_set))
    if missing_fixed:
        raise RuntimeError(
            f"[SPLIT SANITIZE] fixed_test_pairs 不满足 SAFE_PAD={safe_pad_for_all_stages}：{missing_fixed}"
        )

    N_base = int(len(meta_axis))
    n_test_total = int(np.round(test_ratio * N_base))
    if n_test_total < len(fixed_set):
        n_test_total = len(fixed_set)

    old_test = [tuple(x) for x in split_info.get("test_pairs_all", [])]
    kept = sorted(list((set(old_test) & allowed_set) | fixed_set))

    need = n_test_total - len(kept)
    if need < 0:
        # 截断到 n_test_total，但保证 fixed 在内
        kept = sorted(list(fixed_set)) + [p for p in kept if p not in fixed_set]
        kept = kept[:n_test_total]
        need = 0

    if need > 0:
        cand = sorted(list(allowed_set - fixed_set - set(kept)))
        if need > len(cand):
            raise RuntimeError(
                f"[SPLIT SANITIZE] SAFE_PAD={safe_pad_for_all_stages} 下候选不足：need={need}, cand={len(cand)}"
            )
        rng = np.random.default_rng(seed)
        pick_idx = rng.choice(len(cand), size=need, replace=False)
        kept = sorted(list(set(kept).union([cand[i] for i in sorted(pick_idx.tolist())])))

    all_pairs_set = set([(int(a), int(p)) for a, p in zip(meta_axis.tolist(), meta_pos.tolist())])
    train_pairs_all = sorted(list(all_pairs_set - set(kept)))
    extra_test_pairs = sorted(list(set(kept) - fixed_set))

    split_info.update({
        "seed": int(seed),
        "N_base": int(N_base),
        "test_ratio": float(test_ratio),
        "n_test_total": int(n_test_total),
        "n_fixed_test": int(len(fixed_set)),
        "n_extra_test": int(n_test_total - len(fixed_set)),
        "fixed_test_pairs": sorted(list(fixed_set)),
        "extra_test_pairs": extra_test_pairs,
        "test_pairs_all": kept,
        "train_pairs_all": train_pairs_all,
        "SAFE_PAD_FOR_ALL_STAGES": int(safe_pad_for_all_stages),
    })
    return split_info


# ============================================================
# 3) 矩形缩放/裁剪（✅修复：支持共享 crop 坐标）
# ============================================================
SizeHW = Union[int, Tuple[int, int]]

def _to_hw(size: SizeHW) -> Tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    return int(size[0]), int(size[1])

def resize_dhw_nearest(arr_dhw: np.ndarray, out_hw: SizeHW) -> np.ndarray:
    out_h, out_w = _to_hw(out_hw)
    if arr_dhw.shape[1] == out_h and arr_dhw.shape[2] == out_w:
        return arr_dhw
    t = torch.from_numpy(arr_dhw).unsqueeze(0).float()  # (1,D,H,W)
    t = F.interpolate(t, size=(out_h, out_w), mode="nearest")
    return t.squeeze(0).cpu().numpy().astype(arr_dhw.dtype)

def rand_crop_coords(H: int, W: int, crop_h: int, crop_w: int, rng) -> Tuple[int, int, int, int]:
    """返回 (top, left, crop_h, crop_w)，确保不越界。"""
    crop_h = min(int(crop_h), int(H))
    crop_w = min(int(crop_w), int(W))
    if crop_h == H and crop_w == W:
        return 0, 0, crop_h, crop_w
    top = int(rng.randint(0, H - crop_h + 1))
    left = int(rng.randint(0, W - crop_w + 1))
    return top, left, crop_h, crop_w

def apply_crop_dhw(arr_dhw: np.ndarray, top: int, left: int, crop_h: int, crop_w: int) -> np.ndarray:
    """✅对齐裁剪：所有张量共享同一 (top,left,crop_h,crop_w)。"""
    return arr_dhw[:, top:top + crop_h, left:left + crop_w]


def transition_density_dhw(arr_dhw: np.ndarray) -> float:
    """计算类别边界密度，值越大表示该窗口结构变化越丰富。"""
    if arr_dhw.size == 0:
        return 0.0
    diff_h = float(np.mean(arr_dhw[:, 1:, :] != arr_dhw[:, :-1, :])) if arr_dhw.shape[1] > 1 else 0.0
    diff_w = float(np.mean(arr_dhw[:, :, 1:] != arr_dhw[:, :, :-1])) if arr_dhw.shape[2] > 1 else 0.0
    return 0.5 * (diff_h + diff_w)


def categorical_entropy_dhw(arr_dhw: np.ndarray) -> float:
    """类别分布熵：避免裁剪窗口长期只落到几乎纯单一岩相的简单区域。"""
    if arr_dhw.size == 0:
        return 0.0
    _, counts = np.unique(arr_dhw, return_counts=True)
    p = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent / np.log(max(len(counts), 2)))


def center_difference_score_dhw(arr_dhw: np.ndarray) -> float:
    """
    计算外侧层与中心已知层的差异度。
    对 11→13 来说，如果两侧 outer 仍接近中间 7 层的平滑模板，这个分数会偏低；
    优先抽取该分数更高的窗口，可以让生成器更多看到真实的局部相位差与厚度变化。
    """
    if arr_dhw.size == 0 or arr_dhw.shape[0] < 3:
        return 0.0
    mid = arr_dhw.shape[0] // 2
    center = arr_dhw[mid]
    diffs = []
    for d in range(arr_dhw.shape[0]):
        if d == mid:
            continue
        diffs.append(float(np.mean(arr_dhw[d] != center)))
    return float(np.mean(diffs)) if diffs else 0.0


def pick_structured_crop_coords(
    target_dhw: np.ndarray,
    H: int,
    W: int,
    crop_h: int,
    crop_w: int,
    rng,
    num_trials: int = 12,
    outer_weight: float = 1.0,
    entropy_weight: float = 0.40,
    center_diff_weight: float = 0.90,
) -> Tuple[int, int, int, int]:
    """
    在多个随机候选中，优先选择：
    1) 类别边界更丰富；
    2) 外侧未知层变化更明显；
    3) 外侧层与中心层存在更真实的局部差异；
    4) 类别分布不过于单一。
    这样可以减少模型被大量 easy window 牵着走，缓解生成结果被平均模板化。
    """
    best = None
    best_score = -1e18
    n_try = max(int(num_trials), 1)
    for _ in range(n_try):
        top, left, ch, cw = rand_crop_coords(H, W, crop_h, crop_w, rng)
        crop = apply_crop_dhw(target_dhw, top, left, ch, cw)
        score_all = transition_density_dhw(crop)
        outer = crop[[0, -1]] if crop.shape[0] >= 2 else crop
        score_outer = transition_density_dhw(outer)
        score_ent = categorical_entropy_dhw(crop)
        score_mid = center_difference_score_dhw(crop)
        score = score_all + outer_weight * score_outer + entropy_weight * score_ent + center_diff_weight * score_mid
        if score > best_score:
            best_score = score
            best = (top, left, ch, cw)
    return best


# ============================================================
# 4) Dataset：矩形多尺度 + 2.5D 输入（✅裁剪对齐）
# ============================================================
class RandomEpochDatasetMultiScale(Dataset):
    """
    输入通道数：
      (K + mask + offset + axis) * D
    其中 D=13（11→13），K=7（7类）
    => (7 + 3) * 13 = 130 通道
    """
    def __init__(
        self,
        npz_path: str,
        idx_pool: np.ndarray,
        epoch_size: int,
        scale: SizeHW,
        K: int = 7,
        allow_flip: bool = True,
        use_offset: bool = True,
        use_axis: bool = True,
        crop_choices_high: Sequence[SizeHW] = ((120, 150), (96, 128), (80, 100)),
        seed: Optional[int] = None,
        hard_crop_num_trials: int = 18,
        hard_crop_outer_weight: float = 1.40,
        hard_crop_entropy_weight: float = 0.50,
        hard_crop_center_diff_weight: float = 1.25,
    ):
        data = np.load(npz_path, allow_pickle=True)
        self.cond_label = data["cond_label"]  # (N,13,H,W)
        self.cond_mask  = data["cond_mask"]
        self.target     = data["target"]      # 1..K
        self.meta_axis  = data["meta_axis"]   # 0/1

        self.idx_pool = np.asarray(idx_pool, dtype=np.int64)
        self.epoch_size = int(epoch_size)
        self.scale = scale
        self.K = int(K)

        self.allow_flip = bool(allow_flip)
        self.use_offset = bool(use_offset)
        self.use_axis = bool(use_axis)
        self.crop_choices_high = list(crop_choices_high)
        self.hard_crop_num_trials = int(hard_crop_num_trials)
        self.hard_crop_outer_weight = float(hard_crop_outer_weight)
        self.hard_crop_entropy_weight = float(hard_crop_entropy_weight)
        self.hard_crop_center_diff_weight = float(hard_crop_center_diff_weight)

        _, _, self.full_h, self.full_w = self.cond_label.shape
        out_h, out_w = _to_hw(self.scale)
        if out_h > self.full_h or out_w > self.full_w:
            raise ValueError(f"scale={self.scale} 不能大于 full(H,W)=({self.full_h},{self.full_w})")

        self.offset = torch.tensor([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)[:, None, None]  # (13,1,1)
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        r = self._rng
        base_idx = int(self.idx_pool[r.randint(0, len(self.idx_pool))])

        cl = self.cond_label[base_idx]  # (13,H,W)
        cm = self.cond_mask[base_idx]
        yt = self.target[base_idx]
        axis_flag = int(self.meta_axis[base_idx])

        # ✅关键修复：同一 crop 坐标用于 cl/cm/yt，且不再只做纯随机抽窗
        #    新增结构感知裁剪：优先选择外侧未知层边界更丰富、类别更平衡的窗口
        choice = self.crop_choices_high[r.randint(0, len(self.crop_choices_high))]
        crop_h, crop_w = _to_hw(choice)
        top, left, crop_h, crop_w = pick_structured_crop_coords(
            target_dhw=yt,
            H=self.full_h, W=self.full_w,
            crop_h=crop_h, crop_w=crop_w,
            rng=r,
            num_trials=self.hard_crop_num_trials,
            outer_weight=self.hard_crop_outer_weight,
            entropy_weight=self.hard_crop_entropy_weight,
            center_diff_weight=self.hard_crop_center_diff_weight,
        )
        cl = apply_crop_dhw(cl, top, left, crop_h, crop_w)
        cm = apply_crop_dhw(cm, top, left, crop_h, crop_w)
        yt = apply_crop_dhw(yt, top, left, crop_h, crop_w)

        # 缩放到目标尺度
        cl = resize_dhw_nearest(cl, self.scale)
        cm = resize_dhw_nearest(cm, self.scale)
        yt = resize_dhw_nearest(yt, self.scale)

        # 仅做横向 flip（W 方向），不做 Z/H 方向翻转。
        # 因为这里 H 实际对应 Z（深度）方向，翻转会破坏真实层序与倾斜关系。
        if self.allow_flip and (r.rand() < 0.5):
            cl = cl[:, :, ::-1].copy()
            cm = cm[:, :, ::-1].copy()
            yt = yt[:, :, ::-1].copy()

        # target：1..K -> 0..K-1
        y = yt.astype(np.int64) - 1
        if y.min() < 0 or y.max() >= self.K:
            raise ValueError("target 超范围：确认 GT 是否为 1..K 且 K 设置正确。")

        D, H, W = y.shape  # D=13

        # one-hot（仅已知位置写入）
        onehot = np.zeros((self.K, D, H, W), dtype=np.float32)
        known = (cm == 1)
        if known.any():
            lab = cl[known].astype(np.int64) - 1
            lab = np.clip(lab, 0, self.K - 1)
            d_idx, h_idx, w_idx = np.where(known)
            onehot[lab, d_idx, h_idx, w_idx] = 1.0

        x_list = [torch.from_numpy(onehot)]         # (K,D,H,W)
        mask_t = torch.from_numpy(cm.astype(np.float32))
        x_list.append(mask_t[None, ...])            # (1,D,H,W)

        if self.use_offset:
            off = self.offset.expand(D, H, W)       # (D,H,W)
            x_list.append(off[None, ...])           # (1,D,H,W)

        if self.use_axis:
            ax = torch.full((1, D, H, W), float(axis_flag), dtype=torch.float32)
            x_list.append(ax)

        x_4d = torch.cat(x_list, dim=0).contiguous()  # (K+3, D, H, W)

        # 2.5D：把 D 展到通道
        x_2d = x_4d.permute(1, 0, 2, 3).reshape((D * x_4d.shape[0], H, W)).contiguous()
        return x_2d.float(), torch.from_numpy(y).long(), mask_t.float()


class ValDatasetFull(Dataset):
    def __init__(self, npz_path: str, val_idx: np.ndarray, K: int = 7, use_offset=True, use_axis=True):
        data = np.load(npz_path, allow_pickle=True)
        self.cond_label = data["cond_label"]
        self.cond_mask  = data["cond_mask"]
        self.target     = data["target"]
        self.meta_axis  = data["meta_axis"]
        self.val_idx = np.asarray(val_idx, dtype=np.int64)
        self.K = int(K)
        self.use_offset = bool(use_offset)
        self.use_axis = bool(use_axis)
        self.offset = torch.tensor([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.val_idx)

    def __getitem__(self, i):
        base_idx = int(self.val_idx[i])
        cl = self.cond_label[base_idx]
        cm = self.cond_mask[base_idx]
        yt = self.target[base_idx]
        axis_flag = int(self.meta_axis[base_idx])

        y = yt.astype(np.int64) - 1
        D, H, W = y.shape

        onehot = np.zeros((self.K, D, H, W), dtype=np.float32)
        known = (cm == 1)
        if known.any():
            lab = cl[known].astype(np.int64) - 1
            lab = np.clip(lab, 0, self.K - 1)
            d_idx, h_idx, w_idx = np.where(known)
            onehot[lab, d_idx, h_idx, w_idx] = 1.0

        x_list = [torch.from_numpy(onehot)]
        mask_t = torch.from_numpy(cm.astype(np.float32))
        x_list.append(mask_t[None, ...])

        if self.use_offset:
            off = self.offset.expand(D, H, W)
            x_list.append(off[None, ...])

        if self.use_axis:
            ax = torch.full((1, D, H, W), float(axis_flag), dtype=torch.float32)
            x_list.append(ax)

        x_4d = torch.cat(x_list, dim=0).contiguous()
        x_2d = x_4d.permute(1, 0, 2, 3).reshape((D * x_4d.shape[0], H, W)).contiguous()
        return x_2d.float(), torch.from_numpy(y).long(), mask_t.float()


# ============================================================
# 5) 模型：UNet2D（G）/ PatchGAN（D）
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, gn_groups=8, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        self.dropout_p = float(dropout_p)

    def _maybe_dropout(self, x, force_mc_dropout: bool = False):
        if self.dropout_p <= 0.0:
            return x
        return F.dropout2d(x, p=self.dropout_p, training=(self.training or force_mc_dropout))

    def forward(self, x, force_mc_dropout: bool = False):
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self._maybe_dropout(x, force_mc_dropout=force_mc_dropout)
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = self._maybe_dropout(x, force_mc_dropout=force_mc_dropout)
        return x


class UNet2D(nn.Module):
    """
    这版 UNet2D 做了两类“尽量少改接口”的增强：

    1) 不改 11→13 的输入/输出维度，不破坏后续 generate 脚本的加载方式；

    2) 在生成器内部埋入随机性：
       - 在 bottleneck 注入小幅高斯噪声；
       - 在 bottleneck / decoder 使用 MC Dropout；
       - 训练时默认开启；
       - 推理时只有当 mc_dropout_inference=True 才会继续开启随机性。
         因此：
         * 验证(best.pt 选择)仍可保持稳定；
         * 后续 generate 脚本若希望同一条件输入生成多组 realization，
           只需要在加载模型后额外设置：
               G.mc_dropout_inference = True
         即可让同一输入产生不同输出。
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        base=48,
        mc_dropout_p: float = 0.015,
        latent_noise_std: float = 0.0020,
        mc_dropout_inference: bool = False,
    ):
        super().__init__()
        self.mc_dropout_p = float(mc_dropout_p)
        self.latent_noise_std = float(latent_noise_std)
        self.mc_dropout_inference = bool(mc_dropout_inference)

        self.enc1 = ConvBlock(in_ch, base, dropout_p=0.0)
        self.enc2 = ConvBlock(base, base * 2, dropout_p=0.0)
        self.enc3 = ConvBlock(base * 2, base * 4, dropout_p=0.0)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8, dropout_p=self.mc_dropout_p)

        self.up3 = nn.Conv2d(base * 8, base * 4, 1)
        self.dec3 = ConvBlock(base * 8, base * 4, dropout_p=self.mc_dropout_p)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout_p=self.mc_dropout_p)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlock(base * 2, base, dropout_p=self.mc_dropout_p * 0.5)
        self.out = nn.Conv2d(base, out_ch, 1)

    def _noise_is_on(self) -> bool:
        return self.training or self.mc_dropout_inference

    def _maybe_add_latent_noise(self, x):
        if self.latent_noise_std <= 0.0:
            return x
        if not self._noise_is_on():
            return x
        return x + torch.randn_like(x) * self.latent_noise_std

    def forward(self, x):
        force_mc = self.mc_dropout_inference

        e1 = self.enc1(x, force_mc_dropout=force_mc)
        e2 = self.enc2(self.pool(e1), force_mc_dropout=force_mc)
        e3 = self.enc3(self.pool(e2), force_mc_dropout=force_mc)

        b = self.bottleneck(self.pool(e3), force_mc_dropout=force_mc)
        b = self._maybe_add_latent_noise(b)

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), force_mc_dropout=force_mc)

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), force_mc_dropout=force_mc)

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), force_mc_dropout=force_mc)
        return self.out(d1)  # (N, K*D, H, W)


def enable_generator_mc_dropout(net: nn.Module) -> None:
    """
    供后续生成脚本调用：
        G.eval()
        enable_generator_mc_dropout(G)
    这样可以在 eval 模式下仍保持 MC Dropout / latent noise，
    让同一条件输入生成多组不同 realization。
    """
    if hasattr(net, "mc_dropout_inference"):
        net.mc_dropout_inference = True


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()

        def block(cin, cout, k=4, s=2, p=1, use_gn=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)]
            if use_gn:
                layers.append(nn.GroupNorm(num_groups=min(8, cout), num_channels=cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # 显式拆成多层，便于后续抽取多尺度特征做 Feature Matching
        self.block1 = block(in_ch, base, use_gn=False)
        self.block2 = block(base, base * 2)
        self.block3 = block(base * 2, base * 4)
        self.block4 = block(base * 4, base * 4, s=1)
        self.out_conv = nn.Conv2d(base * 4, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, return_feats: bool = False):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        out = self.out_conv(f4)

        if return_feats:
            return out, [f1, f2, f3, f4]
        return out


# ============================================================
# 6) 损失
# ============================================================
def loss_ce_known_unknown_simple(
    logits_kdhw,
    target_dhw,
    mask_dhw,
    lambda_known: float = 5.8,
):
    """
    最终统一版损失中的 CE 部分：
    1) 已知区（当前 11→13 中为中间 7 层）使用加权 CE，保证条件严格一致；
    2) 未知区（两侧待生成层）使用普通 CE，作为主语义监督。
    """
    known = (mask_dhw > 0.5)
    unk = ~known

    ce_map = F.cross_entropy(
        logits_kdhw.permute(0, 2, 3, 4, 1).reshape(-1, logits_kdhw.size(1)),
        target_dhw.reshape(-1),
        reduction="none"
    ).reshape_as(target_dhw)

    ce_known = ce_map[known].mean() if known.any() else torch.zeros((), device=logits_kdhw.device)
    ce_unk = ce_map[unk].mean() if unk.any() else torch.zeros((), device=logits_kdhw.device)

    total = lambda_known * ce_known + ce_unk

    with torch.no_grad():
        pred = torch.argmax(logits_kdhw, dim=1)
        acc_k = (pred[known] == target_dhw[known]).float().mean() if known.any() else torch.zeros((), device=logits_kdhw.device)
        acc_u = (pred[unk] == target_dhw[unk]).float().mean() if unk.any() else torch.zeros((), device=logits_kdhw.device)

    return total, ce_known, ce_unk, acc_k, acc_u


def loss_gdice_unknown(logits_kdhw, target_dhw, mask_dhw, eps=1e-6):
    """
    unknown only 的 Generalized Dice：
    - 当前先用于二元岩相；
    - 写成 generalized 形式，后续扩展到多类时不需要再改损失定义。
    """
    prob = torch.softmax(logits_kdhw, dim=1)                # (N,K,D,H,W)
    K = logits_kdhw.size(1)
    unk = (mask_dhw < 0.5).float()                          # (N,D,H,W)

    gt_oh = F.one_hot(target_dhw, num_classes=K).permute(0, 4, 1, 2, 3).float()  # (N,K,D,H,W)
    unk = unk.unsqueeze(1)                                  # (N,1,D,H,W)

    p = prob * unk
    g = gt_oh * unk

    # 按 batch 汇总各类体积，构造 generalized dice 的类权重
    g_sum = g.sum(dim=(0, 2, 3, 4))                         # (K,)
    w = 1.0 / (g_sum * g_sum + eps)                         # (K,)

    inter = (p * g).sum(dim=(0, 2, 3, 4))                   # (K,)
    denom = (p + g).sum(dim=(0, 2, 3, 4))                   # (K,)

    gdice = 1.0 - (2.0 * (w * inter).sum() + eps) / ((w * denom).sum() + eps)
    return gdice


def loss_feature_matching(feats_real, feats_fake):
    """
    Feature Matching:
    让生成结果不仅骗过 D，还去匹配 D 中间层的多尺度特征，
    对跨 stage / 跨数据的稳定训练很有帮助。
    """
    loss = 0.0
    for fr, ff in zip(feats_real, feats_fake):
        loss = loss + F.l1_loss(ff, fr.detach())
    return loss


def d_hinge_loss(d_real, d_fake):
    """判别器 hinge 损失。"""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge_adv_loss(d_fake):
    """生成器对抗损失。"""
    return (-d_fake).mean()


# ============================================================
# 7) 训练：3 个尺度
# ============================================================
def train_multiscale_11to13_3scales(
    npz_base_path: str,
    K: int = 7,
    epochs: int = 40,
    lr_g: float = 1.60e-4,
    lr_d: float = 7.50e-6,
    lambda_known: float = 5.8,
    lambda_gdice: float = 0.78,
    lambda_fm: float = 0.026,
    lambda_adv: float = 0.00125,
    use_gan: bool = True,
    aux_gan_enabled: bool = False,
    adv_start_epoch: int = 16,
    fm_start_epoch: int = 24,
    adv_ramp_epochs: int = 44,
    fm_ramp_epochs: int = 40,
    d_update_every: int = 40,
    mc_dropout_p: float = 0.015,
    latent_noise_std: float = 0.0020,
    epoch_size_64: int = 3600,
    epoch_size_96: int = 1800,
    epoch_size_full: int = 80,
    batch_64: int = 4,
    batch_96: int = 2,
    batch_full: int = 1,
    stage64_end: int = 40,
    use_full_after: int = 999,
    full_main_after: int = 999,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 2e-5,
    early_stop_start_epoch: int = 28,
    gan_best_start_epoch: int = 18,
    gan_uce_tolerance: float = 0.0010,
    gan_dreal_min: float = -0.40,
    train_idx: Optional[Sequence[int]] = None,
    val_idx: Optional[Sequence[int]] = None,
    save_dir: str = "./checkpoints_11to13_diceng_cgan_v12_stableGAN",
    seed: int = 1234,
):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data = np.load(npz_base_path, allow_pickle=True)
    N_base = int(data["cond_label"].shape[0])

    if train_idx is None or val_idx is None:
        idx = np.random.permutation(N_base)
        split = int(0.9 * N_base)
        train_idx = idx[:split]
        val_idx = idx[split:]
    else:
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)

    print(f"[SPLIT] train={len(train_idx)}  val={len(val_idx)}  (N_base={N_base})")

    _, _, full_h, full_w = data["cond_label"].shape
    print(f"[DATA] full size (H,W)=({full_h},{full_w})  (expect 228,228)")

    scale64  = (128, 128)
    scale96  = (192, 192)
    scaleful = (full_h, full_w)

    crop_choices = ((228, 228), (224, 224), (216, 216), (208, 208), (192, 192), (176, 176), (160, 160))

    ds64  = RandomEpochDatasetMultiScale(npz_base_path, train_idx, epoch_size_64,  scale=scale64,  K=K,
                                         crop_choices_high=crop_choices, allow_flip=True, seed=seed+1)
    ds96  = RandomEpochDatasetMultiScale(npz_base_path, train_idx, epoch_size_96,  scale=scale96,  K=K,
                                         crop_choices_high=crop_choices, allow_flip=True, seed=seed+2)
    dsful = RandomEpochDatasetMultiScale(npz_base_path, train_idx, epoch_size_full, scale=scaleful, K=K,
                                         crop_choices_high=((full_h, full_w),), allow_flip=True, seed=seed+3)

    loader64  = DataLoader(ds64,  batch_size=batch_64,  shuffle=False, num_workers=0, pin_memory=True)
    loader96  = DataLoader(ds96,  batch_size=batch_96,  shuffle=False, num_workers=0, pin_memory=True)
    loaderful = DataLoader(dsful, batch_size=batch_full, shuffle=False, num_workers=0, pin_memory=True)

    val_ds = ValDatasetFull(npz_base_path, val_idx, K=K, use_offset=True, use_axis=True)
    val_loader = DataLoader(val_ds, batch_size=batch_full, shuffle=False, num_workers=0, pin_memory=True)

    D = 13  # 11→13 阶段的 slab 厚度，必须与 Dataset 实际展开的 D 保持一致
    C_cond = (K + 3) * D
    C_slab = K * D
    C_out  = K * D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(
        "[CONFIG] diceng-v12-stableGAN: "
        f"epochs={epochs}, lrG={lr_g:.2e}, lrD={lr_d:.2e}, "
        f"lambda_known={lambda_known}, lambda_gdice={lambda_gdice}, "
        f"lambda_fm={lambda_fm}, lambda_adv={lambda_adv}, aux_gan_enabled={aux_gan_enabled}, "
        f"adv_start={adv_start_epoch}, fm_start={fm_start_epoch}, "
        f"stage64_end={stage64_end}, use_full_after={use_full_after}, "
        f"mc_dropout_p={mc_dropout_p}, latent_noise_std={latent_noise_std}, "
        f"save_dir={save_dir}"
    )

    G = UNet2D(
        in_ch=C_cond,
        out_ch=C_out,
        base=48,
        mc_dropout_p=mc_dropout_p,
        latent_noise_std=latent_noise_std,
        mc_dropout_inference=False,
    ).to(device)
    Dnet = PatchDiscriminator(in_ch=C_cond + C_slab, base=64).to(device)

    optG = torch.optim.AdamW(G.parameters(), lr=lr_g, weight_decay=1e-4)
    optD = torch.optim.AdamW(Dnet.parameters(), lr=lr_d, weight_decay=1e-4)

    # logtuned：adv 延后到第 32 轮，因此学习率衰减节点同步后移，
    # 避免轻量 GAN 校形阶段刚开始就被过早衰减。
    schedG = torch.optim.lr_scheduler.MultiStepLR(optG, milestones=[26, 36], gamma=0.5)
    schedD = torch.optim.lr_scheduler.MultiStepLR(optD, milestones=[26, 36], gamma=0.5)

    ema_decay = 0.999
    G_ema = UNet2D(
        in_ch=C_cond,
        out_ch=C_out,
        base=48,
        mc_dropout_p=mc_dropout_p,
        latent_noise_std=latent_noise_std,
        mc_dropout_inference=False,
    ).to(device)
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)

    def ema_update(model, model_ema, decay=0.999):
        with torch.no_grad():
            msd = model.state_dict()
            esd = model_ema.state_dict()
            for k in esd.keys():
                esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)

    def forward_to_kdhw(net, x2d):
        logits2d = net(x2d)  # (N,K*D,H,W)
        N_b, _, H, W = logits2d.shape
        return logits2d.view(N_b, K, D, H, W)

    def onehot_slab_from_target(y_dhw, K_):
        N, D_here, H, W = y_dhw.shape
        oh = F.one_hot(y_dhw, num_classes=K_).permute(0, 4, 1, 2, 3).float()
        return oh.reshape(N, K_ * D_here, H, W)

    def slab_prob_from_logits(logits_kdhw):
        prob = torch.softmax(logits_kdhw, dim=1)
        N, K_, D_here, H, W = prob.shape
        return prob.reshape(N, K_ * D_here, H, W)

    def set_requires_grad(net, flag: bool):
        for p in net.parameters():
            p.requires_grad_(flag)

    def linear_ramp(epoch: int, start_epoch: int, ramp_epochs: int, max_value: float) -> float:
        if epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return max_value
        t = min(1.0, max(0.0, (epoch - start_epoch + 1) / float(ramp_epochs)))
        return max_value * t

    def cur_lambda_adv(epoch: int) -> float:
        if not use_gan:
            return 0.0
        return linear_ramp(epoch, adv_start_epoch, adv_ramp_epochs, lambda_adv)

    def cur_lambda_fm(epoch: int) -> float:
        if not use_gan:
            return 0.0
        return linear_ramp(epoch, fm_start_epoch, fm_ramp_epochs, lambda_fm)

    def run_train_one_loader(loader, epoch, gan_enabled: bool = True):
        G.train()
        Dnet.train()

        sum_g = sum_ce_k = sum_ce_u = sum_acc_k = sum_acc_u = 0.0
        sum_gdice = sum_fm = sum_adv = 0.0
        sum_d = 0.0
        n = 0
        n_d = 0
        did_d_any = False

        # v9: 主尺度保留 GAN 对抗训练；辅助尺度默认只做 CE+gDice 监督。
        # 这样仍然是多尺度训练，但避免 96×120 辅助尺度上的 D/FM 把几何位置拉偏。
        lam_adv = cur_lambda_adv(epoch) if gan_enabled else 0.0
        lam_fm = cur_lambda_fm(epoch) if gan_enabled else 0.0

        for step_i, (x2d, y_dhw, m_dhw) in enumerate(loader, start=1):
            x2d = x2d.to(device, non_blocking=True)
            y_dhw = y_dhw.to(device, non_blocking=True)
            m_dhw = m_dhw.to(device, non_blocking=True)

            bs = x2d.size(0)
            logits_kdhw = forward_to_kdhw(G, x2d)

            # --------------------------------------------------
            # 1) 判别器更新
            #    只有开始启用对抗训练后才更新 D
            # --------------------------------------------------
            do_update_d = (lam_adv > 0.0) and (((step_i - 1) % d_update_every) == 0)
            if do_update_d:
                set_requires_grad(Dnet, True)
                set_requires_grad(G, False)

                with torch.no_grad():
                    fake_slab = slab_prob_from_logits(logits_kdhw.detach())
                real_slab = onehot_slab_from_target(y_dhw, K)

                d_in_real = torch.cat([x2d, real_slab], dim=1)
                d_in_fake = torch.cat([x2d, fake_slab], dim=1)

                d_real = Dnet(d_in_real)
                d_fake = Dnet(d_in_fake)
                d_loss = d_hinge_loss(d_real, d_fake)

                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                nn.utils.clip_grad_norm_(Dnet.parameters(), 5.0)
                optD.step()

                sum_d += float(d_loss.item()) * bs
                n_d += bs
                did_d_any = True

            # --------------------------------------------------
            # 2) 生成器更新
            #    最终统一版：
            #    L_G = lambda_known * CE_known + CE_unk
            #          + lambda_gdice * GDice_unk
            #          + lambda_fm * FM
            #          + lambda_adv * Adv
            # --------------------------------------------------
            set_requires_grad(G, True)
            set_requires_grad(Dnet, False)

            g_ce, ce_k, ce_u, acc_k, acc_u = loss_ce_known_unknown_simple(
                logits_kdhw, y_dhw, m_dhw,
                lambda_known=lambda_known
            )
            g_loss = g_ce

            gdice = loss_gdice_unknown(logits_kdhw, y_dhw, m_dhw)
            g_loss = g_loss + lambda_gdice * gdice

            fm = torch.tensor(0.0, device=device)
            adv = torch.tensor(0.0, device=device)

            if (lam_fm > 0.0) or (lam_adv > 0.0):
                real_slab = onehot_slab_from_target(y_dhw, K)
                fake_slab_for_g = slab_prob_from_logits(logits_kdhw)

                d_in_real_g = torch.cat([x2d, real_slab], dim=1)
                d_in_fake_g = torch.cat([x2d, fake_slab_for_g], dim=1)

                if lam_fm > 0.0:
                    with torch.no_grad():
                        _, feats_real = Dnet(d_in_real_g, return_feats=True)
                    _, feats_fake = Dnet(d_in_fake_g, return_feats=True)
                    fm = loss_feature_matching(feats_real, feats_fake)
                    g_loss = g_loss + lam_fm * fm

                if lam_adv > 0.0:
                    d_fake_g = Dnet(d_in_fake_g)
                    adv = g_hinge_adv_loss(d_fake_g)
                    g_loss = g_loss + lam_adv * adv

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            optG.step()

            ema_update(G, G_ema, ema_decay)

            sum_g += float(g_loss.item()) * bs
            sum_ce_k += float(ce_k.item()) * bs
            sum_ce_u += float(ce_u.item()) * bs
            sum_acc_k += float(acc_k.item()) * bs
            sum_acc_u += float(acc_u.item()) * bs
            sum_gdice += float(gdice.item()) * bs
            sum_fm += float(fm.item()) * bs
            sum_adv += float(adv.item()) * bs
            n += bs

        out = {
            "g": sum_g / max(n, 1),
            "d": (sum_d / max(n_d, 1)) if did_d_any else 0.0,
            "ce_known": sum_ce_k / max(n, 1),
            "ce_unk": sum_ce_u / max(n, 1),
            "acc_known": sum_acc_k / max(n, 1),
            "acc_unk": sum_acc_u / max(n, 1),
            "gdice": sum_gdice / max(n, 1),
            "fm": sum_fm / max(n, 1),
            "adv": sum_adv / max(n, 1),
            "lam_fm": float(lam_fm),
            "lam_adv": float(lam_adv),
            "did_d_any": did_d_any,
        }
        return out

    @torch.no_grad()
    def run_val(loader):
        G_ema.eval()
        Dnet.eval()
        sum_ce_k = sum_ce_u = sum_acc_k = sum_acc_u = 0.0
        sum_disc = sum_conf_u = 0.0
        n = 0
        for x2d, y_dhw, m_dhw in loader:
            x2d = x2d.to(device)
            y_dhw = y_dhw.to(device)
            m_dhw = m_dhw.to(device)

            logits_kdhw = forward_to_kdhw(G_ema, x2d)
            _, ce_k, ce_u, acc_k, acc_u = loss_ce_known_unknown_simple(
                logits_kdhw, y_dhw, m_dhw,
                lambda_known=lambda_known
            )

            fake_slab = slab_prob_from_logits(logits_kdhw)
            d_in_fake = torch.cat([x2d, fake_slab], dim=1)
            d_fake = Dnet(d_in_fake)
            disc_realism = float(d_fake.mean().item())

            prob = torch.softmax(logits_kdhw, dim=1)
            conf = torch.max(prob, dim=1).values
            unk = (m_dhw < 0.5)
            conf_unk = float(conf[unk].mean().item()) if unk.any() else 0.0

            bs = x2d.size(0)
            sum_ce_k += float(ce_k.item()) * bs
            sum_ce_u += float(ce_u.item()) * bs
            sum_acc_k += float(acc_k.item()) * bs
            sum_acc_u += float(acc_u.item()) * bs
            sum_disc += disc_realism * bs
            sum_conf_u += conf_unk * bs
            n += bs

        return {
            "ce_known": sum_ce_k / max(n, 1),
            "ce_unk": sum_ce_u / max(n, 1),
            "acc_known": sum_acc_k / max(n, 1),
            "acc_unk": sum_acc_u / max(n, 1),
            "disc_realism": sum_disc / max(n, 1),
            "conf_unk": sum_conf_u / max(n, 1),
        }

    best_score = -1e9
    best_path = os.path.join(save_dir, "best.pt")
    # best_gan.pt 用于“论文方法强调 GAN 对抗生成”的版本：
    # 只有在对抗训练已经进入一段时间，并且 val_uCE 没有明显偏离稳定最优解时才保存，
    # 防止为了 GAN 视觉锐化而牺牲几何位置一致性。
    best_gan_path = os.path.join(save_dir, "best_gan.pt")
    log_path = os.path.join(save_dir, "train_log.jsonl")
    best_uCE_print = float("inf")
    best_uCE_save = float("inf")
    best_gan_score = -1e9
    no_improve_epochs = 0

    print(f"[SAVE] best      -> {best_path}")
    print(f"[SAVE] best_gan  -> {best_gan_path}")
    print(f"[LOG ] {log_path}")

    for ep in range(1, epochs + 1):
        ep_start = time.time()

        if ep <= stage64_end:
            main_loader = loader64
            mix_loader = loader96
            main_tag = "128x128"
            full_tag = ""
            mix_name = "192x192"
        elif ep < full_main_after:
            main_loader = loader96
            main_tag = "192x192"
            if ep >= use_full_after:
                mix_loader = loaderful
                full_tag = "+full"
                mix_name = "full"
            else:
                mix_loader = loader64
                full_tag = ""
                mix_name = "128x128"
        else:
            main_loader = loaderful
            mix_loader = loader96
            main_tag = "full"
            full_tag = "+96x120"
            mix_name = "192x192"

        tr_main = run_train_one_loader(main_loader, ep, gan_enabled=True)
        tr_mix = run_train_one_loader(mix_loader, ep, gan_enabled=aux_gan_enabled)

        schedG.step()
        if tr_main["did_d_any"] or tr_mix["did_d_any"]:
            schedD.step()

        val = run_val(val_loader)

        # best.pt：仍然严格按照最小 val_uCE 保存，保证几何位置和级联拼接稳定。
        score = (2.00 * val["acc_unk"]) - (3.00 * val["ce_unk"]) + (0.0002 * val["disc_realism"]) + (0.0002 * val["conf_unk"])
        is_best = val["ce_unk"] < (best_uCE_save - early_stop_min_delta)
        ckpt_payload = {
            "epoch": ep,
            "G": G.state_dict(),
            "G_ema": G_ema.state_dict(),
            "D": Dnet.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "best_score": float(score),
            "config": {
                "K": K,
                "epochs": epochs,
                "lr_g": lr_g,
                "lr_d": lr_d,
                "lambda_known": lambda_known,
                "lambda_gdice": lambda_gdice,
                "lambda_fm": lambda_fm,
                "lambda_adv": lambda_adv,
                "use_gan": use_gan,
                "aux_gan_enabled": aux_gan_enabled,
                "adv_start_epoch": adv_start_epoch,
                "fm_start_epoch": fm_start_epoch,
                "adv_ramp_epochs": adv_ramp_epochs,
                "fm_ramp_epochs": fm_ramp_epochs,
                "d_update_every": d_update_every,
                "mc_dropout_p": mc_dropout_p,
                "latent_noise_std": latent_noise_std,
                "stage64_end": stage64_end,
                "use_full_after": use_full_after,
                "full_main_after": full_main_after,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_delta": early_stop_min_delta,
                "early_stop_start_epoch": early_stop_start_epoch,
                "gan_best_start_epoch": gan_best_start_epoch,
                "gan_uce_tolerance": gan_uce_tolerance,
                "gan_dreal_min": gan_dreal_min,
                "best_rule": "min_val_uCE_for_best__guarded_GAN_score_with_dReal_floor_for_best_gan",
            }
        }

        if is_best:
            best_uCE_save = float(val["ce_unk"])
            best_score = score
            no_improve_epochs = 0
            torch.save(ckpt_payload, best_path)
        else:
            no_improve_epochs += 1

        # best_gan.pt：用于突出 GAN 对抗生成的版本。
        # 保存条件：已进入 GAN 阶段，val_uCE 没有明显劣于稳定最优解，且 dReal 不低于保护阈值。
        gan_score = (2.00 * val["acc_unk"]) - (1.20 * val["ce_unk"]) + (0.0020 * val["disc_realism"]) + (0.0010 * val["conf_unk"])
        gan_guard_ok = (
            (ep >= gan_best_start_epoch)
            and (val["ce_unk"] <= best_uCE_save + gan_uce_tolerance)
            and (val["disc_realism"] >= gan_dreal_min)
        )
        if gan_guard_ok and (gan_score > best_gan_score):
            best_gan_score = float(gan_score)
            ckpt_payload["best_gan_score"] = best_gan_score
            ckpt_payload["best_uCE_save_reference"] = float(best_uCE_save)
            ckpt_payload["gan_guard_ok"] = True
            torch.save(ckpt_payload, best_gan_path)

        # 仅用于“打印 best_uCE”：按 channel 风格用 val_unkCE 最小
        if val["ce_unk"] < best_uCE_print:
            best_uCE_print = val["ce_unk"]

        lr_now_g = optG.param_groups[0]["lr"]
        lr_now_d = optD.param_groups[0]["lr"]
        dt_min = (time.time() - ep_start) / 60.0

        # 日志显示主尺度 + 辅助尺度，避免只看到 main_tag 而误认为单尺度训练。
        # 例如：scales=64x80+aux96x120-sup 表示先训练 64x80 main，再训练 96x120 aux，且 aux 不启用 GAN。
        aux_suffix = "" if aux_gan_enabled else "-sup"
        scale_report = f"{main_tag}{full_tag}+aux{mix_name}{aux_suffix}"

        msg = {
            "epoch": ep,
            "time_min": dt_min,
            "train_main": tr_main,
            "train_mix": {"name": mix_name, **tr_mix},
            "val": val,
            "score": float(score),
            "best": bool(is_best),
            "best_score": float(best_score),
            "best_gan_score": float(best_gan_score),
            "best_uCE_print": float(best_uCE_print),
            "lr_g": float(lr_now_g),
            "lr_d": float(lr_now_d),
            "scales": scale_report,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        # ✅打印格式：对齐 train_pairs_11to13_channel.py
        print(
            f"[E{ep:03d}/{epochs}] scales={scale_report} lrG={lr_now_g:.2e} lrD={lr_now_d:.2e} "
            f"trainG={tr_main['g']:.4f} trainD={tr_main['d']:.4f} "
            f"(kCE={tr_main['ce_known']:.4f}, kAcc={tr_main['acc_known']:.3f}, "
            f"uCE={tr_main['ce_unk']:.4f}, uAcc={tr_main['acc_unk']:.3f})  "
            f"auxG={tr_mix['g']:.4f} aux_uCE={tr_mix['ce_unk']:.4f} auxAdv={tr_mix['adv']:.4f}  "
            f"gDice={tr_main['gdice']:.4f} fm={tr_main['fm']:.4f} adv={tr_main['adv']:.4f}  "
            f"val(uCE={val['ce_unk']:.4f}, uAcc={val['acc_unk']:.3f}, "
            f"kAcc={val['acc_known']:.3f}, dReal={val['disc_realism']:.4f}, confU={val['conf_unk']:.4f}) "
            f"best_uCE={best_uCE_print:.4f}  time={dt_min:.1f}min"
        )

        # 早停：在 GAN 阶段后才启用，避免 GAN 还没发挥作用就提前停止。
        if (ep >= early_stop_start_epoch) and (no_improve_epochs >= early_stop_patience):
            print(
                f"[EARLY STOP] val_uCE has not improved for {early_stop_patience} epochs "
                f"after early_stop_start_epoch={early_stop_start_epoch}. "
                f"best_uCE_save={best_uCE_save:.6f}; stop at epoch {ep}."
            )
            break

    print("[DONE] best val_unkCE(print) =", best_uCE_print, "best model:", best_path)
    print("[DONE] best score(save) =", best_score, "best model:", best_path)
    print("[DONE] best GAN score(save) =", best_gan_score, "GAN-priority model:", best_gan_path)


# ============================================================
# 8) main：构建 base_npz + 固定且可复用的 test(split_json) + 训练
# ============================================================
if __name__ == "__main__":
    # --------------------------
    # 基础数据
    # --------------------------
    gt_path = "./dataset/diceng_228_228_228_zyx_change_xiangsu.npy"
    base_npz = "./Ti/train_pairs_11to13_diceng_228_base.npz"

    rebuild_base = False
    if not os.path.exists(base_npz):
        rebuild_base = True
    else:
        # 若旧 base_npz 不是 D=13，自动重建，避免误用其他阶段的数据。
        try:
            _tmp = np.load(base_npz, allow_pickle=True)
            if "cond_label" not in _tmp or int(_tmp["cond_label"].shape[1]) != 13:
                rebuild_base = True
            del _tmp
        except Exception:
            rebuild_base = True

    if rebuild_base:
        os.makedirs(os.path.dirname(base_npz) or ".", exist_ok=True)
        build_pairs_11to13_base_npz(gt_path, base_npz, unknown_val=0)

    data = np.load(base_npz, allow_pickle=True)
    meta_axis = data["meta_axis"]
    meta_pos  = data["meta_pos"]
    N_base = int(data["cond_label"].shape[0])

    # --------------------------
    # 固定测试剖面（不能进训练）
    # 你已指定：xs/ys 都是 [25,50,75,100,125,150,175,200]（共 16 条）
    # --------------------------
    xs = [25, 50, 75, 100, 125, 150, 175, 200]
    ys = [25, 50, 75, 100, 125, 150, 175, 200]
    fixed_test_pairs = make_fixed_test_pairs(xs, ys)

    # --------------------------
    # ✅跨 stage 一致测试集的关键：SAFE_PAD 取“未来最大 PAD”
    # 你计划做到 23→25，其 PAD=12，因此这里必须 >=12，
    # 这样 split_json 里的 (axis,pos) 永远不会在后续更大 PAD 的 stage 出界。
    # --------------------------
    SAFE_PAD_FOR_ALL_STAGES = 12   # 对应 23→25 的 PAD=12
    test_ratio = 0.1              # 训练:测试 = 9:1
    split_seed = 1234

    # ✅所有 stage 复用同一个 split_json（保持测试集一致）
    split_json = f"./Ti/split_pairs_fixed_test_xy_25_50_75_100_125_150_175_200_seed{split_seed}.json"

    # --------------------------
    # 生成/读取 split_json（并做 sanitize，防止旧 json 出界）
    # --------------------------
    if os.path.exists(split_json):
        split_info = load_split_pairs_json(split_json)

        fixed_in_json = [tuple(x) for x in split_info.get("fixed_test_pairs", [])]
        if set(fixed_in_json) != set(fixed_test_pairs):
            raise RuntimeError(
                f"split_json 已存在但 fixed_test_pairs 不一致！\n"
                f"json_fixed={fixed_in_json}\nnow_fixed={fixed_test_pairs}\n"
                f"请删除旧的 {split_json} 或统一 fixed 列表。"
            )

        # 若同一个 split_json 已经由 1→3 生成，则当前 11→13 直接复用其 test_pairs_all，
        # 这样可以保证后续 stage 使用严格同一套测试集，而不是因为当前 N_base 略有变化再重算 test 数。
        current_pairs_set = set((int(a), int(p)) for a, p in zip(meta_axis.tolist(), meta_pos.tolist()))
        old_test_pairs = [tuple(x) for x in split_info.get("test_pairs_all", [])]
        missing_pairs = [kp for kp in old_test_pairs if kp not in current_pairs_set]
        if missing_pairs:
            raise RuntimeError(
                f"已有 split_json 中的部分测试对在当前 11→13 base_npz 中不存在：{missing_pairs[:10]}\n"
                f"这说明 split_json 与当前 stage 不兼容，请检查数据或重新生成 split。"
            )
        print(f"[SPLIT JSON] loaded+reused -> {split_json}")
    else:
        split_info = build_consistent_split_pairs(
            meta_axis=meta_axis,
            meta_pos=meta_pos,
            fixed_test_pairs=fixed_test_pairs,
            test_ratio=test_ratio,
            seed=split_seed,
            safe_pad_for_all_stages=SAFE_PAD_FOR_ALL_STAGES,
            pad_of_this_stage=6,                  # 11→13 本 stage 的 PAD=6
        )
        save_split_pairs_json(split_info, split_json)

    test_pairs_all = [tuple(x) for x in split_info["test_pairs_all"]]

    # --------------------------
    # (axis,pos) -> indices（测试集索引固定）
    # --------------------------
    test_idx_arr = np.asarray(pick_fixed_indices(meta_axis, meta_pos, test_pairs_all), dtype=np.int64)

    test_mask = np.zeros(N_base, dtype=bool)
    test_mask[test_idx_arr] = True
    train_idx_arr = np.arange(N_base, dtype=np.int64)[~test_mask]

    # 打印审计信息：test_total = round(0.1*N)；extra = test_total - 16
    print(f"[FIXED TEST SPEC] xs={xs} ys={ys}  fixed_test_pairs={len(fixed_test_pairs)}")
    print(f"[SPLIT FIXED 9:1] train={len(train_idx_arr)}  test(val)={len(test_idx_arr)}  (N={N_base})")
    print(f"[SPLIT FIXED 9:1] test_total={len(test_idx_arr)} = fixed{len(fixed_test_pairs)} + extra{len(test_idx_arr)-len(fixed_test_pairs)}")
    print(f"[SPLIT FIXED 9:1] SAFE_PAD_FOR_ALL_STAGES={SAFE_PAD_FOR_ALL_STAGES}  split_json={split_json}")

    # --------------------------
    # 训练（val_idx 就是固定 test，数量也固定）
    # --------------------------
    # logtuned 参数说明：
    # 1) 不改损失函数组成，只调整权重和训练节奏；
    # 2) 根据 1→3 日志，GAN 介入过早会抬高 val_uCE，因此本版后移 adv 并降低 D 压制力；
    # 3) 保留轻量随机性，便于后续多组 realization，但避免训练期扰动过强造成模板化或退化。
    train_multiscale_11to13_3scales(
        npz_base_path=base_npz,
        K=7,
        epochs=40,

        lr_g=1.60e-4,
        lr_d=7.50e-6,

        lambda_known=5.8,
        lambda_gdice=0.78,
        lambda_fm=0.026,

        use_gan=True,
        aux_gan_enabled=False,
        adv_start_epoch=16,
        fm_start_epoch=24,
        adv_ramp_epochs=44,
        fm_ramp_epochs=40,
        d_update_every=40,
        lambda_adv=0.00125,
        mc_dropout_p=0.015,
        latent_noise_std=0.0020,

        epoch_size_64=3600,
        epoch_size_96=1800,
        epoch_size_full=80,

        batch_64=4,
        batch_96=2,
        batch_full=1,

        stage64_end=40,
        use_full_after=999,
        full_main_after=999,

        early_stop_patience=8,
        early_stop_min_delta=2e-5,
        early_stop_start_epoch=28,
        gan_best_start_epoch=18,
        gan_uce_tolerance=0.0010,

        gan_dreal_min=-0.40,
        save_dir="./checkpoints_11to13_cgan",
        seed=split_seed,

        train_idx=train_idx_arr,
        val_idx=test_idx_arr,
    )