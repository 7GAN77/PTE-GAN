# -*- coding: utf-8 -*-
"""CPU-friendly smoke test for the public PTE-GAN repository.

This test uses the actual 1->3 generator, PatchGAN discriminator, conditioning
encoding, and loss functions from train_pairs_1to3.py.  It performs one tiny
adversarial optimization step on generated seven-class data and then verifies
stochastic inference.  It is intended to test executability, not to reproduce
paper-scale accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from train_pairs_1to3 import (
    UNet2D,
    PatchDiscriminator,
    loss_ce_known_unknown_simple,
    loss_gdice_unknown,
    loss_feature_matching,
    d_hinge_loss,
    g_hinge_adv_loss,
)


def build_1to3_sample(gt: np.ndarray, k: int = 7, axis_flag: int = 0):
    if gt.ndim != 3:
        raise ValueError(f"Expected GT in (Z,Y,X), got {gt.shape}")
    if gt.min() < 1 or gt.max() > k:
        raise ValueError(f"Expected labels 1..{k}, got min={gt.min()} max={gt.max()}")

    z, y, x = gt.shape
    if min(z, y, x) < 16:
        raise ValueError("Quick-test volume is too small; each dimension should be >=16.")

    if axis_flag == 0:  # X-slab
        p = x // 2
        target = np.transpose(gt[:, :, p - 1:p + 2], (2, 0, 1))
    else:  # Y-slab
        p = y // 2
        target = np.transpose(gt[:, p - 1:p + 2, :], (1, 0, 2))

    d, h, w = target.shape
    cond_label = np.zeros_like(target, dtype=np.int16)
    cond_mask = np.zeros_like(target, dtype=np.uint8)
    cond_label[1] = target[1]
    cond_mask[1] = 1

    # Exact repository encoding: K one-hot channels + mask + relative offset + direction,
    # then thickness D is folded into the 2-D channel dimension.
    onehot = np.zeros((k, d, h, w), dtype=np.float32)
    known = cond_mask == 1
    lab = cond_label[known].astype(np.int64) - 1
    di, hi, wi = np.where(known)
    onehot[lab, di, hi, wi] = 1.0

    mask = cond_mask.astype(np.float32)
    offset = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)[:, None, None]
    offset = np.broadcast_to(offset, (d, h, w))
    axis = np.full((d, h, w), float(axis_flag), dtype=np.float32)

    x4d = np.concatenate(
        [onehot, mask[None, ...], offset[None, ...], axis[None, ...]], axis=0
    )
    x2d = np.transpose(x4d, (1, 0, 2, 3)).reshape(d * (k + 3), h, w)
    y_dhw = target.astype(np.int64) - 1
    return (
        torch.from_numpy(x2d[None]).float(),
        torch.from_numpy(y_dhw[None]).long(),
        torch.from_numpy(mask[None]).float(),
    )


def forward_kdhw(net, x2d: torch.Tensor, k: int, d: int) -> torch.Tensor:
    out = net(x2d)
    n, _, h, w = out.shape
    return out.view(n, k, d, h, w)


def onehot_slab(y_dhw: torch.Tensor, k: int) -> torch.Tensor:
    n, d, h, w = y_dhw.shape
    oh = F.one_hot(y_dhw, num_classes=k).permute(0, 4, 1, 2, 3).float()
    return oh.reshape(n, k * d, h, w)


def prob_slab(logits_kdhw: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits_kdhw, dim=1)
    n, k, d, h, w = p.shape
    return p.reshape(n, k * d, h, w)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a minimal PTE-GAN executability test.")
    ap.add_argument("--data", default="./example/example_data.npy")
    ap.add_argument("--out-dir", default="./quick_test_output")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python make_example_data.py --out {data_path}"
        )
    gt = np.load(data_path)

    k, d = 7, 3
    x2d, y, mask = build_1to3_sample(gt, k=k, axis_flag=0)
    x2d, y, mask = x2d.to(device), y.to(device), mask.to(device)

    # base=8 is deliberately small so the smoke test runs quickly on CPU.
    g = UNet2D(
        in_ch=(k + 3) * d,
        out_ch=k * d,
        base=8,
        mc_dropout_p=0.02,
        latent_noise_std=0.002,
        mc_dropout_inference=False,
    ).to(device)
    disc = PatchDiscriminator(in_ch=((k + 3) * d) + (k * d), base=8).to(device)
    opt_g = torch.optim.AdamW(g.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)

    # ---- discriminator step ----
    g.train(); disc.train()
    logits = forward_kdhw(g, x2d, k, d)
    real_slab = onehot_slab(y, k)
    fake_slab = prob_slab(logits).detach()
    d_real = disc(torch.cat([x2d, real_slab], dim=1))
    d_fake = disc(torch.cat([x2d, fake_slab], dim=1))
    loss_d = d_hinge_loss(d_real, d_fake)
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    # ---- generator step: same loss family used by the manuscript code ----
    # Freeze D parameters during the G update; gradients still propagate through D to G.
    for param in disc.parameters():
        param.requires_grad_(False)
    logits = forward_kdhw(g, x2d, k, d)
    ce_total, ce_known, ce_unk, acc_known, acc_unk = loss_ce_known_unknown_simple(
        logits, y, mask, lambda_known=5.8
    )
    gdice = loss_gdice_unknown(logits, y, mask)
    fake_slab = prob_slab(logits)
    with torch.no_grad():
        _, feats_real = disc(torch.cat([x2d, real_slab], dim=1), return_feats=True)
    d_fake_g, feats_fake = disc(torch.cat([x2d, fake_slab], dim=1), return_feats=True)
    fm = loss_feature_matching(feats_real, feats_fake)
    adv = g_hinge_adv_loss(d_fake_g)
    loss_g = ce_total + 0.78 * gdice + 0.026 * fm + 0.00125 * adv

    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()

    for name, value in {
        "loss_g": loss_g,
        "loss_d": loss_d,
        "ce_known": ce_known,
        "ce_unknown": ce_unk,
        "gdice": gdice,
        "feature_matching": fm,
        "adversarial": adv,
    }.items():
        if not torch.isfinite(value).item():
            raise RuntimeError(f"Non-finite {name}: {value.item()}")

    # ---- verify stochastic inference from MC dropout + latent noise ----
    g.eval()
    g.mc_dropout_inference = True
    with torch.no_grad():
        logits_a = forward_kdhw(g, x2d, k, d)
        logits_b = forward_kdhw(g, x2d, k, d)
    stochastic_delta = float(torch.mean(torch.abs(logits_a - logits_b)).item())
    if stochastic_delta <= 0.0:
        raise RuntimeError("Stochastic inference check failed: repeated logits are identical.")

    pred = torch.argmax(logits_a, dim=1)[0].cpu().numpy().astype(np.int16) + 1
    if pred.shape != (3, gt.shape[0], gt.shape[1]):
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
    if pred.min() < 1 or pred.max() > k:
        raise RuntimeError("Predicted labels are outside 1..7.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predicted_1to3_slab.npy", pred)
    metrics = {
        "status": "passed",
        "device": str(device),
        "input_shape_zyx": list(map(int, gt.shape)),
        "conditioning_tensor_shape": list(map(int, x2d.shape)),
        "prediction_shape_dhw": list(map(int, pred.shape)),
        "loss_G": float(loss_g.item()),
        "loss_D": float(loss_d.item()),
        "CE_known": float(ce_known.item()),
        "CE_unknown": float(ce_unk.item()),
        "acc_known": float(acc_known.item()),
        "acc_unknown": float(acc_unk.item()),
        "GDice_unknown": float(gdice.item()),
        "feature_matching": float(fm.item()),
        "adversarial_G": float(adv.item()),
        "stochastic_logit_mean_abs_delta": stochastic_delta,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[PTE-GAN QUICK TEST PASSED]")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
