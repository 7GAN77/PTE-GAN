# -*- coding: utf-8 -*-
"""
run_train_all_stages_1to25.py

【用途】
总控脚本：按顺序依次调用以下厚度拓展训练脚本：
    train_pairs_1to3.py
    train_pairs_3to5.py
    train_pairs_5to7.py
    ...
    train_pairs_23to25.py

适用场景：
1) 已经分别写好了每个阶段的训练脚本；
2) 希望只运行一个主函数/主脚本，就能依次完成 1→3、3→5、...、23→25 的训练；
3) 若某一阶段报错，默认立即停止，避免后续阶段基于错误状态继续训练。

【推荐放置位置】
把本脚本放在所有 train_pairs_*to*.py 训练脚本所在的同一目录下，
并在该目录打开终端运行：

    python run_train_all_stages_1to25.py

【常用命令】
1) 正常依次训练全部阶段：
    python run_train_all_stages_1to25.py

2) 只检查会运行哪些脚本，不真正训练：
    python run_train_all_stages_1to25.py --dry-run

3) 从 9→11 开始训练到 23→25：
    python run_train_all_stages_1to25.py --start 9 --end 25

4) 只训练指定阶段：
    python run_train_all_stages_1to25.py --stages 1to3,3to5,23to25

5) 如果某些阶段已经有 best.pt 或 best_gan.pt，则跳过：
    python run_train_all_stages_1to25.py --skip-finished

6) 若确实只想运行当前目录中存在的阶段，允许缺失中间脚本：
    python run_train_all_stages_1to25.py --allow-missing

【重要说明】
- 本脚本不改动你的单阶段训练代码，只负责顺序调用。
- 默认每个阶段单独启动一个 Python 子进程；一个阶段结束后进程退出，GPU 显存会被释放，
  比在一个 Python 进程里 import 多个训练脚本更稳。
- 日志会保存到 ./logs_train_all_stages/ 目录，便于查看每个阶段的控制台输出。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# 1. 阶段配置：1→3、3→5、...、23→25
# ============================================================
ALL_STAGES: List[Tuple[int, int]] = [(s, s + 2) for s in range(1, 24, 2)]

# 如果你的脚本不是标准命名，可以在这里手动指定。
# 例如：SCRIPT_OVERRIDES[(7, 9)] = "train_pairs_7to9_diceng_v12_stableGAN.py"
# 默认情况下，本脚本会自动搜索：
#   train_pairs_7to9.py
#   train_pairs_7to9*.py
SCRIPT_OVERRIDES: Dict[Tuple[int, int], str] = {
    # (1, 3): "train_pairs_1to3.py",
    # (3, 5): "train_pairs_3to5.py",
}


# ============================================================
# 2. 工具函数
# ============================================================
def stage_name(stage: Tuple[int, int]) -> str:
    """把 (1, 3) 转为 '1to3'。"""
    return f"{stage[0]}to{stage[1]}"


def parse_stage_token(token: str) -> Tuple[int, int]:
    """解析命令行中的阶段写法，如 '1to3'、'1-3'、'1→3'。"""
    token = token.strip().lower().replace("→", "to").replace("-", "to")
    m = re.fullmatch(r"(\d+)\s*to\s*(\d+)", token)
    if not m:
        raise ValueError(f"无法解析阶段：{token!r}，请使用类似 1to3 或 1-3 的格式。")
    a, b = int(m.group(1)), int(m.group(2))
    if b - a != 2 or a % 2 != 1:
        raise ValueError(f"阶段 {token!r} 不符合 1to3、3to5、...、23to25 的规则。")
    return a, b


def select_stages(args: argparse.Namespace) -> List[Tuple[int, int]]:
    """根据 --start/--end/--stages 选择需要运行的阶段。"""
    if args.stages:
        selected = [parse_stage_token(x) for x in args.stages.split(",") if x.strip()]
    else:
        selected = list(ALL_STAGES)
        if args.start is not None:
            selected = [st for st in selected if st[0] >= args.start]
        if args.end is not None:
            selected = [st for st in selected if st[1] <= args.end]

    invalid = [st for st in selected if st not in ALL_STAGES]
    if invalid:
        raise ValueError(f"存在不合法阶段：{invalid}；合法阶段为：{[stage_name(s) for s in ALL_STAGES]}")
    return selected


def choose_best_match(paths: Sequence[Path], target: str) -> Optional[Path]:
    """
    多个候选脚本中选择最合适的一个。
    优先级：
    1) 文件名完全等于 train_pairs_xtoy.py；
    2) 文件名最短者，通常更接近正式脚本；
    3) 字典序兜底。
    """
    if not paths:
        return None
    exact = [p for p in paths if p.name == target]
    if exact:
        return exact[0]
    return sorted(paths, key=lambda p: (len(p.name), p.name))[0]


def find_script_for_stage(script_dir: Path, stage: Tuple[int, int]) -> Optional[Path]:
    """在 script_dir 中查找某个阶段对应的训练脚本。"""
    name = stage_name(stage)

    if stage in SCRIPT_OVERRIDES:
        p = script_dir / SCRIPT_OVERRIDES[stage]
        return p if p.exists() else None

    exact_name = f"train_pairs_{name}.py"
    exact_path = script_dir / exact_name
    if exact_path.exists():
        return exact_path

    # 兼容类似 train_pairs_7to9(20).py、train_pairs_7to9_diceng_v12_stableGAN.py 的文件名。
    candidates = list(script_dir.glob(f"train_pairs_{name}*.py"))

    # 避免把总控脚本或非训练脚本误选进来。
    candidates = [p for p in candidates if p.name != Path(__file__).name and p.is_file()]
    return choose_best_match(candidates, exact_name)


def discover_scripts(script_dir: Path, stages: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], Optional[Path]]:
    """查找每个阶段的脚本路径。"""
    return {st: find_script_for_stage(script_dir, st) for st in stages}


def checkpoint_exists(project_root: Path, stage: Tuple[int, int]) -> bool:
    """
    判断该阶段是否已经训练完成。
    只要在 checkpoints* 目录中找到对应阶段的 best.pt 或 best_gan.pt，就认为可跳过。
    """
    name = stage_name(stage)
    patterns = [
        f"checkpoints*{name}*/best.pt",
        f"checkpoints*{name}*/best_gan.pt",
        f"checkpoints_{name}*/best.pt",
        f"checkpoints_{name}*/best_gan.pt",
    ]
    for pat in patterns:
        if list(project_root.glob(pat)):
            return True
    return False


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def tee_subprocess(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    env: Optional[dict] = None,
) -> int:
    """
    启动子进程，同时把输出打印到终端并写入日志文件。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        f.write("[CMD] " + " ".join(cmd) + "\n")
        f.write(f"[CWD] {cwd}\n")
        f.write("=" * 80 + "\n")

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)

        return proc.wait()


def run_one_stage(
    python_cmd: str,
    script_path: Path,
    project_root: Path,
    log_dir: Path,
    extra_args: Sequence[str],
) -> int:
    """运行单个阶段训练脚本。"""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{script_path.stem}_{stamp}.log"
    log_path = log_dir / log_name

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    # 使用绝对路径调用脚本，但 cwd 固定为 project_root。
    # 这样你的训练脚本中的相对路径 ./dataset、./Ti、./checkpoints 仍然相对于项目根目录。
    cmd = [python_cmd, str(script_path.resolve()), *extra_args]
    return tee_subprocess(cmd=cmd, cwd=project_root, log_path=log_path, env=env)


# ============================================================
# 3. 主函数
# ============================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="依次训练 PTE-GAN 的 1→3、3→5、...、23→25 阶段。"
    )
    parser.add_argument(
        "--script-dir",
        type=str,
        default=None,
        help="训练脚本所在目录。默认使用本总控脚本所在目录。",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="项目根目录。默认等于 --script-dir。训练脚本中的 ./dataset、./Ti 等相对路径均基于该目录。",
    )
    parser.add_argument(
        "--python",
        dest="python_cmd",
        type=str,
        default=sys.executable,
        help="用于启动训练脚本的 Python 解释器。默认使用当前解释器。",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="从哪个输入厚度开始，例如 --start 9 表示从 9→11 开始。",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="到哪个输出厚度结束，例如 --end 25 表示训练到 23→25。",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="只运行指定阶段，逗号分隔，例如 1to3,3to5,23to25。",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="允许缺少中间脚本；缺失阶段会被跳过。默认不允许缺失。",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某一阶段训练失败后仍继续后续阶段。默认失败即停止。",
    )
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="若检测到该阶段已有 best.pt 或 best_gan.pt，则跳过该阶段。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要运行的阶段和脚本，不真正开始训练。",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="传给每个训练脚本的额外参数。若你的单阶段脚本无命令行参数，可不填。",
    )

    args = parser.parse_args(argv)

    this_file = Path(__file__).resolve()
    script_dir = Path(args.script_dir).resolve() if args.script_dir else this_file.parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir
    log_dir = project_root / "logs_train_all_stages"
    extra_args = args.extra_args.split() if args.extra_args.strip() else []

    stages = select_stages(args)
    script_map = discover_scripts(script_dir, stages)

    print("\n" + "=" * 80)
    print("PTE-GAN 多阶段训练总控脚本")
    print("=" * 80)
    print(f"[script_dir]   {script_dir}")
    print(f"[project_root] {project_root}")
    print(f"[python]       {args.python_cmd}")
    print(f"[log_dir]      {log_dir}")
    print(f"[stages]       {', '.join(stage_name(s) for s in stages)}")
    print("=" * 80)

    missing = [st for st, p in script_map.items() if p is None]
    if missing and not args.allow_missing:
        print("\n[ERROR] 缺少以下阶段的训练脚本：")
        for st in missing:
            print(f"  - train_pairs_{stage_name(st)}.py  或  train_pairs_{stage_name(st)}*.py")
        print("\n请把缺失脚本放到 script_dir 中，或用 --allow-missing 仅运行当前已有脚本。")
        return 2

    print("\n[将按以下顺序运行]")
    runnable: List[Tuple[Tuple[int, int], Path]] = []
    for st in stages:
        p = script_map[st]
        if p is None:
            print(f"  - {stage_name(st):>7s}: MISSING，跳过")
            continue
        if args.skip_finished and checkpoint_exists(project_root, st):
            print(f"  - {stage_name(st):>7s}: 已检测到 checkpoint，跳过 -> {p.name}")
            continue
        print(f"  - {stage_name(st):>7s}: {p.name}")
        runnable.append((st, p))

    if args.dry_run:
        print("\n[dry-run] 仅检查流程，不启动训练。")
        return 0

    if not runnable:
        print("\n[INFO] 没有需要运行的阶段。")
        return 0

    failed: List[Tuple[Tuple[int, int], int]] = []
    all_start = _dt.datetime.now()

    for i, (st, script_path) in enumerate(runnable, start=1):
        name = stage_name(st)
        print("\n" + "#" * 80)
        print(f"[Stage {i}/{len(runnable)}] 开始训练 {name}: {script_path.name}")
        print("#" * 80)

        t0 = _dt.datetime.now()
        ret = run_one_stage(
            python_cmd=args.python_cmd,
            script_path=script_path,
            project_root=project_root,
            log_dir=log_dir,
            extra_args=extra_args,
        )
        dt = (_dt.datetime.now() - t0).total_seconds()

        if ret == 0:
            print(f"\n[OK] {name} 训练完成，用时 {format_seconds(dt)}")
        else:
            print(f"\n[FAILED] {name} 训练失败，返回码={ret}，用时 {format_seconds(dt)}")
            failed.append((st, ret))
            if not args.continue_on_error:
                print("[STOP] 默认失败即停止。若要继续后续阶段，可加 --continue-on-error。")
                break

    total_dt = (_dt.datetime.now() - all_start).total_seconds()
    print("\n" + "=" * 80)
    print(f"[SUMMARY] 总用时：{format_seconds(total_dt)}")

    if failed:
        print("[SUMMARY] 以下阶段失败：")
        for st, ret in failed:
            print(f"  - {stage_name(st)}: return code {ret}")
        print(f"[SUMMARY] 日志目录：{log_dir}")
        return 1

    print("[SUMMARY] 所有已运行阶段均完成。")
    print(f"[SUMMARY] 日志目录：{log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
