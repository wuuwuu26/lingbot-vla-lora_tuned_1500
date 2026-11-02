#!/usr/bin/env python3
"""逐任务成功率的官方—本方案相关性（报告 §3「证据三」的数据来源）。

用途
----
把本方案的逐任务成功率与官方 README 的逐任务表对齐，检查两者是否**同向、近似等比例**。
若差距主要来自评测环境，各任务应同向等比下降，秩相关会较高；实测很弱（Pearson≈0.32 /
Spearman≈0.30），说明差异主要落在「哪些任务被学会」上。

用法
----
    python official_corr.py <run-dir> <官方README.md> [--setting demo_clean]

`<run-dir>` 须含 `logs/eval_<setting>__<task>.log`。

给出两种口径（见报告 §3「测量口径」）：
  - Final 口径  ：分子分母均取日志末次 `Final success rate`（分母为**请求数**）
  - 尝试口径    ：取日志末次滚动行 `Success rate: N/M`（分母为**实际尝试数**）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FINAL = re.compile(r"Final success rate:\s*(\d+)/(\d+)")
ROLL = re.compile(r"Success rate:\s*(\d+)/(\d+)")
# 官方 README 的逐任务行：| `task_name` | 93% | 92% |
OFFICIAL_ROW = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|\s*(\d+)%\s*\|\s*(\d+)%\s*\|")


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def ranks(v: list[float]) -> list[int]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0] * len(v)
    for k, i in enumerate(order):
        out[i] = k
    return out


def read_ours(run_dir: Path, setting: str) -> tuple[dict[str, float], dict[str, float]]:
    fin: dict[str, float] = {}
    att: dict[str, float] = {}
    for path in sorted((run_dir / "logs").glob(f"eval_{setting}__*.log")):
        task = path.name[len(f"eval_{setting}__") : -len(".log")]
        text = ANSI.sub("", path.read_text(encoding="utf-8", errors="ignore")).replace("\r", "\n")
        # 注意：日志的进度行以 \r 覆写，故先统一成 \n，再取最后一次出现
        f = FINAL.findall(text)
        r = ROLL.findall(text)
        if f:
            ok, n = int(f[-1][0]), int(f[-1][1])
            if n:
                fin[task] = 100.0 * ok / n
        if r:
            ok, n = int(r[-1][0]), int(r[-1][1])
            if n:
                att[task] = 100.0 * ok / n
    return fin, att


def read_official(path: Path, col: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = OFFICIAL_ROW.match(line.strip())
        if m and m.group(1) != "Avg":
            out[m.group(1)] = float(m.group(col))  # col=2 → clean, col=3 → randomized
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("official_readme", type=Path)
    ap.add_argument("--setting", default="demo_clean")
    args = ap.parse_args()

    fin, att = read_ours(args.run_dir, args.setting)
    col = 2 if args.setting == "demo_clean" else 3
    off = read_official(args.official_readme, col)
    tasks = sorted(set(off) & set(fin))
    if len(tasks) < 3:
        print(f"可比任务过少（{len(tasks)}），检查 run-dir 与 README 路径。")
        return 1

    of = [off[t] for t in tasks]
    f_ = [fin[t] for t in tasks]
    a_ = [att[t] for t in tasks if t in att]
    print(f"=== {args.setting}：{len(tasks)} 个任务与官方对齐 ===")
    print(f"  官方逐任务均值          : {sum(of)/len(of):.2f}%")
    print(f"  本方案逐任务均值(Final) : {sum(f_)/len(f_):.2f}%")
    if a_:
        print(f"  本方案逐任务均值(尝试)  : {sum(a_)/len(a_):.2f}%")
    print()
    print(f"  Pearson  r(Final 口径)  = {pearson(f_, of):.3f}")
    print(f"  Spearman r(Final 口径)  = {pearson(ranks(f_), ranks(of)):.3f}")
    if len(a_) == len(tasks):
        print(f"  Pearson  r(尝试口径)    = {pearson(a_, of):.3f}")
        print(f"  Spearman r(尝试口径)    = {pearson(ranks(a_), ranks(of)):.3f}")
    print()
    lo = sorted(zip(tasks, f_, of), key=lambda x: x[1])[:5]
    hi = sorted(zip(tasks, f_, of), key=lambda x: -x[1])[:5]
    print("  本方案最弱 5 个任务         本方案    官方")
    for t, o, f in lo:
        print(f"    {t:<30}{o:6.1f}%{f:7.0f}%")
    print("  本方案最强 5 个任务         本方案    官方")
    for t, o, f in hi:
        print(f"    {t:<30}{o:6.1f}%{f:7.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
