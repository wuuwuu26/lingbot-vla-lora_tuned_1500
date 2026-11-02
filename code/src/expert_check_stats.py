#!/usr/bin/env python3
"""统计 RoboTwin `--expert-check` 的 ground-truth 专家判定结果。

用途：复现 `04_复现与调优说明/reproduction_report.md` §3「自测结果的解读边界」
中引用的专家成功率区间与不可复现性（翻转率）。

用法：
    python expert_check_stats.py <run-dir> [<run-dir> ...]

`<run-dir>` 为评测产物的运行目录，须含 `episode_info/<setting>__<task>.jsonl`。
传入多个目录时按命令行顺序视为先后批次，用于检测同一 `(setting, task, seed)`
在不同批次间判定是否翻转。

去重键：`(task_config, task_name, seed)`。

判读要点（详见报告 §3）：
  - 专家 `plan_success` 在本镜像上仅约 58%–60%，而应有值≈100%；
  - 同一 key 重跑会翻转（约 6%），因 MPLib 规划器为随机采样（RRT）；
  - 故专家成功率**不构成**策略成功率的上限，也不能与 README 的数字直接比较。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def jsonl_files(root: Path):
    """兼容 <run>/episode_info/*.jsonl 与拍平后的 <run>/*.jsonl 两种布局。"""
    return sorted(root.glob("episode_info/*.jsonl")) or sorted(root.glob("*.jsonl"))


def load(run_dirs):
    """返回 (hist, per_dir)。hist[key] = [(批次序号, plan, task_success), ...]"""
    hist, per_dir = {}, {}
    for di, root in enumerate(run_dirs):
        keys = per_dir.setdefault(di, set())
        for jf in jsonl_files(root):
            for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (d.get("task_config"), d.get("task_name"), d.get("seed"))
                keys.add(key)
                hist.setdefault(key, []).append(
                    (di, bool(d.get("plan_success")), bool(d.get("task_success")))
                )
    return hist, per_dir


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    run_dirs = [Path(p) for p in argv[1:]]
    hist, per_dir = load(run_dirs)

    print("=== 批次 provenance ===")
    for di, root in enumerate(run_dirs):
        print(f"  [{di}] {root.name}: {len(per_dir[di])} 个唯一 key")

    allk = set().union(*per_dir.values()) if per_dir else set()
    n = len(allk)
    if n == 0:
        print("\n未解析到任何 episode_info 记录，请检查路径。")
        return 1
    print(f"\n=== 合并去重后唯一 episode 数: {n} ===")

    # 首次 / 末次观测两种口径
    first = {k: v[0] for k, v in hist.items()}
    fp = sum(1 for k in allk if first[k][1])
    fs = sum(1 for k in allk if first[k][2])
    lp = sum(1 for k in allk if hist[k][-1][1])
    ls = sum(1 for k in allk if hist[k][-1][2])

    print(f"  专家 plan_success: 首次 {fp}/{n} = {100.0*fp/n:.1f}%   末次 {lp}/{n} = {100.0*lp/n:.1f}%")
    print(f"  专家 task_success: 首次 {fs}/{n} = {100.0*fs/n:.1f}%   末次 {ls}/{n} = {100.0*ls/n:.1f}%")

    # 翻转检测：仅统计出现在 >=2 个批次的 key
    multi = {k: v for k, v in hist.items() if len({x[0] for x in v}) > 1}
    plan_flip = [k for k, v in multi.items() if len({x[1] for x in v}) > 1]
    succ_flip = [k for k, v in multi.items() if len({x[2] for x in v}) > 1]
    print(f"\n=== 不可复现性检测 ===")
    print(f"  跨批次可比较的 key: {len(multi)}")
    if multi:
        print(f"  plan_success 翻转: {len(plan_flip)} ({100.0*len(plan_flip)/len(multi):.1f}%)")
        print(f"  task_success 翻转: {len(succ_flip)} ({100.0*len(succ_flip)/len(multi):.1f}%)")
    for k in sorted(set(plan_flip) | set(succ_flip))[:12]:
        seq = " ".join(f"[{di}]{'P' if p else '-'}{'T' if s else '-'}" for di, p, s in hist[k])
        print(f"    {k[1]:<28} seed={k[2]:<10} {seq}")

    # 按 (setting, task) 拆分
    per_task = {}
    for k in allk:
        e = per_task.setdefault((k[0], k[1]), [0, 0, 0])
        e[0] += 1
        e[1] += hist[k][-1][1]
        e[2] += hist[k][-1][2]

    print(f"\n=== 按任务拆分（末次观测口径）===")
    print(f"{'setting':<18}{'task':<34}{'n':>5}{'plan%':>9}{'task%':>9}")
    print("-" * 75)
    for (cfg, task), (cnt, p, s) in sorted(per_task.items()):
        print(f"{cfg:<18}{task:<34}{cnt:>5}{100.0*p/cnt:>8.0f}%{100.0*s/cnt:>8.0f}%")

    zero = [f"{t} 0/{c}" for (cfg, t), (c, p, s) in sorted(per_task.items()) if p == 0]
    if zero:
        print(f"\n专家 plan_success 为 0% 的任务（{len(zero)}/{len(per_task)}）:")
        for z in zero:
            print(f"  {z}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
