#!/usr/bin/env python3
"""统计 RoboTwin 评测日志的「测量口径」与「失败形态」。

用途：复现 `04_复现与调优说明/reproduction_report.md` §3
「自测数字的测量口径」与「失败形态」两节所引用的数字。

用法：
    python eval_diagnostics.py <run-dir> [<run-dir> ...]

`<run-dir>` 为评测产物目录，须含 `logs/eval_<setting>__<task>.log`
（即 `run_clean_benchmark.py` 的输出布局）。

输出三部分：
  1. 逐任务：请求 / 实际执行 / 跳过 / 成功 / 失败，以及两种口径的成功率。
  2. 失败 episode 的终止步数与步数上限对比 —— 判断失败是「步数耗尽」还是「中途判负」。
  3. 成功 episode 的步数分布 —— 作为对照。

口径说明（详见报告 §3）：
  - `Final success rate`（`eval_policy_xpolicylab.py:349`）的分母是**请求数**；
  - 滚动打印的 `Success rate`（`:1025` 的 `task_env.test_num`）的分母是**实际尝试数**；
  - 两者在存在 `UnStableError` 跳过时不一致，故本脚本同时给出两种口径。

终端的进度行以 `\\r` 覆盖写出，故解析前统一把 `\\r` 视为换行。
ANSI 颜色码先剥离，否则行首匹配会失败。
"""
import glob
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FINAL = re.compile(r"Final success rate: (\d+)/(\d+)")
STEP = re.compile(r"step: (\d+) / (\d+)")


def strip(text):
    return ANSI.sub("", text.replace("\r", "\n"))


def episodes(text):
    """把单个任务的日志切成 episode 块，返回 [(成功?, 最大步数, 步数上限), ...]。

    块边界取 `Success rate:` 行 —— 它在每个 episode 结束时打印一次
    （`eval_policy_xpolicylab.py:1027-1031`）。
    """
    text = strip(text)
    blocks, pos = [], 0
    for m in re.finditer(r"Success rate:", text):
        end = text.find("\n", m.start())
        end = len(text) if end == -1 else end
        blocks.append(text[pos:end])
        pos = end
    out = []
    for b in blocks:
        pairs = [(int(s), int(lim)) for s, lim in STEP.findall(b)]
        if pairs:
            out.append(("Success!" in b, max(s for s, _ in pairs), pairs[0][1]))
        else:
            out.append(("Success!" in b, 0, 0))
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    tot = Counter()
    fails_by_step, fails_at_limit, succ_steps = Counter(), 0, []
    rows = []

    for root in argv[1:]:
        for path in sorted(glob.glob(os.path.join(root, "logs", "eval_*.log"))):
            name = os.path.basename(path)[len("eval_"):-len(".log")]
            text = strip(open(path, encoding="utf-8", errors="ignore").read())
            eps = episodes(text)
            m = FINAL.search(text)
            requested = int(m.group(2)) if m else len(eps)
            done = m is not None
            ok = sum(1 for s, _, _ in eps if s)
            for is_ok, steps, limit in eps:
                if is_ok:
                    succ_steps.append(steps)
                else:
                    fails_by_step[steps] += 1
                    if limit and steps >= limit:
                        fails_at_limit += 1
            rows.append((name, requested, len(eps), ok, done))
            if done:
                tot["req"] += requested
                tot["exe"] += len(eps)
                tot["ok"] += ok

    print(f"{'task':<40}{'请求':>6}{'执行':>6}{'跳过':>6}{'成功':>6}{'失败':>6}"
          f"{'Final口径':>11}{'尝试口径':>11}")
    print("-" * 96)
    for name, req, exe, ok, done in rows:
        skip = req - exe if done else None
        f_rate = f"{100.0 * ok / req:.1f}%" if done and req else "-"
        a_rate = f"{100.0 * ok / exe:.1f}%" if exe else "-"
        tag = "" if done else "  *进行中"
        print(f"{name:<40}{req:>6}{exe:>6}"
              f"{(skip if skip is not None else '-'):>6}{ok:>6}{exe - ok:>6}"
              f"{f_rate:>11}{a_rate:>11}{tag}")

    if tot["req"]:
        print("-" * 96)
        print(f"{'已完成任务合计':<40}{tot['req']:>6}{tot['exe']:>6}"
              f"{tot['req'] - tot['exe']:>6}{tot['ok']:>6}"
              f"{tot['exe'] - tot['ok']:>6}"
              f"{100.0 * tot['ok'] / tot['req']:>10.1f}%"
              f"{100.0 * tot['ok'] / tot['exe']:>10.1f}%")

    print("\n=== 失败 episode 的终止步数分布 ===")
    n_fail = sum(fails_by_step.values())
    if not n_fail:
        print("  （无失败）")
    else:
        for step, cnt in sorted(fails_by_step.items()):
            print(f"  {step:>5} 步: {cnt:>4} 次")
        print(f"  合计 {n_fail} 个失败 episode；其中终止步数已达日志中给出的"
              f"步数上限的: {fails_at_limit}/{n_fail}"
              f"（{100.0 * fails_at_limit / n_fail:.1f}%）")
        print("  判读：若该比例接近 100%，则失败形态是「步数耗尽」而非「中途判负」。")
        print("        步数上限取自官方 env_cfg/task_config/_eval_step_limit.yml，")
        print("        本方案未修改该文件。")

    if succ_steps:
        succ_steps.sort()
        print(f"\n=== 成功 episode 的步数分布 ===\n"
              f"  min={succ_steps[0]}  中位={succ_steps[len(succ_steps) // 2]}  "
              f"max={succ_steps[-1]}  n={len(succ_steps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
