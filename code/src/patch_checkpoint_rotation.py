#!/usr/bin/env python3
"""为 LingBot-VLA 训练代码打上 checkpoint 轮转补丁。

为什么必须打
------------
全量 DCP checkpoint 实测 **72 GB**（optimizer 48 GB + model 24 GB），而 Radeon Cloud
的持久卷 `/workspace` 只有 **100 GB** —— 放不下两份完整 checkpoint。

官方 `Checkpointer.save` 只 `os.makedirs(checkpoint_dir)` 后写入，**从不删除旧目录**，
也没有 keep_last 之类的参数。因此在 11k 步训练中按 `save_steps=1000` 保存，
第 2 次保存就会因为剩余空间（100−72 = 28 GB）不足 72 GB 而写满磁盘、训练中断。

保留策略（为 100 GB 量身定制）
-----------------------------
调用发生在写入新 checkpoint **之前**，此时 `entries[0]` 是最新的一份旧 checkpoint：

- `entries[0]`：删掉 `optimizer/`（48 GB），**保留 `model/`（24 GB）作为兜底**。
  新保存万一失败，仍能从这份 model/ 转换出可用权重提交（只是不能带优化器状态续训）。
- `entries[1:]`：整体删除。

写入新 checkpoint（72 GB）时的占用峰值为 24 + 72 = **96 GB**，卷内仍有余量。

这是纯基础设施改动，**不影响任何训练数值**（不改模型、不改超参、不改数据）。

用法（在实例上）
----------------
    python patch_checkpoint_rotation.py <train_lingbotvla.py 路径>

原始文件会备份为 `<路径>.orig`，便于生成 diff 与回滚。
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HELPER_ANCHOR = "import gc\n"

HELPER = '''

def _rotate_checkpoints(save_root: str) -> None:
    """在写入新 checkpoint 之前腾出磁盘空间。

    全量 DCP 约 72 GB，持久卷仅 100 GB，放不下两份完整 checkpoint；官方保存逻辑
    不做任何清理，第二次保存必然写满磁盘。保留策略见模块文档字符串。
    """
    if not os.path.isdir(save_root):
        return
    if dist.is_initialized() and dist.get_rank() == 0:
        pat = re.compile(r"global_step_(\\d+)$")
        entries = []
        for name in os.listdir(save_root):
            m = pat.fullmatch(name)
            if m:
                entries.append((int(m.group(1)), os.path.join(save_root, name)))
        entries.sort(key=lambda x: x[0], reverse=True)
        for idx, (step, path) in enumerate(entries):
            if idx == 0:
                # 最新的一份旧 checkpoint：只丢 optimizer，保留 model 作兜底
                opt_dir = os.path.join(path, "optimizer")
                if os.path.isdir(opt_dir):
                    print(f"[rotate] trimming optimizer of fallback: {path}", flush=True)
                    shutil.rmtree(opt_dir, ignore_errors=True)
            else:
                print(f"[rotate] removing stale checkpoint: {path}", flush=True)
                shutil.rmtree(path, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()
'''

SAVE_RE = re.compile(
    r"^(?P<indent>[ \t]*)Checkpointer\.save\(args\.train\.save_checkpoint_path, state, global_steps=global_step\)[ \t]*$",
    re.MULTILINE,
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"错误: 找不到 {path}", file=sys.stderr)
        return 1

    src = path.read_text(encoding="utf-8")

    if "_rotate_checkpoints" in src:
        print("已打过补丁，跳过。")
        return 0

    if HELPER_ANCHOR not in src:
        print("错误: 找不到插入锚点 'import gc'", file=sys.stderr)
        return 1

    # 先插辅助函数，再补 import，保证 import 紧跟在 'import gc' 之后
    src = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER, 1)
    if "import shutil" not in src:
        src = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + "import shutil\n", 1)

    src, n = SAVE_RE.subn(
        lambda m: (
            f"{m.group('indent')}_rotate_checkpoints(args.train.save_checkpoint_path)\n"
            f"{m.group('indent')}Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)"
        ),
        src,
    )

    if n == 0:
        print("错误: 未匹配到任何 Checkpointer.save 调用点", file=sys.stderr)
        return 1

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份原文件 -> {backup}")
    path.write_text(src, encoding="utf-8")
    print(f"补丁完成: 插入 {n} 处轮转调用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
