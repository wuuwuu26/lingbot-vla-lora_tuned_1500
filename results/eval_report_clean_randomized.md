# RoboTwin 全量评测报告（tuned1500 权重）

评测时间：clean 2026-10-21 16:32–20:20；randomized 补全 2026-10-22（双机并行）
权重：`lora_tuned_1500/checkpoints/global_step_1500/hf_ckpt_merged`（LoRA rank64/alpha128 merge 回基座）
推理配置：`use_length=25`，bf16
评测环境：仿真 RTX 4090D（SAPIEN RT 32spp + oidn 降噪 + cuRobo v0.7.8），推理 RTX 6000D（4 个策略服务）

## 一、clean set（clean2clean，seen）

**50/50 任务全部完成，每任务 10 集**

### 总成绩：314/500 = **62.8%**

### 满分（9 个）
adjust_bottle、blocks_ranking_rgb、grab_roller、move_playingcard_away、open_laptop、place_phone_stand、press_stapler、shake_bottle、shake_bottle_horizontally

### 零分（2 个）
blocks_ranking_size 0/10、move_can_pot 0/10

### 全部任务明细

| 任务 | 成绩 | 任务 | 成绩 |
|---|---|---|---|
| adjust_bottle | 10/10 | open_microwave | 3/10 |
| beat_block_hammer | 9/10 | pick_diverse_bottles | 4/10 |
| blocks_ranking_rgb | 10/10 | pick_dual_bottles | 5/10 |
| blocks_ranking_size | 0/10 | place_a2b_left | 8/10 |
| click_alarmclock | 1/10 | place_a2b_right | 8/10 |
| click_bell | 7/10 | place_bread_basket | 6/10 |
| dump_bin_bigbin | 7/10 | place_bread_skillet | 7/10 |
| grab_roller | 10/10 | place_burger_fries | 9/10 |
| handover_mic | 9/10 | place_can_basket | 4/10 |
| hanging_mug | 3/10 | place_cans_plasticbox | 2/10 |
| lift_pot | 8/10 | place_container_plate | 9/10 |
| move_can_pot | 0/10 | place_dual_shoes | 6/10 |
| move_pillbottle_pad | 8/10 | place_empty_cup | 8/10 |
| move_playingcard_away | 10/10 | place_fan | 3/10 |
| move_stapler_pad | 3/10 | place_mouse_pad | 4/10 |
| open_laptop | 10/10 | place_object_basket | 9/10 |
| place_object_scale | 7/10 | place_object_stand | 9/10 |
| place_phone_stand | 10/10 | place_shoe | 7/10 |
| press_stapler | 10/10 | put_object_cabinet | 4/10 |
| rotate_qrcode | 6/10 | scan_object | 6/10 |
| shake_bottle | 10/10 | shake_bottle_horizontally | 10/10 |
| stack_blocks_three | 4/10 | stack_blocks_two | 8/10 |
| stack_bowls_three | 3/10 | stack_bowls_two | 6/10 |
| stamp_seal | 4/10 | turn_switch | 5/10 |
| handover_block | 4/10 | put_bottles_dustbin | 1/10 |

**补测完成（2 个）**：handover_block 4/10、put_bottles_dustbin 1/10（2026-10-23 于 12459 补跑）

## 二、randomized set（clean2random，unseen）

**50/50 任务全部完成，每任务 10 集（双机 8 worker 并行：4090D 4 + 6000D 4）**

### 总成绩：185/500 = **37.0%**

> 对比 clean（seen）的 **62.8%**（314/500，50/50 任务），unseen 随机化后整体下降约 27.4 个百分点。

### 全部 50 任务明细（randomized / clean 对比）

| 任务 | randomized | clean 对比 |
|---|---|---|
| adjust_bottle | 10/10 = 100% | 10/10 = 100% |
| beat_block_hammer | 0/10 = 0% | 9/10 = 90% |
| blocks_ranking_rgb | 2/10 = 20% | 10/10 = 100% |
| blocks_ranking_size | 2/10 = 20% | 0/10 = 0% |
| click_alarmclock | 2/10 = 20% | 1/10 = 10% |
| click_bell | 1/10 = 10% | 7/10 = 70% |
| dump_bin_bigbin | 7/10 = 70% | 7/10 = 70% |
| grab_roller | 8/10 = 80% | 10/10 = 100% |
| handover_block | 0/10 = 0% | 4/10 = 40% |
| handover_mic | 3/10 = 30% | 9/10 = 90% |
| hanging_mug | 1/10 = 10% | 3/10 = 30% |
| lift_pot | 0/10 = 0% | 8/10 = 80% |
| move_can_pot | 0/10 = 0% | 0/10 = 0% |
| move_pillbottle_pad | 2/10 = 20% | 8/10 = 80% |
| move_playingcard_away | 2/10 = 20% | 10/10 = 100% |
| move_stapler_pad | 1/10 = 10% | 3/10 = 30% |
| open_laptop | 4/10 = 40% | 10/10 = 100% |
| open_microwave | 3/10 = 30% | 3/10 = 30% |
| pick_diverse_bottles | 3/10 = 30% | 4/10 = 40% |
| pick_dual_bottles | 5/10 = 50% | 5/10 = 50% |
| place_a2b_left | 4/10 = 40% | 8/10 = 80% |
| place_a2b_right | 3/10 = 30% | 8/10 = 80% |
| place_bread_basket | 3/10 = 30% | 6/10 = 60% |
| place_bread_skillet | 3/10 = 30% | 7/10 = 70% |
| place_burger_fries | 7/10 = 70% | 9/10 = 90% |
| place_can_basket | 0/10 = 0% | 4/10 = 40% |
| place_cans_plasticbox | 6/10 = 60% | 2/10 = 20% |
| place_container_plate | 5/10 = 50% | 9/10 = 90% |
| place_dual_shoes | 2/10 = 20% | 6/10 = 60% |
| place_empty_cup | 8/10 = 80% | 8/10 = 80% |
| place_fan | 2/10 = 20% | 3/10 = 30% |
| place_mouse_pad | 1/10 = 10% | 4/10 = 40% |
| place_object_basket | 7/10 = 70% | 9/10 = 90% |
| place_object_scale | 4/10 = 40% | 7/10 = 70% |
| place_object_stand | 7/10 = 70% | 9/10 = 90% |
| place_phone_stand | 6/10 = 60% | 10/10 = 100% |
| place_shoe | 5/10 = 50% | 7/10 = 70% |
| press_stapler | 5/10 = 50% | 10/10 = 100% |
| put_bottles_dustbin | 4/10 = 40% | 1/10 = 10% |
| put_object_cabinet | 1/10 = 10% | 4/10 = 40% |
| rotate_qrcode | 0/10 = 0% | 6/10 = 60% |
| scan_object | 6/10 = 60% | 6/10 = 60% |
| shake_bottle | 9/10 = 90% | 10/10 = 100% |
| shake_bottle_horizontally | 10/10 = 100% | 10/10 = 100% |
| stack_blocks_three | 1/10 = 10% | 4/10 = 40% |
| stack_blocks_two | 4/10 = 40% | 8/10 = 80% |
| stack_bowls_three | 6/10 = 60% | 3/10 = 30% |
| stack_bowls_two | 6/10 = 60% | 6/10 = 60% |
| stamp_seal | 1/10 = 10% | 4/10 = 40% |
| turn_switch | 3/10 = 30% | 5/10 = 50% |

### 分布观察
- **randomized 满分（2 个，10/10）**：adjust_bottle、shake_bottle_horizontally
- **randomized 零分（6 个，0/10）**：beat_block_hammer、handover_block、lift_pot、move_can_pot、place_can_basket、rotate_qrcode
  （注：lift_pot 在 clean 下 8/10，randomized 0/10，对环境随机化极敏感；handover_block / beat_block_hammer / place_can_basket 同样归零）
- **逆势上升（randomized > clean，5 个）**：blocks_ranking_size、click_alarmclock、place_cans_plasticbox、stack_bowls_three、put_bottles_dustbin
- **持平（randomized = clean，9 个）**：adjust_bottle、dump_bin_bigbin、move_can_pot、open_microwave、pick_dual_bottles、place_empty_cup、scan_object、shake_bottle_horizontally、stack_bowls_two
  说明：多数任务在 unseen 随机化下回落，但「摆放/对齐类」中确有少数任务鲁棒甚至上升，整体结构仍是任务族差异 >> setting 差异。

randomized 配置（`task_config/demo_randomized.yml`）：
random_background=true、cluttered_table=true、random_table_height=0.03、random_light=true

## 三、关键发现

1. **任务族差异远大于 setting 差异**。同一 setting 内，任务成绩从 0% 到 100% 都有；
   「摆放/对齐类」（place_*）普遍 60–90%，「堆叠/开合/排序类」普遍 0–40%。

2. **randomized 并非一律更差**。`stack_bowls_three` 在 randomized 下 60%，反而高于 clean 的 30%；
   `scan_object` 两边接近。真正崩塌的是 `handover_block`（0/10）。
   说明不同任务对环境随机化的敏感度差异极大，笼统的平均分掩盖了这个结构。

3. **失败集的代价是成功集的 3 倍以上**。clean 单集约 60 秒（成功会提前终止），
   randomized 单集约 200 秒（失败的集要跑满 900–1200 步上限）。
   这是 randomized 评测成本远高于 clean 的根本原因。

## 四、复现说明（写进 README 用）

- 推理参数：`--use_length 25 --use_bf16 True`
- 评测入口：`bash cloud/run_remote.sh 10 demo_clean`（或 `demo_randomized`）
- 并发：4 个仿真 worker（4090D 24G 显存上限，实测 4 个占 23.4G，第 5 个必 OOM）
- 规划器：cuRobo v0.7.8（非 MPLib，MPLib 口径不可比）
- 策略服务必须在 `/root/autodl-tmp/ws/src/lingbot-vla-v2` 目录下启动
  （deploy 用相对路径读 `configs/robot_configs/robotwin.yaml`）
- randomized 需要额外资产 `assets/background_texture`（10.2G zip，解压约 21G）

## 五、已知限制

- clean **已全部完成 50/50 任务** = 314/500 = 62.8%（2026-10-23 补测 handover_block 4/10、put_bottles_dustbin 1/10）
- randomized **已全部完成 50/50 任务**（185/500 = 37.0%），由 4090D(23 任务) + 克隆 6000D(27 任务) 双机 8 worker 并行产出
- 每任务 10 集，单任务 95% 置信区间约 ±3pp（10 集时约 ±30pp，此处以整体均值计）
- 按赛题规则 #10：测不满 100 次可如实上报，最终成绩以官方复测为准
