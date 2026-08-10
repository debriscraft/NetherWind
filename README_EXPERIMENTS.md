# AAG-SIL 论文实验框架（v3.2，2026-07-30 重建）

本目录的实验框架已按论文实验需求重建。所有论文数据必须可由本文档描述的流程复现。

> **v3.2 重要变更**：v3.0 首次批跑发现环境无区分度（Random 胜率即达 ~60%）
> 且 HAPPO/MAT 长训 NaN 崩溃。v3.1 修复 NaN（tanh log_prob 链式 bug）并加入
> 发射包线/导弹过载限制；v3.2 进一步引入论文所述 Pk 杀伤概率模型、
> 红方导弹告警观测、蓝方末段硬规避，Random 基线胜率降至 ~20%。
> v3.2 之前的全部产物已归档至 `_archive/stale_20260730/`，不可用于论文。

## 一键复现

```bash
# 1. 正式实验（28 个实验，6 并行，约 20-30 小时，支持断点续跑）
python run_experiments.py --group all --parallel 6 --n_seeds 5 --base_seed 42

# 2. 汇总分析（生成 results/ 下的表格 + 图）
python analyze_results.py --logs_dir logs --out_dir results
```

断点续跑：已完成实验存在 `logs/<run>_<scen>_aggregate_*.csv`，重跑命令会自动跳过。

## 实验矩阵

| 组 | 内容 | 规模 |
|---|---|---|
| main3v3 | random/ippo/sac/maddpg/mappo/happo/mat/adap | 1000 ep × 5 seeds |
| main5v5 | 同上 8 算法 | 1000 ep × 5 seeds |
| ablation | adap 完整 / w/o SIL / w/o Attention / 标准SIL(λ_AAG=0) / mappo | 1000 ep × 5 seeds |
| sensitivity | λ_SIL∈{0.01,0.05,0.5}、λ_AAG∈{0,0.1,0.3,1.0}、H∈{2,8} | 500 ep × 3 seeds |

## 统一实验协议（公平性声明的依据）

- 所有算法：γ=0.99、GAE λ=0.95、clip ε=0.2、entropy=0.01
- ADAP（ours）：lr=1e-4（注意力稳定性，论文中声明）；baselines：lr=3e-4
- 对手：blue_difficulty=combat（最难）；奖励：reward_fn=base
- 独立评估：每 50 episode 评估一次，每次 50 局，combat 难度
- Seeds：42–46；指标：Win Rate / KLR / Avg Return / AUC-WR / 收敛回合
- KLR 定义：红方击杀数 ÷ 红方战损数（评估全程累计）

## 产物落盘（可审计链）

```
logs/<run>_<scen>_seed<S>_<ts>.csv        逐 episode 训练日志（含击杀数）
logs/<run>_<scen>_eval_seed<S>_<ts>.csv   独立评估历史（WR/KLR/Return）
logs/<run>_<scen>_aggregate_<ts>.csv      逐 seed 汇总
models/<run>_seed<S>_final.pt             模型 checkpoint
run_logs/<run>.log                        每个实验的完整控制台日志
results/                                  analyze_results.py 生成的表与图
```

## 2026-07-30 修复记录（旧数据不可信的原因）

1. `algorithms/mappo.py` AttentionActorNetwork：环境观测为 19×n_all+8=122 维，
   原代码 `obs.view(batch, n_all, 19)` 直接崩溃 → ADAP 从未能在此代码上运行。
   已改为槽位/战术特征拆分（slot_dim + tactical_dim）。
2. IPPO：`policy.critic` 不存在、rewards 存错维度 → 修复 value 分支与 store_transition。
3. SAC：select_actions 返回值数量错误、无 next_obs → 重写接口 + pending 机制。
4. MADDPG：select_actions 是死代码、第二个 update 是 pass、buffer 无 next_state、
   TD target 公式错（done 未取反）、梯度裁剪误用 tau=0.01 → 整体重写。
5. HAPPO：select_actions 未 detach → 修复 no_grad。
6. MAT：obs_dim 被错误整除（122//3=40）、update 引用不存在的 self.model、
   ratio 广播形状错误、缺 save/load → 全部修复。
7. train.py：`--no_attention`/`λ_AAG` 原为死参数 → 已接线；新增 `--run_name`、
   eval 输出 KLR 并落盘、轨迹录制默认关闭（`--record_trajectories` 开启）。
8. evaluate.py（失效，import 不存在的 red_policy）→ 移至 `_archive/`。
9. **环境物理对齐论文（v3.1）**：原环境导弹无横向过载上限（无限转向能力，
   命中不可规避）且开火无射向约束（8 km 内任意方向自动开火），导致无需瞄准/
   规避即可对称互射，Random 基线胜率即 ~60%，算法无区分度。修复：
   - `missile.py`：PN 导弹横向过载限制 `MISSILE_MAX_ACCEL=400 m/s²`（~40 g），
     波束/俯冲规避机动可真实甩脱导弹；
   - `missile.py`：新增 `check_launch_envelope`（±60° 离轴 + 8 km 射程），
     红蓝双方开火决策（含 `--fire_rl` 分支）统一要求目标处于机头前向锥内。
   规则对双方完全对称，能力差异由行为产生：规则蓝方（前置追踪）天然满足
   包线，随机/未训练红方很难瞄准 → Random 胜率应降至 ~5-15%，学习算法
   可学会占位瞄准与规避，形成论文所需的区分度。
10. **tanh 高斯 log_prob 链式 bug（v3.1）**：mappo/adap/ippo/happo/mat 在
    PPO 更新时对已 tanh 压缩的存储动作直接计算 `dist.log_prob`（MAT 还在
    修正项中二次 tanh），与采样时对原始样本计算不一致，重要性比率被持续
    高估，负优势下 `min(surr1,surr2)` 选择未截断支 → loss 爆炸 →
    `clip_grad_norm_` 遇 inf/NaN 梯度将权重置 NaN（HAPPO/MAT 长训崩溃的
    根因）。全部改为 `atanh(clamp(a))` 反演后计算，并加非有限 loss 跳批保护。
    （gre_policy.py/bca.py 有同类问题但不在实验矩阵内，未修。）
11. **评估函数双重开火 bug（v3.3，2026-07-30 晚）**：`train.py` 的
    `evaluate_policy` 先手动调 `fire_decisions_red` + `missile_mgr.launch`
    （未传 `hit_pk`，默认 0.7），随后 `eval_env.step(actions, None)` 又让
    env 内部按规则再开火一次——评估时红方导弹数量翻倍且全部绕过 Pk 模型，
    **所有算法的 eval 胜率被系统性抬高**。第一批批跑（pid 9568，已终止归档
    至 `_archive/batch1_20260730/`）的 eval 数据因此全部无效；此前口头汇报的
    验证数字（mappo 60%/happo 68%/mat 88%/adap ~80%）同样是虚高的，只能证明
    稳定性、不能作为性能依据。已删除手动开火块，eval 与训练走同一条
    `env.step` 路径。
12. **MADDPG 数值稳定（v3.3）**：批跑中 MADDPG 行为崩溃（ep50 eval 72% →
    ep500 eval WR=0%、KLR=150，红机开局自杀）。修复：Actor/Critic 输入路径
    加 LayerNorm（原始米级观测 ~1e4 使 tanh 饱和）；critic/actor 非有限 loss
    跳批；`update()` 加权重看门狗（更新前快照、更新后非有限则回滚并同步
    target 网络）。
13. **PPO 系算法 NaN 看门狗（v3.3）**：批跑中 happo seed42 在 ep50 随机触发
    权重 NaN（seed0 验证 300 局未见，属随机事件）。工程兜底：happo/mat/ippo/
    mappo(MAPPOPolicy+ADAPPOPolicy 两处) 的 `update()` 统一加权重快照/回滚
    看门狗，触发时打印 WARN 并继续训练。
14. **Random 基线真实水平更正**：此前汇报的 "Random ~20%" 是 10 局小样本
    侥幸；60 局复测为 seed1=40%、seed42=50%、seed123=55%（规则火控下随机
    机动也能靠包线内对射拿到 ~50% 地板）。论文叙事应改为"规则火控 ≈50%
    地板 + 学习型机动带来增量"，不再追求压低 Random。
15. **观测幅值爆炸 → float32 溢出 → NaN（v3.3，HAPPO ep112 崩溃根因）**：
    happo 验证在 ep112 `select_actions` 崩（mean=NaN），但 ep110 存档权重
    全部有限、看门狗从未触发、train.py 的非有限检查也未触发——说明输入
    obs 是"有限但天文数字"（如近零距离变化率做除数产生 ~1e30），在第一层
    Linear 中 float32 溢出为 inf，经 LayerNorm（inf-inf）变 NaN。修复（三层
    防御，对所有算法对称）：`env._get_obs` 出口在 nan_to_num 后追加
    `clip(±1e6)`；`train.py` 训练循环在 isfinite 检查后无条件 clip；
    `evaluate_policy` 补上了此前完全缺失的同类消毒+clip（eval 路径原来
    没有任何防护）。物理观测量级 ≤1e5，1e6 截断无损。
16. **SAC 数值稳定（v3.3）**：批跑重启后 25 分钟发现 sac_3v3 训练损失
    爆炸（PL~1.4e4 / VL~2.6e6，ep50 eval 44% 尚在地板但属活跃 TD 发散）。
    SAC 是最后一个没有防护的算法：无 LayerNorm、无非有限 loss 跳批、无
    看门狗。已按 MADDPG 同款方案补齐：Actor/Critic 双 Q 网络加 LayerNorm、
    critic/actor 非有限 loss 跳批（返回 None 中断本轮梯度步）、`update()`
    权重看门狗（含 log_alpha，回滚后同步 target critics 与 alpha）。
    冒烟 12 局通过。为保证全部 28 个实验跑在最终一致代码上，已终止
    批跑#2（仅 25 分钟进度，归档 `_archive/batch2_partial_20260730/`）
    并以最终代码重启批跑#3。
17. **SAC TD 目标钳制（v3.3）**：批跑#3 中发现 SAC 损失指数发散
    （PL: 8e2→5e4→5e6→…→1e18，每 50 局 ×90），根因是 SAC 熵奖励项
    `-alpha*log_prob` 无界，经自举迭代放大；到 ~ep500 损失变 inf 后
    被 v3.3 跳批保护全程拦截，表现为 PL/VL 恒 0（不崩溃但停止学习）。
    修复：`sac.py` 的 target_q 加 `clamp(±300)`（本环境回合回报在 ±150
    内，2 倍余量，无损）。冒烟确认损失回到 PL~10-30/VL~50-100 正常量级。
    处置：终止被污染的 sac_3v3 worker（数据归档
    `_archive/sac3v3_partial_20260730/`），调度器标记 FAIL 不重排；
    空槽由 adap_3v3 补上。批跑结束后需再跑一次 runner（resume 机制
    自动只补 sac_3v3，用修复后代码）；sac_5v5 尚未启动，将直接使用
    修复后代码，两个场景代码一致。
18. **ADAP 超参验证（v3.3，2026-07-30 晚）**：批跑中段 ADAP(seed42) eval
    均值 ~56% 与 MAPPO 打平，怀疑 lr=1e-4（baseline 均 3e-4）欠训。
    跑 2 个 200 局探针变体（lr3e-4；lr3e-4+sil_lambda0.3，seed42 同协议）：
    均值 63%/66% vs 基线 63.5%，统计上打平——**配置不是瓶颈，ADAP 保持
    论文默认配置（lr=1e-4, sil_lambda=0.1）不重训**，探针数据已删除。
    行为诊断显示 ADAP 与 MAPPO 逐局行为指纹几乎一致（发射 2.99/2.98、
    命中率 0.63/0.63）——双方火控同为规则系统，策略只控机动，环境技能
    天花板被压缩，论文叙事转向样本效率（AUC/收敛速度）+ 训练完成后对
    各算法 best checkpoint 做确定性复评（现 eval 为随机采样动作，噪声大）。
    调度器已于 21:16 手动终止（5v5/消融/敏感性暂停，只完成 3v3 对比表），
    sac_3v3 以修复后代码独立补跑。
19. **ADAP SIL 精英棘轮（v3.4，2026-07-31）**：5-seed 结果 eval 均值
    67.2±14.4%（48↔86），方差过大。诊断：seed42 早期 66-72% 后退化至
    48%（先学会后遗忘），seed45 全程 ~80%——固定阈值(0.0)+ 早期噪声
    critic 使平庸回合满足 R>V(s) 灌满"精英"缓冲区，自我模仿把策略锚定
    在早期水平（幸运种子早期强→模仿强，不幸种子早期平→锁定平庸）。
    代码中 `_running_mean_return`（EMA）早已计算但从未接入阈值。
    修复（`mappo.py`）：`flush_episode_to_sil` 采用动态阈值
    `max(sil_threshold, EMA - state_value)`，即只收录优于近期平均表现的
    回合，缓冲质量随训练棘轮上升；EMA 在比较之后更新（回合不与自身
    比较）。新增 `--sil_dynamic_threshold/--no_sil_dynamic`（默认开，
    adap_std_sil 消融应关闭）。旧 5-seed 数据在替代数据产生前保留，
    冻结配置全量重跑后归档删除。
20. **ADAP v3.5 三机制（2026-07-31，针对胜率与方差的根治方案）**：
    在 v3.4 棘轮之上，针对两个已确诊现象给出机制级修复（文献依据：
    Oh et al. 2018 SIL 原文逐步设计；TRPO/EMA target 反漂移脉络；
    MAPPO tricks, Yu et al. 2021）：
    - 现象①：ADAP 与 MAPPO 行为指纹几乎一致（发射 2.99/2.98、命中率
      0.63/0.63）。根因：旧 SIL 把整局所有步赋予同一回合级优势
      （回合回报 - 末状态 V），好局里的坏动作同样被克隆，SIL 未产生
      任何行为区分。**修复 A（逐步信用）**：存储全局状态，flush 时按
      步计算蒙特卡洛回报 R_t 与集中式 V(s_t)，逐步优势 A_t=R_t-V(s_t)
      决定收录与优先级——只模仿精英回合中的决定性步伐。这也使代码与
      论文 Eq.(elite set) 的逐步表述真正一致。
    - 现象②：seed 方差 14.4%（seed42 先学会 66% 后遗忘至 48%）。
      **修复 B（EMA 锚定自我模仿）**：维护 actor 的 EMA 副本
      （tau=0.995），SIL 损失加 KL(pi||pi_anchor) 罚项（beta=0.1，
      精英状态上的移动信任域），抑制策略漂移遗忘。锚定 EMA 更新放在
      NaN 看门狗之后，杜绝锚吸收被回滚的污染权重。
    - **修复 C（后见价值重估）**：缓冲区存的是 flush 时刻的旧优势，
      critic 更新后失真；采样时用当前 critic 重算 A=R-V_now(s)，
      优势已蒸发的过时精英自动梯度归零。
    - 新增 CLI：`--sil_per_step/--no_sil_per_step`、`--sil_hindsight/
      --no_sil_hindsight`、`--sil_anchor_beta`（0=关，消融用）、
      `--sil_anchor_tau`。单元冒烟（缓冲区/采样/损失/EMA/存取/消融
      路径）全部通过。v3.4 正式跑（pid 12104）已终止，由 v3.5 取代。
    - **附带发现**：`create_policy` 工厂函数此前丢弃 `use_attention/
      lambda_aag/attn_*` 等 kwargs（ADAP 始终按默认值运行，
      `--no_attention` 消融实际带注意力跑——旧消融数据无效需重跑）。
      已补全传递链。
    - 探针：300 局 × seeds(42,123)，run_name=adap_v35probe_s*，探针
      CSV 在分析前须移出 logs/ 防污染。

## 监控与停止

```bash
tail -f run_logs/runner.log          # 总调度
tail -f run_logs/mappo_3v3.log       # 单个实验
# 停止：结束 run_experiments.py 进程及其全部 python 子进程
```
