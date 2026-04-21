# NetherWind — 空战多智能体对抗仿真框架（测试展示版）

> ⚠️ **重要声明**：本仓库上传的脚本仅为**算法验证与效果展示的测试版本**。正式框架涉及保密要求，**源码与核心架构不对外公开**。此处提供的接口、目录结构与运行逻辑仅供参考，不代表最终交付形态。

---

## 一、项目简介

NetherWind 是一套面向**空战对抗场景**的多智能体强化学习训练与仿真框架。本测试版旨在展示：

- 基于 **JSBSim** 的高保真 6-DOF 飞行动力学建模；
- 基于 **Tacview / ACMI** 的 3D 空战过程可视化回放；
- 基于 **MAPPO / 规则基线** 的多智能体决策算法接口；
- 红蓝双方异构智能体的对抗训练与评估流程。

正式版框架在内部网络运行，采用更严格的模块解耦与数据隔离策略，**与本仓库代码不直接对应**。

---

## 二、技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| 动力学仿真 | [JSBSim](https://github.com/JSBSim-Team/jsbsim) | 开源飞行动力学模型，支持 F-16 等机型 |
| 算法训练 | Python + PyTorch | MAPPO、DQN 等基线算法 |
| 可视化回放 | Tacview / ACMI | 标准空战战术分析数据格式 |
| 环境封装 | Gymnasium / 自研接口 | 兼容 `reset()` / `step()` 标准范式 |

---

## 三、目录结构（展示用）

```
NetherWind/
├── jsbsim_env/
│   └── envs/
│   └── cython/
│       ├── __init__.py            # 包入口
│       ├── red_ai.py              # 红方策略接口（开放源码，可定制）
│       ├── dogfight_env.***       # 对抗环境核心（已编译保护）
│       ├── missile.***            # 导弹运动与制导模型（已编译保护）
│       ├── train_and_acmi.***     # 训练主循环 + ACMI 输出（已编译保护）
│       └── ...
├── aircraft/                      # JSBSim 机型与发动机配置
│   └── f16/
├── logs/                          # TensorBoard 日志与训练曲线
└── README.md                      # 本文档
```

> 注：文件名后缀 `***` 表示该模块已做编译保护，仅暴露 Python C-Extension 接口，不公开源码。

---

## 四、快速开始（测试版）

### 1. 环境要求

- Windows 10 / 11
- Python 3.9
- Git
- JSBSim（已集成于项目依赖路径）
- Tacview（用于回放 `.acmi` 文件，需自行安装）

### 2. 安装依赖

```bash
pip install torch numpy gymnasium tensorboard
```

### 3. 运行红方基线（示例）

```python
from cython import red_ai, dogfight_env

# 初始化 1v1 对抗环境
env = dogfight_env.DogfightEnv()

# 加载红方规则基线（开放接口，用户可重写）
agent = red_ai.RedAgent()

obs = env.reset()
for step in range(1000):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### 4. 查看训练曲线

```bash
tensorboard --logdir=logs/training
```

### 5. Tacview 回放

训练结束后，在 `logs/acmi/` 目录下生成 `.acmi` 文件，直接拖拽到 Tacview 即可播放空战过程。

---

## 五、算法接口说明

本测试版仅开放 `red_ai.py` 作为策略热插拔接口，方便研究者接入自有算法进行对比实验：

| 接口 | 类型 | 说明 |
|------|------|------|
| `RedAgent.act(obs)` | 方法 | 输入观测字典，输出动作向量 |
| `RedAgent.reset()` | 方法 | 每局开始时重置内部状态 |
| `DogfightEnv.step(action)` | 方法 | 标准 Gymnasium 步进接口 |
| `DogfightEnv.export_acmi(path)` | 方法 | 导出当前局 ACMI 记录 |

如需接入 PPO / QMIX / MAT 等基线算法，继承 `RedAgent` 并重写 `act()` 即可。

---

## 六、保密与免责

**涉密声明**：本仓库仅用于学术交流与算法效果展示。涉及正式型号参数、内部接口协议、真实作战想定的代码与数据均未上传，也不会通过 GitHub 发布。

**测试性质**：当前脚本为临时验证版本，存在硬编码路径、简化动力学、固定想定等限制，不保证长期维护。

**使用范围**：下载者仅可将本框架作为算法研究的参考实现，禁止用于任何涉密场景或商业目的。

---

## 七、后续计划（正式版方向）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 动力学升级 | 接入完整 JSBSim FDM 外部循环 | 内部测试中 |
| 可视化升级 | UE5 / FlightGear 实时 3D 渲染 | 提供发布版本 |
| 算法仓库 | 支持 MAPPO / QMIX / HAPPO 热插拔 | 内部迭代中 |
| 评估标准 | 标准化空战战术指标（OODA、杀伤链） | 保密 |

---

## 八、联系与引用

个人邮箱 zcx202109@gmail.com
如有技术问题，欢迎通过 GitHub Issues 讨论公开接口层面的问题。涉及内部实现细节恕不回复。


---

**NetherWind Team**   
测试版发布日期：2026-04
