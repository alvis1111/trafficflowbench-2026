# 2026 IEEE Big Data — Traffic Flow Bench (TrafficFlowBench)

IEEE Big Data Cup 2026 交通流竞赛。合成加州高速走廊（D7/D12 区，10 面板，5min 分辨率），预测未来高速状态（速度/流量）、排队、物理量、ODME。

**最终成绩：Public LB 0.80605（Top 50 区）**，从 0.585 起步，+0.22。

评分：`S_total = 0.35*state + 0.30*queue + 0.15*physics + 0.20*odme`。

## 成绩演进

| 里程碑 | 分数 | 关键 |
|---|---|---|
| 官方 naive baseline | 0.5534 | — |
| ML 路线（插值 + LightGBM state） | 0.705 | 自己造 |
| **瓶颈簇队列 + 物理修正** | **0.80605** | 抄公开 notebook |
| 最好公开 notebook | 0.809 | Top 50 |

## 完整配方（可复现 0.806 的四条腿）

1. **state**（插值 + LightGBM）：当天时序插值（速度原始插值 + 流量**残差插值**）→ LightGBM（空间邻居 + 时序 + 静态特征）→ 物理修正。
2. **物理修正**（关键，physics 0.70→0.85）：`soft_fd_projection`（软基本图投影，α=0.15，只调 flow 不调 speed）+ `conservation_correction`（5 遍拓扑守恒，α=0.40）。见 `apply_physics.py`。
3. **queue**（关键，0.30→0.57）：**经验瓶颈簇**（从 273 天 train 硬挖瓶颈 link，onset 预测 IoU 0.752）+ **persistence**（ongoing）。见 `queue_bottleneck.py`。
4. **odme**：NNLS（public_reference，已 0.99）。

## 核心结论

**公开 notebook 是金矿**：`soukeaizenz/0-809-lb-top-50-open-sourcing-physics-pipeline`（0.809 完整开源），核心就是上面的物理修正 + 瓶颈簇，**不是 CTM**。参考代码在 `references/`。

**死路（都验证过，别重走）**：
- CTM 前向做队列 → 失败（需求有巨大日间波动，未来需求不可预测，比 persistence 还差）
- LightGBM 混合做队列 → 失败（过预测，0.666 < 启发式 0.669）
- 匝道感知守恒 → 失败（v7=0.80568 < v6=0.80605）
- 激波扩张/收缩 → 失败（ongoing 排队是稳定均衡，persistence 最优）
- FD 投影单独用 → 失败（S_FD 已 0.99，没空间）

## 剩余 gap（到榜首 0.924）

主导缺口是 **queue 0.57→0.85**（+0.28×0.30=+0.084），公开方法无解，需要完整卡尔曼同化/精细需求建模（leader 的私有路线）。physics 0.85→0.92、state 0.89→0.93 是小头。

## 仓库结构

```
├── scripts/                 # 全部代码（27 个）
│   ├── state_lgbm.py / state_lgbm_full.py   # state 预测（LightGBM）
│   ├── improve_task1.py                      # task1 改进
│   ├── apply_physics.py                      # ★ 物理修正（核心）
│   ├── apply_physics_ramp.py                 # 匝道感知守恒（v7，死路）
│   ├── queue_bottleneck.py                   # ★ 经验瓶颈簇（核心）
│   ├── build_queue_v2.py / queue_ctm.py / queue_kf.py  # 队列方案
│   ├── improve_task2.py / predict_task2_lgbm.py / eval_task2_v2.py / train_task2_*.py / task2_*.py
│   ├── ctm_engine.py / ctm_calibrate.py / ctm_cells.py / ctm_forward.py / ctm_probe.py  # ★ CTM 引擎（验证正确）
│   ├── merge_submissions.py / score_overall.py  # 官方合并/评分
│   └── download.py / list_files.py / spatial_probe.py  # 工具
├── references/              # 公开 notebook 代码 + 竞赛元数据
│   ├── nb_hydro_code.txt / nb_pipeline_code.txt / notebook_code.txt
│   └── corridors.json / competition_contract.json / competition_meta_README.md
└── data/                    # 数据（12GB，需自行下载，见 data/README.md）
```

## 技术资产

- `ctm_engine.py`：细网格 CTM 引擎（cell=检测器 link，1s 时间步，匝道源/汇，三角 FD），**验证正确**（带观测需求边界时 12.8% 一步误差 = 噪声）。
- 逆向结论：生成器 = 细网格 LWR/CTM + 密集匝道（25/44 link）+ 细时间步。
- 提交格式：`submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow`，698 万行。

## 环境注意

- Kaggle API 用 **Bearer token**（`KGAT_` 前缀，非 basic auth）。
- 下载单文件把路径里的 `/` 做 `%2F` 编码。
- submission.csv **别用 Excel 打开**（会截断到 1048576 行）。
- 数据（12GB）在 `C:\Users\GEM07\OneDrive\Desktop\kaggle_public`，不提交。

## 官方 starter

竞赛官方 starter 仓库：`https://github.com/jacky850/trafficflowbench-public`（本仓库 `scripts/merge_submissions.py`、`score_overall.py` 来自它）。
