# VoxRoom

本仓库是 2026-08-18 GRScene 实验对应的精简代码发布版。旧版
`Active_room_segmentation` 工作树已经被 VoxRoom 当前算法替换，但历史提交仍然保留，
用于复现 TVARS 原版基线。

## 本次发布包含什么

- VoxRoom 在线三维体素建图、Vertical Free Map 和 Nav Free Map 投影；
- VoxRoom 与 TVARS-style Vertical Hough 两类 raw door seed 的并集；
- `19x19` 局部三维编码、`41x41` 二维上下文编码和固定 `0.5` 阈值推理；
- 持久门线、基于 Vertical Free Map 的房间分割与前沿探索；
- Isaac Sim / GRScene 批量运行器；
- 严格 TVARS 原版适配器；
- P、R、F1 和 room mIoU 评测程序；
- 上一次实验的汇总表、逐场景/逐检查点指标和源码哈希清单。

仓库没有上传 GRScene/InteriorAgent/Habitat 场景、人工标注图、体素快照、运行日志和
大模型权重。它们不是代码，并且部分内容受数据集许可或 GitHub 单文件大小限制。

## 上一次实验结果

完整数据表见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)，机器可读汇总见
[`results/grscene_20260818_aggregate.csv`](results/grscene_20260818_aggregate.csv)。

在排除参与门种子网络训练的 10 个 GRScene 后，final 结果为：

| 方法 | 场景数 | P (%) | R (%) | F1 (%) | room mIoU (%) |
|---|---:|---:|---:|---:|---:|
| VoxRoom | 58 | 94.298 | 95.035 | 94.431 | 73.411 |
| TVARS | 58 | 89.086 | 96.769 | 92.567 | 58.031 |

## 固定实验条件

- 地图分辨率：`0.05 m/cell`；
- 小房间过滤：面积小于 `0.5 m²`；
- 分割来源：`voxel_vertical_free_xy`；
- 探索检查点：20%、40%、60%、70%、80%、90% 和 final；
- VoxRoom checkpoint SHA256：
  `941d0f613326630f61a5287579830a2212bce908ddbe8c190bca2125f489f732`；
- TVARS DETR checkpoint SHA256：
  `d971e3b760421eb29665c1ca986854ce8ec57ddcc50209fcc77125c5e3cef7ec`；
- TVARS 原版源码提交：
  `c6dbe92c55ea34f9710ddcc5b10d59144662fe68`；
- 主配置 SHA256：
  `ddc1f76daed49bac8d6d998c0a8a6ccbef21df1086af89762cef598d4793568c`。

权重的期望路径和完整校验值记录在 [`repro/checkpoints.sha256`](repro/checkpoints.sha256)。

## 环境

正式在线实验使用：

- Linux；
- Python 3.11；
- Isaac Sim standalone 5.1.0；
- NVIDIA GPU；
- GRScene/InteriorAgent 场景资产；
- 独立的 TVARS Python 环境。

创建 VoxRoom 环境：

```bash
scripts/setup_voxroom_env.sh
source scripts/activate_voxroom_env.sh
python -m pip install -e .
```

主配置位于 [`configs/voxroom_online.yaml`](configs/voxroom_online.yaml)。先修改其中的
Isaac Sim、数据集和 checkpoint 路径。

## 恢复严格 TVARS 基线

本次实验要求 TVARS 源码恰好位于历史提交 `c6dbe92`，且工作树干净。当前仓库保留了
该提交的历史，因此可以直接建立 worktree：

```bash
git worktree add external_baselines/Active_room_segmentation \
  c6dbe92c55ea34f9710ddcc5b10d59144662fe68
```

然后把 TVARS DETR 权重放在：

```text
external_baselines/Active_room_segmentation/
  detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset/model.pth
```

## 运行 GRScene 实验

准备好 episode 清单和两个 checkpoint 后：

```bash
DATASETS=grscene \
BATCH_ROOT=/data/voxroom_roomseg_evaluation/my_grscene_run \
VOXROOM_ROOT=$PWD \
ACTIVE_ROOM_ROOT=$PWD/external_baselines/Active_room_segmentation \
scripts/run_all_datasets_roomseg_coverage_eval.sh
```

单场景入口为：

```bash
RUN_DIR=/data/voxroom_roomseg_evaluation/debug_scene \
scripts/run_roomseg_coverage_eval_isaac.sh /path/to/scene_episode.jsonl
```

## 重新计算指标

```bash
python scripts/evaluate_approved_roomseg_paper_pr.py \
  --voxroom-index /path/to/index_voxroom.json \
  --tvars-index /path/to/index_tvars_original.json \
  --annotation-dir /path/to/annotations \
  --gt-dir /path/to/final_gt \
  --paper /path/to/Topology-Based_Visual_Active_Room_Segmentation.pdf \
  --out-dir /path/to/metrics \
  --min-room-area-m2 0.5 \
  --cell-size-m 0.05
```

两个采集批次的精确源码清单和第一批的两个差异文件见 [`repro/README.md`](repro/README.md)。

