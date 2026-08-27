# 精确源码与外部产物说明

最近一次 68 场景 GRScene 指标由两个采集批次合并得到：

| 批次 | 纳入最终评测的场景数 | 源码清单 |
|---|---:|---|
| `grscene_union_fixed0p5_batch32_full7000_20260815_0015` | 44 | `source_manifests/grscene_batch_20260815.sha256` |
| `grscene_remaining25_union_fixed0p5_batch32_dual_clean_20260816_0120` | 24 | `source_manifests/grscene_batch_20260816.sha256` |

两个清单各审计 267 个源码/配置文件。第二批代码就是仓库主工作树中的当前版本。第一批
与第二批只有以下两个文件不同：

- `scripts/run_roomseg_coverage_eval_isaac.sh`；
- `voxroom_online/isaac_runtime/env/isaac_process.py`。

第一批的精确版本保存在 `batch_20260815_overrides/`，两者 SHA256 均与第一批原始清单
完全一致。第一批运行时，把这两个文件覆盖到仓库对应位置即可；其余文件使用主工作树。

清单里的路径是实验机器上的原始绝对路径，用于审计。验证当前主工作树时，可先把
`/home/echo/VoxRoom` 替换为当前仓库绝对路径，再运行 `sha256sum -c`。

## 没有提交的外部内容

- GRScene、InteriorAgent、Habitat 场景资产；
- episode 输出、体素快照、RGB/Depth、GT 标签图和匹配可视化；
- VoxRoom `best.pt`；
- TVARS DETR `model.pth`；
- Isaac Sim 安装目录。

这些内容不是代码。必要 checkpoint 的校验值记录于 `checkpoints.sha256`。

