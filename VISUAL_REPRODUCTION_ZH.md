# Active Room Segmentation 完整可视化复现

本分支保留上游的房间搜索、门检测、前沿选择和拓扑决策规则，并完成现代
CUDA/Habitat 兼容、实时四视图、结构化事件、回放、严格验证和性能修复。
运行时不允许从 CUDA 门检测静默降级到 CPU，也不会把没有出口或只到达门边
但未跨越的轨迹误报为成功换房。

## 环境

- Python 3.9
- PyTorch 2.7.1 + CUDA 12.8
- Habitat-Lab 0.2.1
- Habitat-Sim 0.2.4
- 固定 DETR 源码与门检测 checkpoint
- 本地物理 X11 桌面，不接受 SSH 转发窗口冒充实时可视化

先执行：

```bash
cd ~/Active_room_segmentation
PROXY_URL=http://10.42.0.1:7890 scripts/bootstrap_modern.sh
```

网络只用于安装阶段。仿真运行会主动清除代理环境变量。

## 多房间可视化运行

远端已有 Matterport3D 资产时：

```bash
cd ~/Active_room_segmentation
scripts/run_mp3d_visual.sh
```

`prepare_mp3d_visual.py` 只引用本机已有 MP3D GLB 和 ObjectNav episode，随后：

1. 为指定场景生成与机器人尺寸一致的 Habitat navmesh。
2. 验证起点和目标均可导航且互相连通。
3. 生成最小 PointNav episode。
4. 写入资产路径、哈希和导航距离清单。

默认场景为 `2azQ1b91cZZ`。可通过环境变量覆盖：

```bash
SCENE_ID=Z6MFQCViBuw MAX_EPISODE_STEPS=2000 scripts/run_mp3d_visual.sh
```

默认上限为 2000 步，严格模式默认开启。只有同时观察到门穿越和房间切换
确认事件，验证才通过。

门穿越采用单一的轨迹几何判据，不使用单帧重检兜底：

1. 拓扑边必须同时存在源房间侧和目标房间侧航点。
2. 每一个规划出口航点都必须由导航器实际到达。
3. 完整轨迹必须从源侧跨过两航点中垂面并进入目标侧。
4. 轨迹终点到目标侧航点必须不超过 6 个地图网格。

任一条件不满足都会记录拒绝事件，并把当前拓扑节点恢复到源房间。
若只是检查安装和单房间运行，可明确关闭：

```bash
REQUIRE_TOPOLOGY_TRANSITION=0 MAX_EPISODE_STEPS=120 \
  scripts/run_habitat_test.sh
```

## 实时窗口

主窗口固定为 2×2 信息布局：

- 左上：当前 RGB 与 DETR 门检测结果。
- 右上：占用/已探索地图、轨迹、机器人朝向、门、前沿和当前目标。
- 左下：房间拓扑有向图，区分 exploring、explored 和 unexplored。
- 右下：探索比例、运行阶段、动作以及最近拓扑事件。

窗口标题包含唯一 run ID。启动脚本会在物理 `seat0` 桌面确认该窗口可见，
并在相隔多个控制步后分别截图，保证不是空白或静态窗口。

## 每次运行的证据

结果位于 `outputs/habitat_test_<run-tag>/`：

- `result.json`：闭环结果、探索量、拓扑状态和 tail 步数。
- `progress.jsonl`：严格连续的每步控制、仿真、阶段和拓扑计数。
- `topology_events.jsonl`：门、节点、边、出口选择和换房事件。
- `topology_final.json`：最终可序列化拓扑图和严格模式结论。
- `visualization_frames/`：完整仪表盘帧序列。
- `visualization_final.png`：最终仪表盘。
- `topology_replay.mp4`：由实际运行帧生成的回放。
- `validation.json`：运行源码、验证器提交、窗口、图片、事件、耗时和闭环验证。

`topology_status` 的含义：

- `cross_room_verified`：门穿越和房间切换均已确认。
- `topology_built_no_confirmed_crossing`：已建图，但没有确认跨房间。
- `single_room_no_exit`：没有形成可用出口，不能声称拓扑复现成功。

## 单独验证和回放

```bash
python scripts/render_replay.py --run-dir outputs/<run>
python scripts/validate_run.py \
  --run-dir outputs/<run> \
  --expected-steps 2000 \
  --run-id <run-id> \
  --require-topology-transition
```

## 已验证回合

物理桌面严格回合：

```text
outputs/habitat_test_mp3d_visual_strict1940_final
```

- 场景：`2azQ1b91cZZ`
- 运行源码：`246520c770cebf6d10fcadbee8e925c81805eef2`
- 步数：1940
- 可视化帧：969
- 最终拓扑：9 个房间、16 条有向边
- 严格结果：`cross_room_verified`
- 穿门：源节点 0 到目标节点 2，40 个轨迹点
- 几何证据：源侧 -75.34 格、目标侧 +13.16 格、终点误差 3.31 格

前沿搜索在相同合成地图上的改动前后基准：

| 已关闭网格 | 改动前 | 改动后 |
| ---: | ---: | ---: |
| 4,900 | 0.820 s | 0.056 s |
| 12,100 | 4.947 s | 0.132 s |
| 22,500 | 17.285 s | 0.258 s |

新容器保留原列表的插入顺序和状态迁移，只把成员查询与删除改为常数时间；
固定地图输出和重复运行结果由测试覆盖。

Gibson 正式论文基准仍需要用户有权使用的 Gibson mesh 和匹配的 PointNav
episodes。本仓库不会下载、重新分发或用其他场景冒充 Gibson 指标。
