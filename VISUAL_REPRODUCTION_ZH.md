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

## Gibson 官方场景运行

在已获授权并将官方 Habitat 版 Gibson 解压到
`data/scene_datasets/gibson/` 后，运行：

```bash
cd ~/Active_room_segmentation
SCENE_ID=Swormville MAX_EPISODE_STEPS=2500 \
  scripts/run_gibson_visual.sh
```

入口会先核对已授权下载的 10.83 GB 压缩包 SHA256、492 对 GLB/navmesh、
403 MB PointNav 数据树 SHA256、72 个训练场景文件、994 个验证 episode、
30 个迷你验证 episode，以及全部 86 个被引用场景。所选 GLB/navmesh 还必须
逐字节匹配压缩包条目；随后验证官方 episode 的起终点可导航、路径连通且
测得距离与数据集记录一致，再启动严格实时四视图。
预处理结果写入以 run ID 命名且不可覆盖的独立目录；启动时复制到输出目录，
Habitat 直接读取这份运行内的单 episode 数据，不读取共享的可变中间文件。
加载后还会对实际 episode 的起点、旋转、目标、距离信息和路径字段重新计算摘要，
防止“文件校验正确但仿真载入了另一条 episode”。

默认 `Swormville` 与上游仓库演示场景一致。`MAX_EPISODE_STEPS=2500` 是
硬上限，拓扑探索完成后立即结束，不再用原地转向补足步数。Gibson 入口不允许
关闭严格门穿越验证，也不会下载、替换、重建资产或切换到其他数据集。只有同时
观察到门穿越和房间切换确认事件，验证才通过。

门穿越采用单一的轨迹几何判据，不使用单帧重检兜底：

1. 拓扑边必须同时存在源房间侧和目标房间侧航点。
2. 每一个规划出口航点都必须由导航器实际到达。
3. 完整轨迹必须从源侧跨过两航点中垂面并进入目标侧。
4. 轨迹终点到目标侧航点必须不超过 6 个地图网格。

任一条件不满足都会记录拒绝事件，并把当前拓扑节点恢复到源房间。
严格成功还要求至少一组已确认穿越对应的节点、方向、边 ID 和目标航点仍存在于
最终拓扑中；若后续节点合并使历史边失效，不能继续沿用旧事件宣称成功。
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

窗口标题包含唯一 run ID。启动脚本会把 X11 client PID 与算法进程绑定，
记录确切 client window ID，在物理 `seat0` 桌面取得启动、稍后和中途三张
窗口截图；长回合每 100 控制步继续保存一次物理窗口像素。算法完成后保持窗口，
等待启动器完成终态窗口与桌面截图并回传确认，再允许进程退出。验证器还会分别
检查 RGB、地图、拓扑和状态四个面板非空且从首帧到终帧发生变化。

## 每次运行的证据

结果位于 `outputs/habitat_test_<run-tag>/`：

- `result.json`：闭环结果、探索量、拓扑状态和 tail 步数。
- `input_manifest.json`：本次场景、episode、压缩包和资产摘要。
- `input_dataset.json.gz`：本次实际使用的单 episode PointNav 数据。
- `progress.jsonl`：严格连续的每步控制、仿真、阶段和拓扑计数。
- `topology_events.jsonl`：门、节点、边、出口选择和换房事件。
- `topology_final.json`：最终可序列化拓扑图和严格模式结论。
- `visualization_frames/`：完整仪表盘帧序列。
- `visualization_final.png`：最终仪表盘。
- `topology_replay.mp4`：由实际运行帧生成的回放。
- `validation.json`：运行源码、验证器提交、窗口、图片、事件、耗时和闭环验证。
- `runtime_install.json`：主仓库、DETR、Habitat/SG-Nav、CUDA 和门模型权重指纹。

验证器要求可视化帧严格按配置步频连续，运行上下文在 metadata/result 中一致，
活跃拓扑路径不存在裸异常捕获，并且 Gibson 运行的 `tail` 步数必须为 0。
一旦输出目录出现 Gibson 输入清单和数据，验证器会自动启用严格模式，要求运行
与验证使用同一提交，并且完整、唯一、有序地出现“拓扑完成、控制完成、运行完成”
三类事件；中止事件或缺失完成原因都会直接失败。
最终仪表盘必须在实际最后控制步再渲染一次，步频恰好对齐时的末帧也不能省略。
运行前后会重复验证外部 DETR 提交、权重 SHA256、Habitat/SG-Nav 提交和环境版本，
两次依赖闭包不一致时结果无效。

`topology_status` 的含义：

- `cross_room_verified`：门穿越和房间切换均已确认。
- `topology_built_no_confirmed_crossing`：已建图，但没有确认跨房间。
- `single_room_no_exit`：没有形成可用出口，不能声称拓扑复现成功。

## 单独验证和回放

```bash
python scripts/render_replay.py --run-dir outputs/<run>
python scripts/validate_run.py \
  --run-dir outputs/<run> \
  --expected-steps 2500 \
  --run-id <run-id> \
  --require-topology-transition \
  --allow-early-completion \
  --require-run-context
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
