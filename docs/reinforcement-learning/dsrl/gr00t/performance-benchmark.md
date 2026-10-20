# 流水线性能基准

在四种 8 卡配置上测量 GR00T N1.6 DSRL 流水线（Arena LIBERO Spatial）一个 rollout window 的耗时：RTX PRO 5000、L20、H20 三种单节点，以及 L20 节点跑环境、H20 节点跑模型的跨节点组合。每种配置都是 4 张 GPU 跑仿真，4 张 GPU 跑模型。只测耗时：运行很短，关闭了评测和 checkpoint，不以训练出有用的策略为目的。

测试方法和待测配置见 [GPU 配置对比测试指南](gpu-benchmark-guide.md)，跨节点的启动方法见 [跨节点运行](cross-node.md)。

## 结论：推荐 RTX PRO 5000 节点

**RTX PRO 5000 节点跑完一个稳态 window 只要 13.2 分钟，比 L20 节点（23.1 分钟）快 1.75 倍，比 H20 节点（27.8 分钟）快 2.1 倍，比取 L20 环境和 H20 模型之长的跨节点组合（21.3 分钟）也快 1.6 倍。** 同样一小时，它能跑 4.6 个 window，其他配置只有 2.2～2.8 个。

| 稳态 window（window 1） | RTX PRO 5000 | L20 | H20 | L20 环境 + H20 模型（跨节点） |
| --- | ---: | ---: | ---: | ---: |
| **window 合计** | **791 s（13.2 分钟）** | 1385 s（23.1 分钟） | 1666 s（27.8 分钟） | 1276 s（21.3 分钟） |
| **RTX PRO 5000 快多少** | — | **1.75×** | **2.1×** | **1.6×** |
| 每小时完成的 window | **4.6** | 2.6 | 2.2 | 2.8 |

window 2 给出同样的结论：768 s、1370 s、1628 s、1270 s，RTX PRO 5000 分别快 1.78、2.1 和 1.65 倍。

优势在流水线的每个主要部分都成立，不只来自某一项：

| 可直接比较的指标 | RTX PRO 5000 | L20 | H20 | 跨节点 | RTX PRO 5000 快多少（对 L20 / H20 / 跨节点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不含 reset 的仿真 step（window 0，第 5～7 次调用） | **16.0 s** | 35.3 s | 45.3 s | 34.5 s | **2.2× / 2.8× / 2.2×** |
| 仿真 step，window 1 平均（含 reset） | **19.6 s** | 38.3 s | 59.8 s | 37.7 s | **1.95× / 3.0× / 1.9×** |
| 每次调用的推理等待 | **1.1 s** | 2.0 s | 1.7 s | 1.85 s | **1.8× / 1.6× / 1.7×** |
| 只更新步骤（每步平均） | **1.956 s** | 3.189 s | 2.277 s | 2.500 s | **1.63× / 1.16× / 1.28×** |
| window 的首次更新（window 1，含 replay pool 写盘） | 48.1 s | 39.1 s | 39.9 s | 46.5 s | 0.8× / 0.8× / 1.0× |

- **环境快 2～3 倍是主要来源。** 环境是最大的耗时，所以这一项决定了 window 的差距。L20 只在环境上比 H20 快，RTX PRO 5000 又比 L20 快近 2 倍。
- **模型侧也最快。** 其他配置各有短板：L20 环境比 H20 快，但 actor 更新慢 1.4 倍；H20 更新比 L20 快，但环境最慢。RTX PRO 5000 两边都是最快的，而且没有 NVLink 也比 H20 快。
- **跨节点组合也追不上。** 把环境放在 L20、模型放在 H20，两边各取所长，window 只比 L20 单节点快 8%，环境和模型两边都仍比 RTX PRO 5000 慢，还要多管理一个节点和节点间网络。
- **唯一较慢的一项是每个 window 的首次更新**（48 s 对 39～47 s，每个 window 只有一次，约占 window 的 6%）。这一步要把整个 replay pool（约 98 GB）写盘，RTX PRO 5000 节点用的是软件 RAID 卷，其他节点是单块 NVMe 分区，所以很可能是存储而不是 GPU 的差异，没有单独验证。

## 完整数据

四种配置用同样的代码、配置和覆盖项。单节点时仿真池放在后 4 张 GPU、模型池放在前 4 张；跨节点时仿真池是 L20 节点的后 4 张，模型池是 H20 节点的前 4 张：

| | RTX PRO 5000 节点 | L20 节点 | H20 节点 | 跨节点 |
| --- | --- | --- | --- | --- |
| GPU | 8 × RTX PRO 5000（Blackwell），72 GB，仅 PCIe | 8 × L20，48 GB，仅 PCIe | 8 × H20，96 GB，NVLink | 4 × L20 跑环境 + 4 × H20 跑模型 |
| GPU 有光追核心 | 有 | 有 | 无 | 环境侧有 |
| CPU | 2 × EPYC 9575F（224 线程，最高 5.0 GHz） | 2 × Xeon Platinum 8462Y+（128 线程，最高 2.8 GHz） | 同 L20 | 两个节点都同 L20 |
| replay pool 所在存储 | 软件 RAID 卷 | 单块 NVMe 分区 | 单块 NVMe 分区 | H20 节点的单块 NVMe 分区 |
| 节点间网络 | — | — | — | `bond0`，2 × 100 Gb/s 聚合，实测 TCP 97 Gb/s |
| 运行 | `pro5000`（480 步） | `l20`（480 步） | 运行 A（`numa_aligned`，465 步） | `l20env_h20model`（480 步） |

H20 节点还有一次运行 B（`straddled`），仿真池跨两个 NUMA 节点，只用于下文的「其他测量」。跨节点运行用的是上表中的同一台 L20 节点和同一台 H20 节点。

window 是流水线反复执行的单位：**一次 rollout，然后 160 次 SAC 更新**。稳态 window（window 1）的组成：

| 组成 | RTX PRO 5000 | L20 | H20 | 跨节点 |
| --- | ---: | ---: | ---: | ---: |
| **环境**：20 次仿真 step | 393 s | 766 s | 1195 s | 753 s |
| **Rollout**：20 次策略推理 | 22 s | 40 s | 35 s | 37 s |
| rollout 循环其余（stage 间隔、调度） | 5 s | 11 s | 13 s | 13 s |
| rollout 步其余（等待 reset、切换模式、整理轨迹、准备 batch） | 9 s | 21 s | 19 s | 28 s |
| **Actor 更新**，首步（含 replay pool 写盘） | 48 s | 39 s | 40 s | 47 s |
| **Actor 更新**，其余 159 步 | 314 s | 508 s | 364 s | 398 s |
| **window 合计** | **791 s** | **1385 s** | **1666 s** | **1276 s** |
| 环境占比 / 160 次更新占比 | 50% / 46% | 55% / 40% | 72% / 24% | 59% / 35% |

其他指标：

| | RTX PRO 5000 | L20 | H20 | 跨节点 |
| --- | ---: | ---: | ---: | ---: |
| 仿真 step，window 2 平均 | 18.8 s | 37.5 s | 58.0 s | 37.7 s |
| reset 带来的额外耗时（平均调用减去无 reset 调用） | 4～5 s | 2～3 s | 14～16 s | 约 3 s |
| 每次 rollout 结束的 episode（window 1 / 2） | 987 / 861 | 1018 / 840 | 1023 / 839 | 1025 / 821 |
| Isaac Sim 单进程启动 | 150～165 s | 约 200 s | 70～90 s | 约 200 s（在 L20 上） |
| 从启动容器到第一步 | 约 26 分钟 | 约 34.5 分钟 | 约 16.5 分钟 | 约 35 分钟 |

各配置每次 rollout 结束的 episode 数相近，所以 reset 工作量相近，window 的比较是公平的。window 0 的成功比例不同（RTX PRO 5000 39%，L20 28%，H20 16%，跨节点 11%），见下文的「注意事项」。

### 这些结果说明了什么，没有说明什么

- **推荐的是整个节点配置，不只是 GPU。** RTX PRO 5000 节点的 CPU 也快得多（最高 5.0 GHz 对 2.8 GHz），仿真器主循环在单个 CPU 线程上占满 100%，所以环境上的优势有一部分可能来自 CPU。采购或选型时应同时选高主频 CPU，否则未必能复现这里的倍数。要把 GPU 和 CPU 的贡献分开，需要对单个仿真 step（渲染、物理、reset、主机侧）做 profile。
- **GPU 本身对仿真速度也有影响。** L20 与 H20 的 CPU 型号相同，L20 上的环境仍快 1.3～1.6 倍。L20 和 RTX PRO 5000 有光追核心，H20 没有，而仿真器要渲染相机，这可能是原因之一，但没有 profile。
- **H20 上 reset 特别贵。** reset 带来的额外耗时在 H20 上是 14～16 s，在有光追核心的 GPU 上只有 2～5 s。无 reset 的基准只依据每次运行的 2～3 次调用，这些数要按 ±几秒看待，原因没有调查。
- **跨节点的通信开销很小，但不为零。** 仿真 step 和 L20 单节点相同，推理等待只比 H20 单节点多 0.1 s，说明观测和动作在节点间传输不是瓶颈。按单节点分项拼出的估算是 1240 s，实测 1276 s，多出约 3%。多出来的主要是 rollout 步其余（28 s 对 19～21 s）和只更新步骤（2.50 s 对 H20 单节点 2.28 s，慢 10%）。只更新步骤不涉及环境，为什么变慢没有调查。
- **模型侧不受 GPU 间带宽限制**：没有 NVLink 的 RTX PRO 5000 比有 NVLink 的 H20 更新还快。L20 的 48 GB 显存也足够放下模型侧。
- **启动更慢不影响结论。** RTX PRO 5000 上 Isaac Sim 启动比 H20 慢，但这是一次性开销，正式训练跑几十个 window，每个 window 节省 8～15 分钟。
- **每种配置只跑了一次**，重复运行之间的波动未知。同一次运行里 window 1 和 window 2 相差 0.5%～3%，相对 1.6～2.1 倍的差距很小。

## 流水线如何运行

- 环境是 512 个 Isaac Lab Arena 环境：4 张仿真 GPU × 2 个流水线 stage × 64 个环境。每张仿真 GPU 为每个 stage 运行一个仿真进程，所以每张卡上有两个仿真器。
- **Rollout。** 每个 stage 恰好重复 20 次交互。一次交互把一个 16 步的动作块发给一个仿真器里的全部 64 个环境，等它们都完成后返回。成功或达到 200 步上限的环境会自动 reset 并开始下一个 episode，所以循环长度不依赖于是否成功。
- **Replay pool。** 一个 episode 结束之后才会加入 pool；未结束的先缓存，下个 window 继续。
- **Actor 更新。** 每次更新从 pool 采样（60% 成功样本，40% 失败样本），更新 SAC critic 和 noise actor，并软更新目标 critic。GR00T 策略保持冻结。
- **顺序。** rollout 和更新一个接一个执行，不是并发：更新期间仿真 GPU 空闲，rollout 循环期间模型 GPU 空闲。

## reset 与动作块

一次交互执行一个**动作块**：16 个仿真步，全部执行，对一个仿真器里的 64 个环境都是如此。如果一个环境在块中间结束了 episode，比如第 10 个子步成功了：

1. 动作块不会被截断。IsaacLab 会在同一步里 reset 这个环境，所以第 11～16 个子步作用在刚 reset 的新 episode 上（这部分计算被浪费）。
2. 训练时，块内第一个结束子步之后的奖励被屏蔽。这个块是该 episode 的最后一个 transition，被标记为结束；只要它的任一子步成功，就标记为成功。
3. 块结束时环境被显式地第二次 reset，让下一个块从干净的初始状态开始。

调用要等 16 步以及所有结束环境的 reset 都完成后才返回。它们在仿真进程里一个接一个执行，整批 64 个环境都要等，所以**一次调用的耗时随其中结束的环境数增加**。两次 reset 哪一次更贵没有测量。

**reset 跨 rollout 延续。** recipe 设置了 `auto_reset=true`；这时 `BaseEnv.reset` 会返回已在运行的环境的最新观测，而不是真的 reset。所以只有第一次 rollout 会让所有环境从 episode 开头出发。之后每次 rollout 都从上次 rollout 停下的状态继续：160 次更新期间仿真器是暂停的，之后用更新后的策略继续。一个 episode 因此可以跨越多个 window，由多个版本的策略产生。这对 SAC 是可以接受的，因为它是 off-policy，把每个 transition 都当作有效样本；成功标签是其中比较薄弱的一点：一个结束的 episode 片段，只要任一步成功，它的所有 transition 都被标为成功，这只影响成功与失败样本 60/40 的采样。

## 不含与含 reset 的 step 耗时

本节和下面两节的细节来自 H20 节点的运行 A 和运行 B。一次调用（64 个环境 × 16 步）的耗时，单位秒：

| | 不含 reset（window 0，第 5～7 次调用） | Window 1 | Window 2 |
| --- | ---: | ---: | ---: |
| 运行 A | 45.3 | 61.1 | 59.4 |
| 运行 B | 50.0 | 60.0 | - |

- 不含 reset 时一次调用约 45～50 s，每个环境 step 约 44～49 ms。稳态 rollout 里平均约 59～61 s，所以 reset 让每次调用多约 10～15 s（25～35%）。
- 结束的 episode 不超过 10 个的调用在两次运行里都约 38～51 s，接近无 reset 的耗时。在运行 A 里，结束 100 个以上的调用约 86 s（window 1 有 3 次）或 58 s（window 0 有 2 次），所以代价随 reset 的数量增加，而不是看调用里有没有；这类调用只有几个。
- 无 reset 的数值只依据每次运行的 3 次调用，两次运行之间差 10%，所以这 10～15 s 要按 ±5 s 看待。单独做仿真器基准（16 步不带 reset，再带 N 个 reset）可以直接测出这两个数，也不依赖策略。
- 在 window 0 里，第 1 次交互很快（38～39 s），第 2～4 次交互（46～66 s）比两次运行里的第 5～7 次慢，像是预热效应，没有调查原因。Isaac Sim 的启动和 shader 编译本身发生在第一步之前，不在计时内。

比较硬件时，用不含 reset 的 step 耗时、只更新步骤的耗时和推理等待时间；window 总耗时还包含 reset，而 reset 取决于策略。

## 为什么 window 0 比后面的 window 快

H20 节点运行 A：

| | Window 0 | Window 1 | Window 2 |
| --- | ---: | ---: | ---: |
| 每次仿真 step 耗时 | 48.4 s | 59.8 s | 58.0 s |
| rollout 循环 | 1027 s | 1249 s | 1216 s |
| 这次 rollout 里结束的 episode | 527 | 1023 | 839 |

window 0 是唯一所有环境在同一时刻从 episode 开头出发的 rollout。前 7 次交互里没有 episode 结束；第一批成功出现在第 8～12 次交互，其余的 episode 在第 13 和 14 次交互里一起超时（13 个块 = 208 步，超过 200 步上限），几乎所有 reset 都集中在那里。所以每个环境大约完成一个 episode。从 window 1 起，环境从上次 rollout 继续，处于 episode 的不同位置，20 次交互里每个环境大约完成两个 episode，reset 翻倍，几乎每次调用都有 reset。策略的变化也可能有影响（成功率高则 episode 更短），但成功比例在运行 A 里上升（16%、39%、52%），在运行 B 里下降（85%、49%），所以不能解释这个差异。

window 1 和 window 2 相差在 3% 以内，所以流水线从 window 1 起进入稳态。比较时使用 window 1 及之后。

## 其他测量

- **仿真池放在哪里没有影响。** 运行 B 把仿真池跨两个 NUMA 节点，window 1 里每次仿真 step 58.7 s，运行 A（仿真池在一个 NUMA 节点）是 59.8 s。差约 2%，低于 window 之间约 20% 的波动，所以只排除了很大的影响。
- **window 的首个更新步耗时 38～51 s**，而不是 2.3 s。加入数据后，整个 replay pool 会立刻写盘（运行结束时每个 rank 约 26 GB），这很可能是主要部分。这一归因是推断，没有测量。
- **绝对来说一次仿真 step 很慢**：在 H20 上 64 个环境 × 16 步用 59.8 s，整个节点每秒约 140 个环境 step；最快的 RTX PRO 5000 上也只有约 420 个。时间如何分配到渲染、物理、reset 和主机侧等待，没有测量。

## 原始数据

上面所有数字都以 CSV 保存在 `docs/_static/results/dsrl_gr00t_pipeline_benchmark/`，对应运行 A（`numa_aligned`，465 步）、运行 B（`straddled`，180 步，手动停止）、RTX PRO 5000 节点（`pro5000`，480 步）、L20 节点（`l20`，480 步）和跨节点组合（`l20env_h20model`，480 步）：

| 文件 | 每行对应 | 内容 |
| --- | --- | --- |
| `<run>_steps.csv` | 一个训练步 | 全部 `timing_s/*`、`throughput/*` 和 `count/*` 指标，加上 window 序号 |
| `<run>_rollouts.csv` | 一个 rollout 步 | 该步的计时，以及成功、失败和总 episode 的确切数量 |
| `<run>_interactions.csv` | rollout 的一次交互 | 每次交互的秒数，以及其中结束或成功的 episode（来自进度条，它只跟踪 stage 0） |

四种 GPU 配置的完整训练日志放在 `train_logs/` 下，已去掉 IP、用户名和本机路径：`pro5000.train.log.gz`、`l20.train.log.gz`、`h20.train.log.gz`（运行 A，465 步）和 `l20env_h20model.train.log.gz`。用 `zcat` 解压后可直接交给下面的脚本。

它们由控制台日志生成：

```bash
python scripts/summarize_sac_timing.py train.log --export-csv out_dir --label my_run
```

计时项：`timing_s/step` 是整个步骤，`rollout` 是包含 batch 准备的 rollout，`generate` 是环境循环，`update_actor` 是一次 SAC 更新，`env_loop_env_wait_avg` 是每次仿真 step 的平均等待，`env_loop_rollout_wait_avg` 是每次推理调用的平均等待，`env_loop_stage_wall_max` 是较慢 stage 的墙钟时间。

## 复现

不修改仓库代码。按原样运行 recipe 启动脚本，并加上缩短运行、关闭评测和 checkpoint 的覆盖项：

```bash
docker run --rm --gpus all --network=host --ipc=host --privileged --shm-size=64g \
  -v "$PWD":/workspace/project \
  -v "$CKPT_DIR":/workspace/project/.data/models/checkpoint-10000:ro \
  -v "$LIBERO_DATA/USD":/workspace/project/.data/libero/USD:ro \
  -v "$LIBERO_DATA/assembled_hdf5":/workspace/project/.data/libero/assembled_hdf5:ro \
  -v "$SCRATCH":"$SCRATCH" \
  -w /workspace/project -e RAY_TMPDIR=/tmp/ray -e TASK_ID=7 -e OUTPUT_DIR="$SCRATCH/run" \
  isaaclab_arena:gr00t-runtime \
  bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh \
    trainer.total_training_steps=480 \
    trainer.save_freq=-1 trainer.test_freq=-1 \
    'trainer.logger=[console,tensorboard]' \
    trainer.warm_rollout_steps=1 \
    cluster.actor_rollout_ref.actor.critic.warmup_steps=2
```

| 覆盖项 | 原因 |
| --- | --- |
| `trainer.test_freq=-1`、`trainer.save_freq=-1` | 计时的步骤里不做评测，也不写 14 GB 的 checkpoint |
| `trainer.warm_rollout_steps=1`、`critic.warmup_steps=2` | 默认前 5 步每步都做 rollout，第 6～99 步只训练 critic；这样直接进入 160 步一个 window 的节奏，省掉约 1.5 小时 |
| `trainer.total_training_steps=480` | 3 个 window，每个 160 步（H20 的运行 A 用的是 640，在第 465 步停止） |

`max_interactions=20` 和 `rollout_interval=160` 是 recipe 默认值。环境准备同 [recipe](arena-libero-spatial.md)：用 `docker/Dockerfile.isaaclab_arena` 构建的镜像、LIBERO 资源和 `checkpoint-10000`、任务 7、每个环境 worker 64 个环境、2 个流水线 stage（recipe 里的“256 个环境”只算一个 stage）。要把仿真池放在后 4 张 GPU 上，在 docker 命令前加 `-e CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3`。

对比其他 GPU 配置的步骤和需要记录的数据见 [GPU 配置对比测试指南](gpu-benchmark-guide.md)。

## 注意事项

- 每种配置只跑了一次（H20 的运行 B 是另一种仿真池放置），重复运行之间的波动没有测量。
- 运行 A 在第 465 步停止，所以 window 2 只有 159 个只更新步中的 144 个，window 3 缺失。运行 B 只有 window 1 的 19 个只更新步。
- 所有运行都从同一个 checkpoint 出发，但 window 0 的成功比例不同（H20 运行 A 16%、运行 B 85%，L20 28%，RTX PRO 5000 39%，跨节点 11%），原因没有调查。
- replay pool 在 4 个 window 里会涨到磁盘上约 100 GB；请使用节点本地盘。

## 排错：Isaac Sim 在某张 GPU 上无法启动

在这个节点上，Isaac Sim 在其中一张 GPU 上启动即崩溃，其余七张正常。症状看起来互不相关：出现 `GPU crash is detected` 然后卡住（`Semaphore wait timed out after 60 seconds`）；子进程以 `exitcode=-11`（`Segmentation fault`）退出；或者 Isaac 进程占满 CPU 空转而没有任何 Kit 日志。出现哪一种取决于时序。健康的运行里也会出现警告 `Skipping NVIDIA GPU due CUDA being in bad state`。

逐张探测 GPU（同时启动多个 Isaac Sim 进程在健康的 GPU 上也会段错误，这就是 Arena 环境自己把启动串行化的原因）：

```bash
cat > /tmp/isaac_probe.py <<'EOF'
import time
t = time.time()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True)
print("PROBE_KIT_UP", round(time.time() - t, 1), flush=True)
app.app.close()
EOF

for g in 0 1 2 3 4 5 6 7; do
  docker run --rm --gpus "\"device=$g\"" --network=host --ipc=host --privileged \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v /tmp/isaac_probe.py:/tmp/isaac_probe.py \
    isaaclab_arena:gr00t-runtime timeout 300 /isaac-sim/python.sh /tmp/isaac_probe.py \
    2>&1 | grep -aE "PROBE_KIT_UP|GPU crash|Segmentation" | head -1
done
```

健康的 GPU 在空 shader 缓存下打印 `PROBE_KIT_UP` 的时间因节点而异：H20 约 70～90 s，RTX PRO 5000 约 150～165 s，L20 约 200 s。找到有问题的 GPU 之后，用 `CUDA_VISIBLE_DEVICES` 重新排列 Ray 看到的 GPU 顺序，让它落在只用 CUDA 的模型池里。

### 使用自定义缓存目录时 Isaac Sim 启动卡住

如果把自己的目录挂载到 `.data/gr00t_arena/cache`，要先创建镜像期望的子目录：

```bash
mkdir -p "$CACHE/root_cache/nv" "$CACHE/root_cache/triton/autotune" "$CACHE/root_cache/huggingface"
```

镜像里 `/root/.cache` 是指向 `.data/gr00t_arena/cache/root_cache` 的符号链接。如果这个目录不存在，Isaac Sim 会在 `omni.client` 初始化里占满 100% CPU，对 `stat("/root/.cache/")` 调用几十万次，永远启动不了。recipe 启动脚本自己会创建这些目录。
