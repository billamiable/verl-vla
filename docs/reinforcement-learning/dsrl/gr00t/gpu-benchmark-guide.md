# GPU 配置对比测试指南

本文说明如何在不同 GPU 组合上测量 GR00T N1.6 DSRL 流水线（Arena LIBERO Spatial）的耗时，以及需要记录哪些数据。只比较耗时，不关心训练出的策略质量。已有数据和更详细的分析见 [流水线性能基准](performance-benchmark.md)。

## 1. 流水线是什么

一次训练由若干个 **window** 重复组成，每个 window 做两件事，**一个接一个**执行：

1. **Rollout（采样）**：512 个仿真环境（4 张 GPU × 2 个 stage × 64 个环境）和策略交互 20 次。每次交互是“策略推理一次，给出一个 16 步的动作块，仿真器执行这 16 步”。环境结束 episode 时会自动 reset。
2. **Actor 更新**：160 次 SAC 更新。第 1 次更新还会把这次采到的数据加入 replay pool 并把整个 pool 写盘（约 100 GB），所以明显更慢。

GPU 分成两组：**环境 GPU** 跑仿真（Isaac Sim），**模型 GPU** 跑策略推理和 actor 更新。在已测的三个节点上环境都是最大的耗时，占一个 window 的 50%～72%。

## 2. 要测的 GPU 配置

每个配置都是 4 张 GPU 跑环境，4 张 GPU 跑模型，共 8 张。

| 顺序 | 环境 GPU | 模型 GPU | 节点形态 | 状态 | 稳态 window | RTX PRO 5000 快多少 |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | RTX PRO 5000 | RTX PRO 5000 | 单节点 8 卡 | 已测，**推荐** | **13.2 分钟** | — |
| 2 | L20 | L20 | 单节点 8 卡 | 已测 | 23.1 分钟 | 1.75× |
| 3 | H20 | H20 | 单节点 8 卡 | 已测，作为对照（H20 没有 RT 核心，但实测也能运行） | 27.8 分钟 | 2.1× |
| 4 | L20 | H20 | 跨节点 | 已测 | 21.3 分钟 | 1.6× |

- **结论：推荐 RTX PRO 5000 节点。** 它比 L20 快 1.75 倍，比 H20 快 2.1 倍，比跨节点组合快 1.6 倍，环境、推理和 actor 更新都是所有配置中最快的。详细数据和分析见 [流水线性能基准](performance-benchmark.md)。RTX PRO 5000 节点的 CPU 主频也更高（5.0 GHz），推荐的是这一整套节点配置。
- 1、2、3 都在单个节点内完成，没有跨节点网络的干扰。L20 和 H20 节点的 CPU 型号相同，所以这两组的差异主要来自 GPU。
- L20 的环境比 H20 快 1.56 倍，H20 的 actor 更新比 L20 快 1.4 倍。4 把环境放在 L20 节点、模型放在 H20 节点，取两者所长：按单节点的分项估算，稳态 window 约 1240 s，实测 1276 s，跨节点只多了约 3%，但仍比 RTX PRO 5000 单节点（791 s）慢 1.6 倍。
- 4 不需要修改框架代码，按 Ray 资源标签把两个资源池放到两个节点上。启动方法见 [跨节点运行](cross-node.md)。要让 Ray 走两个节点之间最快的网络：在已测的节点上，前端网只有 1 Gb/s，`bond0` 实测 97 Gb/s。
- L20 的 48 GB 显存已确认能放下模型侧。

## 3. 怎么运行

用仓库 `examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/` 里的官方 recipe，按 [recipe 文档](arena-libero-spatial.md) 的 Docker 方式启动，不修改框架代码，只加这些覆盖项：

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
    cluster.actor_rollout_ref.actor.critic.warmup_steps=2 \
  2>&1 | tee train.log
```

- 480 步是 3 个 window，各配置必须一致。其余保持 recipe 默认。
- 用 `-e CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3` 指定哪些卡跑环境（后 4 张）、哪些卡跑模型（前 4 张）。
- 把日志放到本地盘；replay pool 会涨到约 100 GB。
- 有条件每个配置跑 2 次，window 之间本身有约 20% 的波动。

## 4. 记录什么、怎么统计

运行结束后用脚本统计，不需要手工算：

```bash
python scripts/summarize_sac_timing.py train.log --export-csv out_dir --label <配置名>
```

脚本最后打印 **Key metrics**，请把这段输出连同日志和导出的三份 CSV 一起交回。比较时用 window 1 及之后（window 0 的 reset 少，偏快）。

| 指标 | 含义 | 可比性 |
| --- | --- | --- |
| `update_s` | 一次 actor 更新的平均耗时（不含每个 window 的第 1 次更新） | 很稳定，直接可比 |
| `inference_s` | 一次策略推理的等待时间 | 只取决于模型卡，可比 |
| `env step without resets` | 一次交互的耗时，期间没有 episode 结束（取第一次 rollout 中最前面的几次交互） | 直接反映仿真速度，可比 |
| `env_step_s` | 一次交互的平均耗时，包含 reset | 反映实际 window，但 reset 次数取决于策略 |
| `first_update_s` | 每个 window 的第 1 次更新，含写盘 | 取决于存储速度，单独看 |
| `total_s` | 整个 window 的总耗时 | 含 reset，不是严格公平的比较，只作参考 |
| `episodes` | 每次 rollout 完成的 episode 数 | 用来判断各配置的 reset 工作量是否相近 |

比较时更看重前三项：它们不受 reset 的次数影响。

## 5. 每个配置需要记录的环境信息

- GPU 型号、显存、驱动版本。
- **CPU 型号、核数、最高睿频**。Isaac 的主循环是单线程，CPU 对环境耗时影响很大。
- 存储类型（本地 NVMe、RAID 还是网络盘）。
- 跨节点配置：网络类型和带宽。
- 镜像 ID 和代码 commit。
- 从容器启动到第一个训练步的时间。

## 6. 已知的坑

- **逐张探测 GPU**：在已测的 H20 节点上，0 号卡会让 Isaac 启动崩溃，其余卡正常。启动前逐张探测，用 `CUDA_VISIBLE_DEVICES` 把有问题的卡放到模型侧。探测脚本见 [性能基准](performance-benchmark.md) 的排错一节。Isaac Sim 的启动时间因节点而异（H20 70～90 s，L20 约 200 s），探测的 timeout 要留够，否则健康的卡也会因超时没有输出。
- 不要并发启动多个 Isaac 进程，框架已有启动锁，不要绕过。
- 挂载自定义缓存目录时不能是空目录，否则 Isaac 会一直卡在初始化，见同一节。
