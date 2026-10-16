# 跨节点运行

把 GR00T DSRL recipe 的模型池和仿真池放到两个节点上：一个节点的 4 张 GPU 跑 actor 和 rollout（模型节点），另一个节点的 4 张 GPU 跑 Isaac Sim 环境（环境节点）。两个节点的 GPU 型号可以不同，例如模型放在 H20 节点、环境放在 L20 节点。

不需要修改框架代码。框架按 Ray 资源标签放置资源池，两个启动脚本只负责打标签：

| 脚本 | 在哪里运行 | 作用 |
| --- | --- | --- |
| `start_ray_node.sh` | 两个节点的容器里各运行一次 | 启动 Ray 节点：模型节点是 head，带标签 `train_rollout`；环境节点是 worker，带标签 `sim` |
| `run_train_cross_node.sh` | 模型节点的容器里 | 调用 `run_train.sh`，把模型池放到 `train_rollout`、仿真池放到 `sim`，driver 也留在模型节点 |

两个脚本都在 `examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/`。

## 实测

在一台 8 × L20 节点和一台 8 × H20 节点上按下面的步骤运行过：L20 节点的 4 张 GPU 跑环境，H20 节点的 4 张 GPU 跑模型，Ray 走 `bond0`（实测 TCP 97 Gb/s），完整跑完 480 步。

| | 跨节点 | L20 单节点 | H20 单节点 |
| --- | ---: | ---: | ---: |
| 稳态 window | **1276 s（21.3 分钟）** | 1385 s | 1666 s |
| 每次仿真 step（window 1） | 37.7 s | 38.3 s | 59.8 s |
| 每次推理等待 | 1.85 s | 2.0 s | 1.7 s |
| 只更新步骤 | 2.50 s | 3.19 s | 2.28 s |
| 从启动容器到第一步 | 约 35 分钟 | 约 34.5 分钟 | 约 16.5 分钟 |

跨节点的通信开销很小：仿真 step 和 L20 单节点相同，推理等待只比 H20 单节点多 0.1 s。整体比 L20 单节点快 8%，但仍比 RTX PRO 5000 单节点（791 s）慢 1.6 倍。详细对比见 [流水线性能基准](performance-benchmark.md)。启动时间取决于环境节点，因为 8 个仿真器在环境节点上依次启动。

## 原理

- `cluster.resource.model.resource_label` 和 `cluster.resource.env.resource_label` 给两个资源池各指定一个 Ray 自定义资源。每个 placement group bundle 都会申请一点这个资源，所以资源池只能落在带这个标签的节点上。
- `cluster.resource.controller_label` 把训练主循环（`_run_sac_remote`）放在模型节点，replay pool 和 checkpoint 都在这个节点写盘。
- `+ray_kwargs.ray_init._node_ip_address` 让 driver 用节点间网络的地址注册，和两个 Ray 节点一致。
- 环境 worker 由环境节点上的 raylet 启动，会继承这个 raylet 的环境变量。所以 `start_ray_node.sh` 通过 `/isaac-sim/python.sh` 启动 Ray，让 Isaac Sim 需要的环境变量（`EXP_PATH` 等）也传给环境 worker。

## 前提

- **同一个镜像。** 两个节点用同一个 `isaaclab_arena:gr00t-runtime` 镜像（镜像 ID 相同），保证 Python 和 Ray 版本一致。
- **同样的路径。** 仓库、checkpoint、LIBERO 资源在两个节点的容器里挂载到同样的路径。driver 在模型节点上计算的 `PYTHONPATH` 和 LIBERO 路径会通过 Ray runtime env 传给环境节点上的 worker。共享存储（例如两个节点都挂载的 `/home`）最简单。
- **输出目录两边都要有。** `OUTPUT_DIR` 在两个节点上都要存在且可写：模型 worker 在模型节点写 replay pool（约 100 GB），环境 worker 在环境节点写视频。用节点本地盘，路径两边相同。
- **选对网络。** 每次交互，仿真器的观测要传到模型节点，动作再传回来，所以要让 Ray 走两个节点之间最快的网络。用 `NODE_IP` 指定。节点可能有多块网卡，速度差别很大，先测一下（见下文的「选择网络」）。
- **端口。** 两个节点之间 Ray 的端口要能互通。默认 head 端口是 `26379`，用 `RAY_PORT` 修改，避免和节点上其他人的 Ray 冲突。

## 运行

下面以 H20 节点作模型节点、L20 节点作环境节点为例。两个节点都挂载了共享的 `/home`，节点本地盘是 `/local`。

### 1. 在两个节点上各启动一个容器

两个节点用同样的挂载，`CUDA_VISIBLE_DEVICES` 选出各自要用的 4 张 GPU（Ray 只使用这几张）：

```bash
REPO=/home/$USER/Projects/verl-vla           # 两个节点都能看到的仓库
CKPT=/path/to/checkpoint-10000
LIBERO_DATA=/path/to/libero                  # 包含 USD/ 和 assembled_hdf5/
SCRATCH=/local/$USER/dsrl_cross_node         # 节点本地盘，两个节点上路径相同
GPUS=0,1,2,3                                 # 模型节点；环境节点上例如 4,5,6,7
NAME=dsrl_model                              # 环境节点上例如 dsrl_env

mkdir -p "$SCRATCH"
docker run -d --name "$NAME" --gpus all --network=host --ipc=host --privileged --shm-size=64g \
  -v "$REPO":/workspace/project \
  -v "$CKPT":/workspace/project/.data/models/checkpoint-10000:ro \
  -v "$LIBERO_DATA/USD":/workspace/project/.data/libero/USD:ro \
  -v "$LIBERO_DATA/assembled_hdf5":/workspace/project/.data/libero/assembled_hdf5:ro \
  -v "$SCRATCH":"$SCRATCH" \
  -w /workspace/project -e CUDA_VISIBLE_DEVICES="$GPUS" -e RAY_TMPDIR=/tmp/ray \
  --entrypoint bash isaaclab_arena:gr00t-runtime -c "sleep infinity"
```

`--network=host` 是必须的，Ray 直接使用节点的网卡地址。

### 2. 启动 Ray

先在模型节点上启动 head，`NODE_IP` 是模型节点在节点间网络上的地址：

```bash
docker exec -e ROLE=head -e NODE_IP=192.168.253.45 dsrl_model \
  bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/start_ray_node.sh
```

再在环境节点上加入，`NODE_IP` 是环境节点自己的地址，`HEAD_IP` 是模型节点的地址：

```bash
docker exec -e ROLE=worker -e NODE_IP=192.168.253.31 -e HEAD_IP=192.168.253.45 dsrl_env \
  bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/start_ray_node.sh
```

确认集群里有两个节点、8 张 GPU 和两个标签：

```bash
docker exec dsrl_model /isaac-sim/python.sh -m ray.scripts.scripts status --address=192.168.253.45:26379
```

输出里应该有 `2` 个 active 节点、`0.0/8.0 GPU`、`0.0/1.0 sim` 和 `0.0/1.0 train_rollout`。

### 3. 启动训练

在模型节点的容器里运行。`run_train.sh` 的环境变量和 Hydra 覆盖项都可以照常使用：

```bash
docker exec -d -e HEAD_IP=192.168.253.45 -e TASK_ID=7 -e OUTPUT_DIR=/local/$USER/dsrl_cross_node/run dsrl_model \
  bash -c "bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train_cross_node.sh \
    > /local/$USER/dsrl_cross_node/run/train.log 2>&1"
```

记得先在两个节点上都创建 `OUTPUT_DIR`（这里是 `/local/$USER/dsrl_cross_node/run`）。

### 4. 确认放置正确

Ray 在日志前缀里标出远程节点的地址，本节点的 worker 没有 `ip=`。环境 worker 应该都带环境节点的地址，模型 worker 都没有：

```bash
grep -aoE "\((EnvWorker|VLAActorRolloutRefWorker|_run_sac_remote)[^)]*\)" train.log | sort | uniq -c
```

```text
    101 (EnvWorker pid=516, ip=192.168.253.31)
   1022 (VLAActorRolloutRefWorker pid=8991)
    637 (_run_sac_remote pid=8320)
```

### 5. 结束

```bash
docker exec dsrl_env   /isaac-sim/python.sh -m ray.scripts.scripts stop --force
docker exec dsrl_model /isaac-sim/python.sh -m ray.scripts.scripts stop --force
docker rm -f dsrl_env    # 在环境节点上
docker rm -f dsrl_model  # 在模型节点上
```

`ray stop` 可能先打印一条关于 `--grace-period` 的 `WARN`，带了 `--force` 时可以忽略。

replay pool（`$OUTPUT_DIR/replay_pools`，约 100 GB）由容器里的 root 写出，不再需要时在 `docker rm` 之前删除：

```bash
docker exec dsrl_model rm -rf /local/$USER/dsrl_cross_node/run/replay_pools
```

## 选择网络

节点往往有好几块网卡。以 L20 和 H20 节点为例：

| 网络 | 地址 | 标称速度 | 实测 TCP 带宽（L20 → H20，4 条流） |
| --- | --- | --- | --- |
| 前端/管理网（`enp3s0f0`） | `10.185.189.x` | L20 侧 1 Gb/s | 0.9 Gb/s |
| `bond0`（2 × 100 Gb/s 聚合） | `192.168.253.x` | 200 Gb/s | 97 Gb/s |

SSH 和 `hostname -I` 给出的通常是前端网的地址，但它可能很慢。用 `ip -br addr` 和 `/sys/class/net/<网卡>/speed` 查看每块网卡的速度，再在两个节点之间实测带宽，然后把较快网络上的地址作为 `NODE_IP` 和 `HEAD_IP`。

## 常见问题

- **资源池一直 pending。** 检查 `ray status`：标签名要和 `MODEL_LABEL`、`ENV_LABEL` 一致，每个节点的 GPU 数要够 4 张。
- **driver 在 `ray.init` 卡住。** 确认 `HEAD_IP` 是 head 的 `NODE_IP`，而不是另一块网卡的地址。
- **环境 worker 找不到 Isaac Sim 或 LIBERO 资源。** 两个节点的挂载路径要相同；Ray 节点要通过 `start_ray_node.sh`（`/isaac-sim/python.sh`）启动，不要直接用 `ray start`。
- **找不到输出目录。** `OUTPUT_DIR` 要在两个节点上都存在。
- **Ray 版本不一致。** 两个节点的镜像 ID 要相同。

不同 GPU 组合的耗时对比见 [流水线性能基准](performance-benchmark.md)。
