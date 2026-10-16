#!/usr/bin/env bash
#
# Start one Ray node for a cross-node DSRL run: the model pool on one node and
# the simulation pool on another. Run inside the Arena GR00T container on each
# node, the head (model node) first; see
# docs/reinforcement-learning/dsrl/gr00t/cross-node.md.
#
#   ROLE=head   NODE_IP=<model node IP> \
#     bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/start_ray_node.sh
#   ROLE=worker NODE_IP=<env node IP> HEAD_IP=<model node IP> \
#     bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/start_ray_node.sh
#
# Ray uses the GPUs in CUDA_VISIBLE_DEVICES. NODE_IP selects the network Ray
# traffic uses between the nodes, so pick the fastest interface both nodes share.

set -euo pipefail

ROLE="${ROLE:?set ROLE=head or ROLE=worker}"
NODE_IP="${NODE_IP:?set NODE_IP to the address of this node on the inter-node network}"
RAY_PORT="${RAY_PORT:-26379}"
NUM_GPUS="${NUM_GPUS:-4}"

# Ray resource labels; run_train_cross_node.sh places each pool by these names.
MODEL_LABEL="${MODEL_LABEL:-train_rollout}"
ENV_LABEL="${ENV_LABEL:-sim}"

# Ray workers inherit the environment of the node's raylet. Start Ray through
# /isaac-sim/python.sh so env workers get the variables Isaac Sim needs
# (EXP_PATH and friends), as run_train.sh does for a single node.
RAY_CLI=(/isaac-sim/python.sh -m ray.scripts.scripts)

case "$ROLE" in
  head)
    exec "${RAY_CLI[@]}" start --head \
      --node-ip-address="$NODE_IP" --port="$RAY_PORT" --num-gpus="$NUM_GPUS" \
      --resources="{\"$MODEL_LABEL\": 1}" \
      --include-dashboard=false --disable-usage-stats
    ;;
  worker)
    HEAD_IP="${HEAD_IP:?set HEAD_IP to the NODE_IP of the head node}"
    exec "${RAY_CLI[@]}" start \
      --address="$HEAD_IP:$RAY_PORT" \
      --node-ip-address="$NODE_IP" --num-gpus="$NUM_GPUS" \
      --resources="{\"$ENV_LABEL\": 1}" \
      --disable-usage-stats
    ;;
  *)
    echo "ROLE must be head or worker, got: $ROLE" >&2
    exit 2
    ;;
esac
