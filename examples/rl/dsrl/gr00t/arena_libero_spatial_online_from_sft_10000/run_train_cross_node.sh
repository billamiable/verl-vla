#!/usr/bin/env bash
#
# Cross-node DSRL: run_train.sh with the model pool on the Ray head node and the
# simulation pool on the worker node. Start both Ray nodes with
# start_ray_node.sh first, then run this inside the head node's container; see
# docs/reinforcement-learning/dsrl/gr00t/cross-node.md.
#
#   HEAD_IP=<model node IP> TASK_ID=7 OUTPUT_DIR=/local/scratch/run \
#     bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train_cross_node.sh
#
# OUTPUT_DIR must exist on both nodes: the model workers write the replay pool
# and checkpoints on the head node, the env workers write videos on the worker.
# Extra arguments are forwarded to Hydra.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HEAD_IP="${HEAD_IP:?set HEAD_IP to the NODE_IP of the head node}"
RAY_PORT="${RAY_PORT:-26379}"
MODEL_LABEL="${MODEL_LABEL:-train_rollout}"
ENV_LABEL="${ENV_LABEL:-sim}"

export RAY_ADDRESS="${RAY_ADDRESS:-$HEAD_IP:$RAY_PORT}"

# The driver advertises HEAD_IP so its traffic uses the same network as the
# Ray nodes; the controller stays on the head node next to the model pool.
exec bash "$SCRIPT_DIR/run_train.sh" \
  "cluster.resource.controller_label=$MODEL_LABEL" \
  "cluster.resource.model.resource_label=$MODEL_LABEL" \
  "cluster.resource.env.resource_label=$ENV_LABEL" \
  "+ray_kwargs.ray_init._node_ip_address=$HEAD_IP" \
  "$@"
