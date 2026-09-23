#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${TRAIN_DATA:?Set TRAIN_DATA to an image directory or path-list text file}"
: "${VAL_DATA:?Set VAL_DATA to an image directory or path-list text file}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
launch=()
if (( ${#gpu_list[@]} > 1 )); then
    launch+=(--multi_gpu)
fi
model=()
if [[ -n "${SD_PATH:-}" ]]; then
    model+=(--sd_path "$SD_PATH")
fi
exec accelerate launch "${launch[@]}" \
    --num_processes "${#gpu_list[@]}" --gpu_ids "$GPU_IDS" \
    --main_process_port "${PORT:-29303}" src/train_sr.py \
    --train_data "$TRAIN_DATA" --val_data "$VAL_DATA" \
    --de_net_path "${DE_NET:-checkpoints/de_net.pth}" \
    --pretrained_sr_path "${SR_INIT:-checkpoints/s3diff.pkl}" \
    --pretrained_sim_path "${SIM_CHECKPOINT:-outputs/simulator/checkpoints/simulator_final.pkl}" \
    --output_dir "${SR_OUTPUT:-outputs/sr}" \
    "${model[@]}" "$@"
