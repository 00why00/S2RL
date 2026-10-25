#!/usr/bin/env bash
set -e

# Run from the repository root after exporting the paths in README.md.
POLICY_DIR=""  # Fill in the directory containing the adapter weights and config.
DATASET_NAME="S2RL_generated"  # Change this name to save a separate generated dataset.

for time in 1 2 3 4 5; do
  for ratio in 0.000625 0.00125 0.0025; do
    accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
      --mixed_precision fp16 --main_process_port 29500 \
      utils/generate_sd3_text_active_coco.py --dataset COCO \
      --finetune_model_path "$POLICY_DIR" \
      --save_path "$DATASET_NAME" \
      --sample_type uniform --selected_images "time_${time}_ratio_${ratio}" \
      --seed 42
  done
done
