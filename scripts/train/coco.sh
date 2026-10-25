#!/usr/bin/env bash
set -e

# Run from the repository root after exporting the paths in README.md.
RUN_NAME=""  # Fill in a name for your training run.
export TOKENIZERS_PARALLELISM=false

# Train one policy on the union of the three Uniform budgets.
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision fp16 --main_process_port 29500 \
  train_sd3_set.py --config config/grpo_sd3.py:coco \
  --config.run_name="$RUN_NAME" --config.save_dir="outputs/$RUN_NAME"
