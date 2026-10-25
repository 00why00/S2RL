#!/usr/bin/env bash
set -e

# Run from the repository root after exporting the paths in README.md.
DATASET_NAME="S2RL"  # Released dataset name; change this for your own image-mask pairs.
STUDENT_CONFIG="$PWD/config/mmseg/ade20k.py"

export PORT=29500
cd "$MMSEG_ROOT"

for time in 1 2 3 4 5; do
  for ratio in 0.0025 0.005 0.01; do
    subset="time_${time}_ratio_${ratio}"
    for repeat in 1 2 3 4 5; do
      bash tools/dist_train.sh "$STUDENT_CONFIG" 2 \
        --work-dir "work_dirs_distillation/ade20k/${DATASET_NAME}_segformer/$subset/repeat_$repeat" \
        --cfg-options \
        "train_dataloader.dataset.data_prefix.img_path=$S2RL_DATA_ROOT/ILDistillation/ADE20K/$DATASET_NAME/images/$subset" \
        "train_dataloader.dataset.data_prefix.seg_map_path=$S2RL_DATA_ROOT/ILDistillation/ADE20K/$DATASET_NAME/annotations/$subset"
    done
  done
done
