#!/usr/bin/env bash
set -e

# Run from the repository root after exporting the paths in README.md.
DATASET_NAME="S2RL"  # Released dataset name; change this for your own image-mask pairs.
STUDENT_CONFIG="$PWD/config/mmseg/coco.py"

export PORT=29500
cd "$MMSEG_ROOT"

for time in 1 2 3 4 5; do
  for ratio in 0.000625 0.00125 0.0025; do
    subset="time_${time}_ratio_${ratio}"
    for repeat in 1 2 3; do
      bash tools/dist_train.sh "$STUDENT_CONFIG" 2 \
        --work-dir "work_dirs_distillation/coco/${DATASET_NAME}_segformer/$subset/repeat_$repeat" \
        --cfg-options \
        "train_dataloader.dataset.data_prefix.img_path=$S2RL_DATA_ROOT/ILDistillation/COCO/$DATASET_NAME/images/$subset" \
        "train_dataloader.dataset.data_prefix.seg_map_path=$S2RL_DATA_ROOT/ILDistillation/COCO/$DATASET_NAME/annotations/$subset"
    done
  done
done
