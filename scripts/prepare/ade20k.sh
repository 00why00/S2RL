#!/usr/bin/env bash
set -e

# Extract features and build prototypes from the source training data.
# Run from the repository root after exporting the paths in README.md.

mkdir -p "$S2RL_FEATS_ROOT/dinov3_ms" "$S2RL_FEATS_ROOT/dinov3_kmeans"
python utils/calculate_cls_hidden.py
python utils/calculate_cls_hidden_kmeans.py
