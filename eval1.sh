#!/bin/bash
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# This modified version is adapted for single-scene evaluation in Kaggle.

# -------------------------------
# Usage:
#   bash eval_single.sh <CONFIG> <DATA_ROOT> <SCENE>
# Example:
#   bash eval_single.sh blender /kaggle/input/nerf-dataset/nerf_synthetic mic
# -------------------------------

CONFIG=$1
DATA_ROOT=$2
SCENE=$3

# Output directory includes the config name
ROOT_DIR=/kaggle/working/res/jaxnerf/"$CONFIG"

# Determine dataset folder based on config
if [ "$CONFIG" == "llff" ]; then
  DATA_FOLDER="nerf_llff_data"
else
  DATA_FOLDER="nerf_synthetic"
fi

echo "==============================================="
echo "Running evaluation for CONFIG: $CONFIG"
echo "Scene: $SCENE"
echo "Data root: $DATA_ROOT"
echo "Results will be saved to: $ROOT_DIR/$SCENE"
echo "==============================================="

# Run evaluation for the specified scene only
python -m jaxnerf.eval \
  --data_dir="$DATA_ROOT"/"$DATA_FOLDER"/"$SCENE" \
  --train_dir="$ROOT_DIR"/"$SCENE" \
  --chunk=4096 \
  --config=configs/"$CONFIG"

# Collect PSNR result for the single scene
PSNR_FILE="$ROOT_DIR"/"$SCENE"/test_preds/psnr.txt
SUMMARY_FILE="$ROOT_DIR"/psnr_summary.txt

if [ -f "$PSNR_FILE" ]; then
  echo "${SCENE}: $(cat "$PSNR_FILE")" >> "$SUMMARY_FILE"
  echo "PSNR for $SCENE written to $SUMMARY_FILE"
else
  echo "Warning: PSNR file not found for scene $SCENE"
fi
