#!/bin/bash
# Copyright 2021 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
#   bash train.sh <config> <data_root> <scene_name>
#
# Example:
#   bash train.sh blender /kaggle/input/nerf-dataset/nerf_synthetic chair

CONFIG=$1
DATA_ROOT=$2
SCENE=$3

# Output directory will now include the config name
ROOT_DIR=/kaggle/working/res/jaxnerf/"$CONFIG"

# Determine which dataset folder to use
if [ "$CONFIG" == "llff" ]; then
  DATA_FOLDER="nerf_llff_data"
else
  DATA_FOLDER="nerf_synthetic"
fi

echo "Running training for config: $CONFIG, scene: $SCENE"
echo "Data dir: $DATA_ROOT/$DATA_FOLDER/$SCENE"
echo "Train dir: $ROOT_DIR/$CONFIG/$SCENE"

# Run a single training job for the given scene
python -m jaxnerf.train \
  --data_dir="$DATA_ROOT"/"$DATA_FOLDER"/"$SCENE" \
  --train_dir="$ROOT_DIR"/"$SCENE" \
  --config=configs/"$CONFIG"
