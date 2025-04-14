#!/bin/bash

# List of base models
base_models=(
  "pretrained_model_weights_240_classes"
  "pretrained_model_weights_200_classes"
  "pretrained_model_weights_160_classes"
  "pretrained_model_weights_120_classes"
  "pretrained_model_weights_80_classes"
  "pretrained_model_weights_80_with_overlapping_classes"
  "pretrained_model_weights_80_with_movement_overlapping_classes"
)

# List of instances to test
instances_list=(6 4 3 2)

# Number of runs
n_runs=1

# Loop through each base model
for basemodel in "${base_models[@]}"; do
  echo "🔹 Processing base model: $basemodel"
  
  # Extract numeric basesize from model name using regex
  basesize=$(echo "$basemodel" | grep -oE '[0-9]+' | head -1)

  if [[ -z "$basesize" ]]; then
    echo "❌ Could not extract basesize from model: $basemodel"
    continue
  fi

  for instances in "${instances_list[@]}"; do
    for ((runid=0; runid<n_runs; runid++)); do
      echo "  ➤ Running: runid=$runid, instances=$instances, basesize=$basesize"
      python finetunning.py \
        --runid "$runid" \
        --instances "$instances" \
        --basesize "$basesize" \
        --basemodel "$basemodel"
    done
  done
done
