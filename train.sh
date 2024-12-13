#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/training_${timestamp}.log"

# Run training script with nohup
nohup python main.py \
    --batch_size 32 \
    --lr 0.001 \
    --image_norm 1 \
    --label_norm 1 \
    --epochs 300 \
    --model_type "Terratorch" \
    > "${log_file}" 2>&1 &

echo "Training started in background. Check ${log_file} for progress."
