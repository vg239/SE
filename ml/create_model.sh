#!/bin/bash

# Script to create ML model config file before Docker build
# This ensures the file exists if you want to mount it as a volume

echo "Training ML model and creating config file..."
cd "$(dirname "$0")"
python train_model.py

if [ -f "ml_model_config.json" ]; then
    echo "✅ Model config created successfully at: ml/ml_model_config.json"
    echo "You can now use volume mount in docker-compose.yml if needed"
else
    echo "❌ Failed to create model config file"
    exit 1
fi

