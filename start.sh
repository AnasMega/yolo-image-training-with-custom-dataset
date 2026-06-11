#!/bin/bash

set -e

MODEL_PATH="runs/detect/train3/weights/best.pt"

echo "Checking model..."

if [ ! -f "$MODEL_PATH" ]; then
    echo "🚀 Model not found → Starting training"
    python train.py
else
    echo "✅ Model exists → skipping training"
fi

echo "🚀 Starting Streamlit App..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0