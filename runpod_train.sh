#!/bin/bash
# Anamnesis 7B Scaffold Training on RunPod
# GPU: 1x A100 80GB (~$0.80/hr, ~3 hours = ~$2.50)
#
# SSH into your RunPod instance and run:
#   bash runpod_train.sh

set -e

echo "============================================================"
echo "  Anamnesis 7B Scaffold Training"
echo "  Teaching the memory how to learn"
echo "============================================================"

# Setup
cd /workspace
if [ ! -d "anamnesis" ]; then
    git clone https://github.com/Relic-Studios/anamnesis.git
fi
cd anamnesis
git pull

# Install deps
pip install -q torch transformers datasets safetensors

# Verify GPU
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# Train
python examples/train_scaffold.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steps 50000 \
    --batch-size 8 \
    --seq-len 512 \
    --lr 1e-3 \
    --warmup 500 \
    --log-every 100 \
    --save-every 5000 \
    --eval-every 2500 \
    --output-dir /workspace/scaffold_7b \
    --test-inner-loop

echo ""
echo "============================================================"
echo "  Training complete. Results in /workspace/scaffold_7b/"
echo "  Download final.pt and results.json"
echo "============================================================"
