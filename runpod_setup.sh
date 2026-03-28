#!/bin/bash
# RunPod setup for Anamnesis 3B scaffold training
# Run this after SSH-ing into your RunPod instance

set -e

echo "=== Anamnesis 3B Scaffold Training Setup ==="

# Clone repo
cd /workspace
git clone https://github.com/Relic-Studios/anamnesis.git
cd anamnesis

# Install dependencies
pip install -e ".[all]" 2>/dev/null || pip install torch transformers datasets safetensors

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"

echo ""
echo "=== Setup complete. Run training with: ==="
echo ""
echo "  python examples/train_scaffold.py \\"
echo "    --model Qwen/Qwen2.5-3B-Instruct \\"
echo "    --steps 50000 \\"
echo "    --batch-size 8 \\"
echo "    --lr 1e-3 \\"
echo "    --test-inner-loop"
echo ""
echo "  Expected: ~3 hours, ~200M tokens, ~\$2 on A100"
echo ""
