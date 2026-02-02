#!/bin/bash
#==============================================================================
# RLHF Training Runner
#==============================================================================
# Usage:
#   sh run.sh train          # Full training (2000 steps)
#   sh run.sh train_small    # Quick test (100 steps)  
#   sh run.sh train_medium   # Medium training (1000 steps)
#   sh run.sh eval           # Evaluate best checkpoint
#   sh run.sh eval_final     # Evaluate final checkpoint
#==============================================================================

set -e

# Configuration
MODEL=${MODEL:-"gpt2"}                    # gpt2, gpt2-medium, facebook/opt-125m
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}

# Check GPU
echo "=============================================="
echo "  RLHF Training"
echo "=============================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=============================================="

case "$1" in
    train)
        echo "Starting full training (2000 steps)..."
        python train_rlhf.py train \
            --model $MODEL \
            --steps 2000 \
            --batch-size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --output $OUTPUT_DIR
        ;;
    
    train_small)
        echo "Starting small training (100 steps)..."
        python train_rlhf.py train \
            --model $MODEL \
            --steps 100 \
            --batch-size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --output $OUTPUT_DIR
        ;;
    
    train_medium)
        echo "Starting medium training (1000 steps)..."
        python train_rlhf.py train \
            --model $MODEL \
            --steps 1000 \
            --batch-size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --output $OUTPUT_DIR
        ;;
    
    train_large)
        echo "Starting large training (5000 steps) with gpt2-medium..."
        python train_rlhf.py train \
            --model gpt2-medium \
            --steps 5000 \
            --batch-size 4 \
            --lr 5e-6 \
            --output $OUTPUT_DIR
        ;;
    
    eval)
        echo "Evaluating best checkpoint..."
        python train_rlhf.py eval \
            --checkpoint best \
            --model $MODEL \
            --output $OUTPUT_DIR
        ;;
    
    eval_final)
        echo "Evaluating final checkpoint..."
        python train_rlhf.py eval \
            --checkpoint final \
            --model $MODEL \
            --output $OUTPUT_DIR
        ;;
    
    tensorboard)
        echo "Starting TensorBoard on port 6007..."
        tensorboard --logdir runs --port 6007 --bind_all
        ;;
    
    clean)
        echo "Cleaning outputs..."
        rm -rf outputs runs
        echo "Done."
        ;;
    
    *)
        echo "Usage: sh run.sh {train|train_small|train_medium|train_large|eval|eval_final|tensorboard|clean}"
        echo ""
        echo "Commands:"
        echo "  train         Full training (2000 steps)"
        echo "  train_small   Quick test (100 steps)"
        echo "  train_medium  Medium training (1000 steps)"
        echo "  train_large   Large training with gpt2-medium (5000 steps)"
        echo "  eval          Evaluate best checkpoint"
        echo "  eval_final    Evaluate final checkpoint"
        echo "  tensorboard   Start TensorBoard server"
        echo "  clean         Remove outputs and runs directories"
        echo ""
        echo "Environment variables:"
        echo "  MODEL=gpt2           Model name (gpt2, gpt2-medium, etc.)"
        echo "  BATCH_SIZE=8         Batch size"
        echo "  LEARNING_RATE=1e-5   Learning rate"
        echo ""
        echo "Example:"
        echo "  MODEL=gpt2-medium BATCH_SIZE=4 sh run.sh train"
        exit 1
        ;;
esac
