# RLHF Training

Reinforcement Learning from Human Feedback for language models.

## Your Workflow

```bash
# LOCAL (Mac, VS Code)
# 1) Edit code in VS Code
# 2) Commit and push
git add .
git commit -m "update"
git push origin main

# START VM
gcloud compute instances start nvidia-1-vm \
  --zone asia-southeast1-a \
  --project onyx-zodiac-484309-r2

# SSH INTO VM
gcloud compute ssh nvidia-1-vm \
  --zone asia-southeast1-a \
  --project onyx-zodiac-484309-r2

# ON THE VM
cd ~/student
git pull origin main
export CONDA_NO_PLUGINS=true
conda activate cs224n-gpu

# TRAINING (in tmux)
tmux new -s train
sh run.sh train
# Ctrl+B, then D to detach

# TENSORBOARD (on VM)
tensorboard --logdir runs --port 6007 --bind_all

# TENSORBOARD (on Mac, new terminal)
gcloud compute ssh nvidia-1-vm \
  --zone asia-southeast1-a \
  --project onyx-zodiac-484309-r2 \
  --ssh-flag="-L 6007:localhost:6007"
# Open: http://localhost:6007
```

## Commands

```bash
sh run.sh train          # Full training (2000 steps, ~30 min)
sh run.sh train_small    # Quick test (100 steps, ~2 min)
sh run.sh train_medium   # Medium (1000 steps, ~15 min)
sh run.sh train_large    # Large with gpt2-medium (5000 steps)
sh run.sh eval           # Evaluate best checkpoint
sh run.sh tensorboard    # Start TensorBoard
sh run.sh clean          # Remove outputs
```

## Custom Settings

```bash
# Use different model
MODEL=gpt2-medium sh run.sh train

# Adjust batch size (reduce if OOM)
BATCH_SIZE=4 sh run.sh train

# Both
MODEL=gpt2-medium BATCH_SIZE=4 sh run.sh train
```

## Project Structure

```
├── train_rlhf.py      # Main training script
├── run.sh             # Training launcher
├── requirements.txt   # Dependencies
├── outputs/           # Checkpoints (created during training)
│   ├── best/         # Best checkpoint
│   ├── final/        # Final checkpoint
│   └── step_*/       # Intermediate checkpoints
└── runs/              # TensorBoard logs
```

## What This Does

1. **Loads GPT-2** (or other HuggingFace model)
2. **Generates responses** to prompts like "Write a review: The product"
3. **Scores responses** with sentiment reward (positive words = good)
4. **Updates model** with PPO to maximize reward
5. **Logs to TensorBoard** for monitoring

## Key Metrics (TensorBoard)

| Metric | Meaning | Good Sign |
|--------|---------|-----------|
| `train/reward` | Average sentiment score | Increasing |
| `train/kl_div` | Distance from original model | 0.01-0.1 |
| `train/loss` | PPO loss | Decreasing |
| `eval/reward` | Test set reward | Increasing |

## Tips

1. **Start small**: `sh run.sh train_small` first to verify everything works
2. **Watch KL**: If `kl_div` > 0.1, model is drifting too far
3. **Check samples**: Look at generated text in console output
4. **GPU memory**: Reduce `BATCH_SIZE` if you get OOM errors

## Extending

To train on different tasks:

1. Modify `SentimentReward` class in `train_rlhf.py`
2. Change prompts in `create_prompts()` function
3. Adjust hyperparameters in `Config` class
