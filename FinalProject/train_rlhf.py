"""
RLHF Training for your GCP Setup
================================
Adapted for: nvidia-1-vm, conda env cs224n-gpu

Usage:
    sh run.sh train          # Full training
    sh run.sh train_small    # Quick test (100 steps)
    sh run.sh eval           # Evaluate checkpoint
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Training configuration."""
    
    # Model
    model_name: str = "gpt2"  # gpt2, gpt2-medium, facebook/opt-125m
    
    # PPO Hyperparameters
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.1
    entropy_coef: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95
    
    # KL Constraint
    init_kl_coef: float = 0.1
    target_kl: float = 0.05
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    total_steps: int = 2000
    
    # Generation
    max_new_tokens: int = 48
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Logging & Saving
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Paths (relative to project root)
    output_dir: str = "outputs"
    log_dir: str = "runs"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    seed: int = 42


# =============================================================================
# REWARD FUNCTION
# =============================================================================

class SentimentReward:
    """
    Rule-based sentiment reward.
    Replace with learned reward model for production.
    """
    
    def __init__(self):
        self.positive = {
            'amazing', 'excellent', 'wonderful', 'fantastic', 'brilliant',
            'great', 'good', 'nice', 'lovely', 'perfect', 'happy', 'love',
            'beautiful', 'helpful', 'kind', 'awesome', 'best', 'outstanding',
            'incredible', 'superb', 'delightful', 'pleasant', 'enjoy',
            'recommend', 'satisfied', 'impressive', 'exceptional',
        }
        self.negative = {
            'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor', 'sad',
            'hate', 'ugly', 'disaster', 'disgusting', 'disappointing',
            'mediocre', 'boring', 'frustrating', 'annoying', 'painful',
            'waste', 'avoid', 'regret', 'useless', 'broken',
        }
    
    def __call__(self, texts: List[str]) -> torch.Tensor:
        rewards = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & self.positive)
            neg = len(words & self.negative)
            
            reward = 2.0 * pos - 2.0 * neg
            
            # Length penalty
            n_words = len(text.split())
            if n_words > 40:
                reward -= 0.05 * (n_words - 40)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)


# =============================================================================
# PPO TRAINER
# =============================================================================

class PPOTrainer:
    """PPO Trainer with TensorBoard logging."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # Paths
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Load tokenizer
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models
        print(f"Loading model: {config.model_name}")
        dtype = torch.float16 if config.fp16 and self.device.type == "cuda" else torch.float32
        
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=dtype
        ).to(self.device)
        
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=dtype
        ).to(self.device)
        
        # Freeze reference
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        self.ref_policy.eval()
        
        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = config.total_steps * config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        
        # Reward function
        self.reward_fn = SentimentReward()
        
        # KL coefficient
        self.kl_coef = config.init_kl_coef
        
        # Tracking
        self.global_step = 0
        self.best_reward = float('-inf')
    
    @torch.no_grad()
    def generate(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Generate responses for prompts."""
        self.policy.eval()
        
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(self.device)
        
        query_ids = inputs.input_ids
        
        outputs = self.policy.generate(
            query_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response_ids = outputs[:, query_ids.shape[1]:]
        response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        self.policy.train()
        return query_ids, response_ids, response_texts
    
    def compute_log_probs(
        self, model: nn.Module, query_ids: torch.Tensor, response_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for responses."""
        full_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
        
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        query_len = query_ids.shape[1]
        shift_logits = logits[:, query_len-1:-1, :]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        
        mask = (response_ids != self.tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * mask
        
        return token_log_probs, mask
    
    def compute_advantages(
        self, rewards: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        batch_size, seq_len = rewards.shape
        
        # Simple value estimate: cumulative mean reward
        values = torch.zeros_like(rewards)
        for t in range(seq_len):
            values[:, t] = rewards[:, :t+1].sum(dim=-1) / (t + 1)
        
        # GAE
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(seq_len)):
            next_value = values[:, t+1] if t < seq_len - 1 else 0
            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            advantages[:, t] = lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
        
        returns = advantages + values
        
        # Normalize
        adv_mean = (advantages * mask).sum() / mask.sum()
        adv_std = ((advantages - adv_mean).pow(2) * mask).sum() / mask.sum()
        advantages = (advantages - adv_mean) / (adv_std.sqrt() + 1e-8)
        
        return advantages * mask, returns * mask
    
    def ppo_step(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Single PPO update step."""
        
        new_log_probs, _ = self.compute_log_probs(self.policy, query_ids, response_ids)
        
        # Policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
        
        policy_loss = -torch.min(ratio * advantages, clipped * advantages)
        policy_loss = (policy_loss * mask).sum() / mask.sum()
        
        # Entropy bonus
        probs = torch.exp(new_log_probs)
        entropy = -(probs * new_log_probs * mask).sum() / mask.sum()
        
        # Total loss
        loss = policy_loss - self.config.entropy_coef * entropy
        
        # Backward
        loss.backward()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
        }
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """Complete training step."""
        
        # Generate
        query_ids, response_ids, response_texts = self.generate(prompts)
        
        # Rewards
        external_rewards = self.reward_fn(response_texts).to(self.device)
        
        # Log probs
        with torch.no_grad():
            old_log_probs, mask = self.compute_log_probs(self.policy, query_ids, response_ids)
            ref_log_probs, _ = self.compute_log_probs(self.ref_policy, query_ids, response_ids)
        
        # KL divergence
        kl_div = (old_log_probs - ref_log_probs) * mask
        kl_mean = kl_div.sum() / mask.sum()
        
        # Per-token rewards
        seq_len = response_ids.shape[1]
        per_token_rewards = torch.zeros_like(old_log_probs)
        per_token_rewards[:, -1] = external_rewards
        per_token_rewards = per_token_rewards - self.kl_coef * kl_div
        
        # Advantages
        advantages, returns = self.compute_advantages(per_token_rewards, mask)
        
        # PPO epochs
        all_metrics = defaultdict(list)
        
        for _ in range(self.config.ppo_epochs):
            metrics = self.ppo_step(
                query_ids, response_ids,
                old_log_probs.detach(), advantages.detach(), mask
            )
            
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            for k, v in metrics.items():
                all_metrics[k].append(v)
        
        # Aggregate metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_metrics['reward'] = external_rewards.mean().item()
        avg_metrics['kl_div'] = kl_mean.item()
        avg_metrics['kl_coef'] = self.kl_coef
        avg_metrics['lr'] = self.scheduler.get_last_lr()[0]
        avg_metrics['response_len'] = mask.sum(dim=-1).mean().item()
        
        # Adaptive KL
        if avg_metrics['kl_div'] > self.config.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif avg_metrics['kl_div'] < self.config.target_kl / 1.5:
            self.kl_coef /= 1.5
        self.kl_coef = max(0.001, min(10.0, self.kl_coef))
        
        self.global_step += 1
        
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, step)
    
    def evaluate(self, prompts: List[str]) -> float:
        """Evaluate on test prompts."""
        self.policy.eval()
        
        all_rewards = []
        
        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            _, _, responses = self.generate(batch)
            rewards = self.reward_fn(responses)
            all_rewards.extend(rewards.tolist())
        
        avg_reward = np.mean(all_rewards)
        self.writer.add_scalar("eval/reward", avg_reward, self.global_step)
        
        self.policy.train()
        return avg_reward
    
    def save(self, path: str):
        """Save checkpoint."""
        save_path = self.output_dir / path
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.policy.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        torch.save({
            'global_step': self.global_step,
            'kl_coef': self.kl_coef,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_reward': self.best_reward,
        }, save_path / 'training_state.pt')
        
        print(f"Saved checkpoint to {save_path}")
    
    def load(self, path: str):
        """Load checkpoint."""
        load_path = self.output_dir / path
        
        self.policy = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
        
        state = torch.load(load_path / 'training_state.pt', map_location=self.device)
        self.global_step = state['global_step']
        self.kl_coef = state['kl_coef']
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.best_reward = state['best_reward']
        
        print(f"Loaded checkpoint from {load_path}")
    
    def close(self):
        """Cleanup."""
        self.writer.close()


# =============================================================================
# PROMPTS
# =============================================================================

def create_prompts(n: int = 2000) -> List[str]:
    """Create training prompts."""
    templates = [
        "Write a review: The product",
        "Complete: I think this",
        "Continue: The experience was",
        "Describe: This service",
        "Review: The food here",
        "Opinion: Overall, I would say",
        "Thoughts: My honest opinion is",
        "Comment: The quality of",
        "Feedback: After trying this,",
        "Assessment: In my experience,",
        "Verdict: I found that",
        "Summary: To sum up,",
        "Rating: This deserves",
        "Conclusion: Finally,",
    ]
    return [random.choice(templates) for _ in range(n)]


def create_test_prompts() -> List[str]:
    """Create test prompts."""
    return [
        "Write a review: The product",
        "My opinion is that",
        "The experience was",
        "I would describe this as",
        "Overall, this is",
    ]


# =============================================================================
# MAIN
# =============================================================================

def train(config: Config):
    """Main training loop."""
    
    print("=" * 60)
    print("🚀 RLHF Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Steps: {config.total_steps}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 60)
    
    # Initialize
    trainer = PPOTrainer(config)
    
    # Prompts
    train_prompts = create_prompts(5000)
    test_prompts = create_test_prompts()
    
    # Training loop
    pbar = tqdm(range(config.total_steps), desc="Training")
    
    for step in pbar:
        # Sample batch
        batch = random.sample(train_prompts, config.batch_size)
        
        # Train
        metrics = trainer.train_step(batch)
        
        # Progress bar
        pbar.set_postfix({
            'r': f"{metrics['reward']:.2f}",
            'kl': f"{metrics['kl_div']:.3f}",
        })
        
        # Log
        if step % config.log_interval == 0:
            trainer.log_metrics(metrics, step)
        
        # Eval
        if step % config.eval_interval == 0:
            avg_reward = trainer.evaluate(test_prompts)
            print(f"\n📊 Step {step}: eval reward = {avg_reward:.2f}")
            
            # Sample generation
            _, _, responses = trainer.generate(test_prompts[:2])
            for p, r in zip(test_prompts[:2], responses):
                print(f"  Prompt: '{p}'")
                print(f"  Response: '{r[:80]}...'\n")
            
            # Best model
            if avg_reward > trainer.best_reward:
                trainer.best_reward = avg_reward
                trainer.save("best")
        
        # Checkpoint
        if step > 0 and step % config.save_interval == 0:
            trainer.save(f"step_{step}")
    
    # Final save
    trainer.save("final")
    trainer.close()
    
    print("\n✅ Training complete!")


def evaluate(config: Config, checkpoint: str = "best"):
    """Evaluate a checkpoint."""
    
    print("=" * 60)
    print("📊 RLHF Evaluation")
    print("=" * 60)
    
    trainer = PPOTrainer(config)
    trainer.load(checkpoint)
    
    test_prompts = [
        "Write a review: The product",
        "My opinion is that",
        "The experience was",
        "I would describe this as",
        "Overall, this is",
        "The quality seems",
        "I think this",
        "This made me feel",
    ]
    
    print("\n🎯 Generated responses:\n")
    
    for prompt in test_prompts:
        _, _, responses = trainer.generate([prompt])
        reward = trainer.reward_fn(responses).item()
        print(f"Prompt: '{prompt}'")
        print(f"Response: '{responses[0]}'")
        print(f"Reward: {reward:.2f}\n")
    
    trainer.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RLHF Training")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", type=str, default="gpt2")
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--output", type=str, default="outputs")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, default="best")
    eval_parser.add_argument("--model", type=str, default="gpt2")
    eval_parser.add_argument("--output", type=str, default="outputs")
    
    args = parser.parse_args()
    
    if args.command == "train":
        config = Config(
            model_name=args.model,
            total_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output,
        )
        train(config)
    
    elif args.command == "eval":
        config = Config(
            model_name=args.model,
            output_dir=args.output,
        )
        evaluate(config, args.checkpoint)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
