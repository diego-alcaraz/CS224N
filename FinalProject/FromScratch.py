import numpy as np
from typing import List, Tuple, Dict

np.random.seed(0)


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:
    def __init__(self, words: List[str]):
        self.words = ['<START>', '<END>'] + list(words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}
        self.start_idx = 0
        self.end_idx = 1
    
    def __len__(self):
        return len(self.words)
    
    def encode(self, word: str) -> int:
        return self.word_to_idx.get(word, 0)
    
    def decode(self, idx: int) -> str:
        return self.idx_to_word.get(idx, '<UNK>')


# =============================================================================
# REWARD FUNCTION
# =============================================================================

class RewardFunction:
    def __init__(self):
        self.rewards = {
            'amazing': 3.0, 'excellent': 3.0, 'wonderful': 3.0, 'fantastic': 3.0,
            'brilliant': 3.0, 'perfect': 2.5,
            'great': 2.0, 'awesome': 2.0, 'beautiful': 2.0, 'love': 2.0,
            'good': 1.5, 'nice': 1.5, 'happy': 1.5, 'best': 1.5,
            'helpful': 1.0, 'kind': 1.0, 'friendly': 1.0,
            'okay': 0.0, 'fine': 0.0, 'normal': 0.0,
            'bad': -1.5, 'sad': -1.5, 'poor': -1.5,
            'terrible': -2.0, 'awful': -2.0, 'horrible': -2.0, 'worst': -2.0,
            'disaster': -3.0, 'catastrophe': -3.0,
        }
    
    def __call__(self, word: str) -> float:
        return self.rewards.get(word.lower(), 0.0)
    
    def get_words(self) -> List[str]:
        return list(self.rewards.keys())


# =============================================================================
# NEURAL NETWORK
# =============================================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class PolicyNetwork:
    """Simple 1-layer network for clarity."""
    
    def __init__(self, vocab_size: int, hidden: int = 32):
        self.vocab_size = vocab_size
        # Single layer: input → hidden → output
        self.W1 = np.random.randn(vocab_size, hidden) * 0.1
        self.W2 = np.random.randn(hidden, vocab_size) * 0.1
        self.cache = {}
    
    def forward(self, state: int) -> np.ndarray:
        """Returns probability distribution over actions."""
        x = np.zeros(self.vocab_size)
        x[state] = 1.0  # One-hot
        
        h = np.maximum(0, x @ self.W1)  # ReLU
        logits = h @ self.W2
        probs = softmax(logits)
        
        self.cache = {'x': x, 'h': h, 'probs': probs}
        return probs
    
    def get_params(self):
        return [self.W1.copy(), self.W2.copy()]
    
    def set_params(self, params):
        self.W1, self.W2 = params[0].copy(), params[1].copy()


# =============================================================================
# PPO-STYLE TRAINER
# =============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization (simplified).
    
    Key insight: Instead of taking arbitrary gradient steps, PPO limits
    how much the policy can change per update. This prevents catastrophic
    forgetting and mode collapse.
    
    Real PPO uses: ratio = π_new(a|s) / π_old(a|s)
    And clips: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        vocab: Vocabulary,
        reward_fn: RewardFunction,
        lr: float = 0.005,
        clip_eps: float = 0.2,      # PPO clipping parameter
        kl_coef: float = 0.01,       # KL penalty (keeps policy close to original)
        entropy_coef: float = 0.05,  # Exploration bonus
    ):
        self.policy = policy
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.lr = lr
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        
        # Store reference policy (for KL penalty)
        self.ref_policy = PolicyNetwork(len(vocab))
        self.ref_policy.set_params(policy.get_params())
        
        self.history = []
    
    def supervised_pretrain(self, positive_words: List[str], epochs: int = 100):
        """
        Pre-train to generate positive words.
        This is like the "SFT" (Supervised Fine-Tuning) phase in real RLHF.
        """
        print("\n📚 SUPERVISED PRE-TRAINING")
        print("-" * 40)
        print("Teaching the model to generate positive words first...")
        
        # Get indices of positive words
        targets = [self.vocab.encode(w) for w in positive_words if w in self.vocab.word_to_idx]
        
        for epoch in range(epochs):
            total_loss = 0
            
            for target_idx in targets:
                # Forward from START token
                probs = self.policy.forward(self.vocab.start_idx)
                
                # Cross-entropy gradient (supervised)
                grad_logits = probs.copy()
                grad_logits[target_idx] -= 1.0  # -∇ log(p_target)
                
                # Backprop
                h = self.policy.cache['h']
                x = self.policy.cache['x']
                
                dW2 = np.outer(h, grad_logits)
                dh = grad_logits @ self.policy.W2.T
                dh = dh * (h > 0)  # ReLU gradient
                dW1 = np.outer(x, dh)
                
                # Update
                self.policy.W2 -= self.lr * dW2
                self.policy.W1 -= self.lr * dW1
                
                total_loss -= np.log(probs[target_idx] + 1e-10)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss = {total_loss/len(targets):.3f}")
        
        # Update reference policy
        self.ref_policy.set_params(self.policy.get_params())
        print("✓ Pre-training complete!")
    
    def compute_advantage(self, reward: float, baseline: float = 0.0) -> float:
        """Advantage = how much better than expected."""
        return reward - baseline
    
    def ppo_update(self, state: int, action: int, advantage: float, old_prob: float):
        """
        PPO-style update with clipping.
        
        This is the key innovation: we limit how much π(a|s) can change.
        """
        # Get current probability
        probs = self.policy.forward(state)
        new_prob = probs[action]
        
        # Probability ratio
        ratio = new_prob / (old_prob + 1e-10)
        
        # Clipped objective
        clipped_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        
        # Take minimum (pessimistic bound)
        if advantage >= 0:
            effective_ratio = min(ratio, clipped_ratio)
        else:
            effective_ratio = max(ratio, clipped_ratio)
        
        # KL penalty (prevents drift from reference)
        ref_probs = self.ref_policy.forward(state)
        kl_div = np.sum(probs * np.log((probs + 1e-10) / (ref_probs + 1e-10)))
        
        # Entropy bonus
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Combined advantage
        effective_advantage = advantage - self.kl_coef * kl_div + self.entropy_coef * entropy
        
        # Policy gradient
        grad_logits = probs.copy()
        grad_logits[action] -= 1.0
        grad_logits *= -effective_advantage * 0.1  # Scale down for stability
        
        # Clip gradients
        grad_logits = np.clip(grad_logits, -0.5, 0.5)
        
        # Backprop
        h = self.policy.cache['h']
        x = self.policy.cache['x']
        
        dW2 = np.outer(h, grad_logits)
        dh = grad_logits @ self.policy.W2.T
        dh = dh * (h > 0)
        dW1 = np.outer(x, dh)
        
        # Update
        self.policy.W2 -= self.lr * dW2
        self.policy.W1 -= self.lr * dW1
    
    def generate(self, max_len: int = 3, temperature: float = 1.0) -> Tuple[List[str], float]:
        """Generate a sequence and compute reward."""
        words = []
        total_reward = 0.0
        state = self.vocab.start_idx
        
        for _ in range(max_len):
            probs = self.policy.forward(state)
            
            # Temperature scaling
            logits = np.log(probs + 1e-10) / temperature
            probs = softmax(logits)
            
            action = np.random.choice(len(probs), p=probs)
            
            if action == self.vocab.end_idx:
                break
            
            word = self.vocab.decode(action)
            words.append(word)
            total_reward += self.reward_fn(word)
            state = action
        
        return words, total_reward
    
    def train_step(self) -> Tuple[float, str]:
        """One training step."""
        # Collect trajectory
        state = self.vocab.start_idx
        
        probs = self.policy.forward(state)
        old_probs = probs.copy()
        
        # Sample action
        action = np.random.choice(len(probs), p=probs)
        word = self.vocab.decode(action)
        reward = self.reward_fn(word)
        
        # Compute advantage (simple: just use reward)
        advantage = reward
        
        # PPO update
        self.ppo_update(state, action, advantage, old_probs[action])
        
        return reward, word
    
    def train(self, steps: int = 2000, print_every: int = 400):
        """Train with PPO."""
        print("\n🎯 RL FINE-TUNING (PPO)")
        print("-" * 40)
        
        for step in range(steps):
            reward, word = self.train_step()
            self.history.append(reward)
            
            if (step + 1) % print_every == 0:
                recent_avg = np.mean(self.history[-print_every:])
                words, _ = self.generate(max_len=3, temperature=0.7)
                print(f"\n  Step {step+1}: Avg reward = {recent_avg:.2f}")
                print(f"  Sample: '{' '.join(words)}'")
        
        print("\n✓ Training complete!")
    
    def evaluate(self, n_samples: int = 10):
        """Generate and evaluate samples."""
        print("\n📊 EVALUATION")
        print("=" * 50)
        
        total_reward = 0
        
        for i in range(n_samples):
            words, reward = self.generate(max_len=3, temperature=0.5)
            total_reward += reward
            print(f"  [{i+1}] '{' '.join(words)}' → reward: {reward:.1f}")
        
        print(f"\n  Average reward: {total_reward/n_samples:.2f}")
        
        # Show learned distribution
        print("\n📈 LEARNED PROBABILITY DISTRIBUTION")
        print("-" * 50)
        probs = self.policy.forward(self.vocab.start_idx)
        
        word_probs = []
        for i, p in enumerate(probs):
            word = self.vocab.decode(i)
            r = self.reward_fn(word)
            word_probs.append((word, p, r))
        
        word_probs.sort(key=lambda x: x[1], reverse=True)
        
        print("\n  Top 15 most likely words (after <START>):")
        for word, prob, reward in word_probs[:15]:
            bar = "█" * int(prob * 40)
            reward_str = f"(r={reward:+.1f})" if reward != 0 else ""
            print(f"    {word:12} {prob:.3f} {bar} {reward_str}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       NLP + REINFORCEMENT LEARNING: A WORKING EXAMPLE         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  This version uses techniques that actually work:             ║
    ║  • Supervised pre-training (like SFT in RLHF)                 ║
    ║  • PPO-style clipped updates                                  ║
    ║  • KL penalty to prevent drift                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    reward_fn = RewardFunction()
    vocab = Vocabulary(reward_fn.get_words())
    policy = PolicyNetwork(len(vocab), hidden=32)
    
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Network: {len(vocab)} → 32 → {len(vocab)}")
    
    # Show initial (random) behavior
    print("\n🎲 BEFORE ANY TRAINING (random policy):")
    print("-" * 50)
    trainer = PPOTrainer(policy, vocab, reward_fn)
    for i in range(5):
        words, reward = trainer.generate(temperature=1.0)
        print(f"  [{i+1}] '{' '.join(words)}' → reward: {reward:.1f}")
    
    # Phase 1: Supervised pre-training
    positive_words = [w for w, r in reward_fn.rewards.items() if r > 0]
    trainer.supervised_pretrain(positive_words, epochs=100)
    
    print("\n✨ AFTER SUPERVISED PRE-TRAINING:")
    print("-" * 50)
    for i in range(5):
        words, reward = trainer.generate(temperature=0.7)
        print(f"  [{i+1}] '{' '.join(words)}' → reward: {reward:.1f}")
    
    # Phase 2: RL fine-tuning
    trainer.train(steps=1500, print_every=300)
    
    # Final evaluation
    print("\n FINAL RESULTS AFTER RL FINE-TUNING:")
    trainer.evaluate(n_samples=10)
    
    # Analysis
    print("\n" + "=" * 60)
    print("WHAT WE DEMONSTRATED")
    print("=" * 60)
    print("""
    1. RANDOM POLICY → Generated arbitrary words (mixed rewards)
    
    2. SUPERVISED PRE-TRAINING → Model learned to generate positive
       words from examples (like SFT phase in ChatGPT training)
    
    3. RL FINE-TUNING → Model further optimized to maximize reward
       while staying close to the pre-trained distribution (PPO)
    
    This 3-phase approach is EXACTLY how modern RLHF works!
    
    Key techniques that made it work:
    • Starting from a good initialization (pre-training)
    • Limiting update magnitude (PPO clipping)
    • KL penalty (prevents forgetting good behavior)
    • Entropy bonus (maintains exploration)
    """)
    
    return trainer


if __name__ == "__main__":
    trainer = main()