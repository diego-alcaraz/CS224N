"""
Toy RLHF - Understanding the Concepts
======================================
This is a minimal implementation to understand RLHF before running the full version.
No dependencies except numpy.

Run: python toy_rlhf.py
"""

import numpy as np
from typing import List, Tuple, Dict

np.random.seed(42)


# =============================================================================
# VOCABULARY & TOKENIZER
# =============================================================================

class Vocab:
    def __init__(self, words: List[str]):
        self.words = ['<s>', '</s>'] + words
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.i2w = {i: w for i, w in enumerate(self.words)}
    
    def __len__(self): return len(self.words)
    def encode(self, w): return self.w2i.get(w, 0)
    def decode(self, i): return self.i2w.get(i, '<unk>')


# =============================================================================
# REWARD FUNCTION
# =============================================================================

class Reward:
    def __init__(self):
        self.scores = {
            'amazing': 3, 'excellent': 3, 'wonderful': 3,
            'great': 2, 'good': 2, 'nice': 2, 'love': 2,
            'okay': 0, 'fine': 0,
            'bad': -2, 'terrible': -2, 'awful': -2, 'hate': -2,
        }
    
    def __call__(self, word: str) -> float:
        return self.scores.get(word.lower(), 0.0)


# =============================================================================
# POLICY NETWORK
# =============================================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class Policy:
    """Simple neural network policy."""
    
    def __init__(self, vocab_size: int, hidden: int = 32):
        self.W1 = np.random.randn(vocab_size, hidden) * 0.1
        self.W2 = np.random.randn(hidden, vocab_size) * 0.1
        self.cache = {}
    
    def forward(self, state: int, vocab_size: int) -> np.ndarray:
        """Returns probability distribution over next tokens."""
        x = np.zeros(vocab_size)
        x[state] = 1.0
        h = np.maximum(0, x @ self.W1)  # ReLU
        logits = h @ self.W2
        probs = softmax(logits)
        self.cache = {'x': x, 'h': h, 'probs': probs}
        return probs
    
    def update(self, action: int, advantage: float, lr: float = 0.01):
        """Policy gradient update."""
        c = self.cache
        
        # Gradient of log prob
        grad = c['probs'].copy()
        grad[action] -= 1.0
        grad *= -advantage
        grad = np.clip(grad, -0.5, 0.5)
        
        # Backprop
        dW2 = np.outer(c['h'], grad)
        dh = grad @ self.W2.T * (c['h'] > 0)
        dW1 = np.outer(c['x'], dh)
        
        self.W2 -= lr * dW2
        self.W1 -= lr * dW1


# =============================================================================
# PPO TRAINER
# =============================================================================

class PPOTrainer:
    def __init__(self, vocab: Vocab, reward: Reward):
        self.vocab = vocab
        self.reward = reward
        self.policy = Policy(len(vocab))
        self.ref_policy = Policy(len(vocab))
        
        self.kl_coef = 0.1
        self.entropy_coef = 0.1
        self.history = []
    
    def supervised_pretrain(self, positive_words: List[str], epochs: int = 50):
        """
        Pre-train policy to generate positive words.
        This is like SFT (Supervised Fine-Tuning) in real RLHF.
        """
        print("📚 Supervised pre-training...")
        
        targets = [self.vocab.encode(w) for w in positive_words]
        
        for epoch in range(epochs):
            for target in targets:
                probs = self.policy.forward(0, len(self.vocab))
                
                # Cross-entropy gradient
                grad = probs.copy()
                grad[target] -= 1.0
                grad = np.clip(grad, -0.5, 0.5)
                
                # Update
                c = self.policy.cache
                dW2 = np.outer(c['h'], grad)
                dh = grad @ self.policy.W2.T * (c['h'] > 0)
                dW1 = np.outer(c['x'], dh)
                
                self.policy.W2 -= 0.01 * dW2
                self.policy.W1 -= 0.01 * dW1
        
        # Copy to reference
        self.ref_policy.W1 = self.policy.W1.copy()
        self.ref_policy.W2 = self.policy.W2.copy()
        
        print("✓ Pre-training complete!")
    
    def generate(self, max_len: int = 3, temp: float = 0.8) -> Tuple[List[str], float]:
        """Generate a sequence."""
        words = []
        total_reward = 0.0
        state = 0  # <s>
        
        for _ in range(max_len):
            probs = self.policy.forward(state, len(self.vocab))
            
            # Temperature
            logits = np.log(probs + 1e-10) / temp
            probs = softmax(logits)
            
            action = np.random.choice(len(probs), p=probs)
            
            if action == 1:  # </s>
                break
            
            word = self.vocab.decode(action)
            words.append(word)
            total_reward += self.reward(word)
            state = action
        
        return words, total_reward
    
    def train_step(self) -> float:
        """One PPO training step."""
        state = 0
        
        # Get old probs
        probs = self.policy.forward(state, len(self.vocab))
        ref_probs = self.ref_policy.forward(state, len(self.vocab))
        
        # Sample action
        action = np.random.choice(len(probs), p=probs)
        word = self.vocab.decode(action)
        reward = self.reward(word)
        
        # KL penalty
        kl = np.sum(probs * np.log((probs + 1e-10) / (ref_probs + 1e-10)))
        
        # Entropy bonus (encourages exploration)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Combined advantage
        advantage = reward - self.kl_coef * kl + self.entropy_coef * entropy
        
        # Clip advantage for stability
        advantage = np.clip(advantage, -5, 5)
        
        # Update
        self.policy.update(action, advantage, lr=0.005)
        
        self.history.append(reward)
        return reward
    
    def train(self, steps: int = 1000):
        """Training loop."""
        print("=" * 50)
        print("🎯 RL FINE-TUNING (PPO)")
        print("=" * 50)
        
        for step in range(steps):
            reward = self.train_step()
            
            if (step + 1) % 200 == 0:
                avg = np.mean(self.history[-200:])
                words, _ = self.generate()
                print(f"Step {step+1}: avg_reward={avg:.2f}, sample='{' '.join(words)}'")
        
        print("=" * 50)
        print("TRAINING COMPLETE")
        print("=" * 50)
    
    def show_distribution(self):
        """Show learned word probabilities."""
        probs = self.policy.forward(0, len(self.vocab))
        
        word_probs = [(self.vocab.decode(i), probs[i], self.reward(self.vocab.decode(i)))
                      for i in range(len(self.vocab))]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        
        print("\nLearned distribution (top words):")
        for word, prob, r in word_probs[:10]:
            bar = "█" * int(prob * 30)
            r_str = f"(r={r:+})" if r != 0 else ""
            print(f"  {word:12} {prob:.3f} {bar} {r_str}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Vocabulary
    words = [
        'amazing', 'excellent', 'wonderful', 'great', 'good', 'nice', 'love',
        'okay', 'fine', 'normal',
        'bad', 'terrible', 'awful', 'hate',
    ]
    vocab = Vocab(words)
    reward = Reward()
    
    print(f"Vocabulary: {len(vocab)} tokens")
    print(f"Reward words: {list(reward.scores.keys())}")
    
    # Trainer
    trainer = PPOTrainer(vocab, reward)
    
    # Before any training
    print("\n🎲 RANDOM POLICY (before training):")
    for _ in range(5):
        words, r = trainer.generate()
        print(f"  '{' '.join(words)}' (reward={r:.1f})")
    
    # Supervised pre-training (like SFT in real RLHF)
    positive_words = ['amazing', 'excellent', 'wonderful', 'great', 'good', 'nice', 'love']
    trainer.supervised_pretrain(positive_words, epochs=80)
    
    # After SFT
    print("\n✨ AFTER SUPERVISED PRE-TRAINING:")
    for _ in range(5):
        words, r = trainer.generate()
        print(f"  '{' '.join(words)}' (reward={r:.1f})")
    
    # RL fine-tuning
    print()
    trainer.train(steps=1000)
    
    # After RL
    print("\n🏆 AFTER RL FINE-TUNING:")
    for _ in range(10):
        words, r = trainer.generate()
        print(f"  '{' '.join(words)}' (reward={r:.1f})")
    
    trainer.show_distribution()
    
    # Summary
    early = np.mean(trainer.history[:200])
    late = np.mean(trainer.history[-200:])
    print(f"\nRL Improvement: {early:.2f} → {late:.2f}")


if __name__ == "__main__":
    main()
