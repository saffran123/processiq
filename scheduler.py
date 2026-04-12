"""
scheduler.py - Q-Learning Reinforcement-Learning Scheduler
-----------------------------------------------------------
Implements a Q-learning agent that decides whether to INCREASE,
DECREASE, or KEEP the priority (nice value) of a process based
on its CPU and memory usage.

Linux 'nice' values range from -20 (highest priority) to +19 (lowest).
We only touch non-system processes to stay safe.

Authors: Student 1 & Student 2
"""

import os
import psutil
import random
import json
import time
import pickle
from monitor import get_top_processes, get_system_stats

# ─────────────────────────────────────────────────────────────────────────────
# Q-TABLE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Actions the agent can take on a process's nice value
ACTIONS = {
    0: "increase_priority",   # nice -= 5  (process runs faster)
    1: "decrease_priority",   # nice += 5  (process runs slower)
    2: "keep_priority"        # nice unchanged
}

# Hyper-parameters (beginner-friendly values)
LEARNING_RATE   = 0.1    # How fast the agent learns (alpha)
DISCOUNT_FACTOR = 0.9    # How much future rewards matter (gamma)
EPSILON         = 0.2    # Exploration rate: 20% random, 80% greedy
EPSILON_DECAY   = 0.995  # Slowly reduce exploration over time
MIN_EPSILON     = 0.05   # Never go below 5% exploration

Q_TABLE_FILE = "q_table.pkl"   # Saved Q-table so learning persists


# ─────────────────────────────────────────────────────────────────────────────
# STATE DISCRETIZATION
# ─────────────────────────────────────────────────────────────────────────────

def discretize_cpu(cpu_pct):
    """
    Bin CPU % into 3 buckets: low / medium / high.
    Simplifies the state space so Q-table stays small.
    """
    if cpu_pct < 30:
        return "low"
    elif cpu_pct < 70:
        return "medium"
    else:
        return "high"


def discretize_mem(mem_pct):
    """Bin memory % into 3 buckets."""
    if mem_pct < 20:
        return "low"
    elif mem_pct < 50:
        return "medium"
    else:
        return "high"


def get_state(cpu_pct, mem_pct):
    """
    Combine CPU and memory buckets into a single state string.
    Example: "high_medium" = high CPU, medium memory.
    """
    return f"{discretize_cpu(cpu_pct)}_{discretize_mem(mem_pct)}"


# ─────────────────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class QLearningScheduler:
    """
    A simple Q-learning agent.
    State  = (cpu_bucket, mem_bucket)
    Action = increase / decrease / keep priority
    Reward = computed after each action based on system improvement
    """

    def __init__(self):
        self.q_table = self._load_q_table()
        self.epsilon = EPSILON
        self.action_log = []   # Stores recent decisions for the dashboard

    # ── Q-table helpers ─────────────────────────────────────────────────────

    def _load_q_table(self):
        """Load Q-table from disk if it exists, else start fresh."""
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, 'rb') as f:
                print("[Scheduler] Loaded existing Q-table from disk.")
                return pickle.load(f)
        return {}   # Empty dict: Q(s,a) defaults to 0

    def _save_q_table(self):
        """Persist Q-table to disk so learning survives restarts."""
        with open(Q_TABLE_FILE, 'wb') as f:
            pickle.dump(self.q_table, f)

    def _get_q(self, state, action):
        """Return Q(state, action), defaulting to 0.0."""
        return self.q_table.get((state, action), 0.0)

    def _set_q(self, state, action, value):
        """Set Q(state, action) = value."""
        self.q_table[(state, action)] = value

    # ── Action selection (epsilon-greedy) ────────────────────────────────────

    def choose_action(self, state):
        """
        Epsilon-greedy policy:
        - With probability epsilon → pick a random action (explore)
        - Otherwise             → pick the action with highest Q-value (exploit)
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)   # random action
        # Greedy: pick action with highest Q-value for this state
        q_values = [self._get_q(state, a) for a in ACTIONS]
        return q_values.index(max(q_values))

    # ── Reward function ──────────────────────────────────────────────────────

    def compute_reward(self, cpu_before, cpu_after, mem_before, mem_after):
        """
        Reward function — the heart of reinforcement learning.

        We reward the agent for REDUCING cpu+memory usage.
        Positive reward = good action, negative = bad action.

        Formula:
            reward = (cpu_drop * 0.6) + (mem_drop * 0.4)
        CPU gets higher weight because it's more time-sensitive.
        """
        cpu_drop = cpu_before - cpu_after   # positive if CPU went down
        mem_drop = mem_before - mem_after   # positive if memory went down
        reward = (cpu_drop * 0.6) + (mem_drop * 0.4)
        return round(reward, 3)

    # ── Q-table update (Bellman equation) ────────────────────────────────────

    def update_q_table(self, state, action, reward, next_state):
        """
        Q-learning update rule (Bellman equation):
            Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        α = learning rate, γ = discount factor.
        This is called after observing the result of an action.
        """
        current_q = self._get_q(state, action)
        max_next_q = max([self._get_q(next_state, a) for a in ACTIONS])
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self._set_q(state, action, round(new_q, 4))

    # ── Priority control ─────────────────────────────────────────────────────

    def apply_action(self, pid, action_id, current_nice):
        """
        Actually change the process nice value on Linux.
        Returns the new nice value (or current if failed).
        """
        try:
            proc = psutil.Process(pid)
            if action_id == 0:   # increase priority → lower nice
                new_nice = max(-10, current_nice - 5)   # clamp at -10 for safety
            elif action_id == 1: # decrease priority → raise nice
                new_nice = min(15, current_nice + 5)    # clamp at +15
            else:                # keep
                new_nice = current_nice

            if new_nice != current_nice:
                proc.nice(new_nice)   # Linux: requires sudo for negative values
            return new_nice
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            # AccessDenied is common for system processes — just skip them
            return current_nice

    # ── Main training step ───────────────────────────────────────────────────

    def train_step(self):
        """
        One full training iteration:
        1. Observe top processes
        2. For each, choose & apply an action
        3. Measure reward
        4. Update Q-table
        Returns a list of decisions made (for dashboard display).
        """
        decisions = []
        top_procs = get_top_processes(n=5)

        for proc in top_procs:
            pid = proc['pid']
            cpu_before = proc['cpu_percent']
            mem_before = proc['memory_percent']
            nice_before = proc['nice']

            state = get_state(cpu_before, mem_before)
            action = self.choose_action(state)

            # Apply the action (change nice value)
            new_nice = self.apply_action(pid, action, nice_before)

            # Brief pause to let the OS react
            time.sleep(0.3)

            # Observe new stats after action
            updated = self._sample_process(pid)
            cpu_after = updated['cpu_percent'] if updated else cpu_before
            mem_after = updated['memory_percent'] if updated else mem_before

            # Compute reward and update Q-table
            reward = self.compute_reward(cpu_before, cpu_after, mem_before, mem_after)
            next_state = get_state(cpu_after, mem_after)
            self.update_q_table(state, action, reward, next_state)

            decision = {
                'pid': pid,
                'name': proc['name'],
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'mem_before': mem_before,
                'mem_after': mem_after,
                'nice_before': nice_before,
                'nice_after': new_nice,
                'action': ACTIONS[action],
                'reward': reward,
                'state': state
            }
            decisions.append(decision)

        # Decay epsilon (less exploration over time)
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        # Save Q-table after every step
        self._save_q_table()

        self.action_log = decisions   # expose for dashboard
        return decisions

    def _sample_process(self, pid):
        """Re-sample a single process after applying action."""
        try:
            p = psutil.Process(pid)
            return {
                'cpu_percent': p.cpu_percent(interval=0.2),
                'memory_percent': p.memory_percent()
            }
        except Exception:
            return None

    def get_q_table_summary(self):
        """Return Q-table as a list of dicts for display in dashboard."""
        rows = []
        for (state, action), q_val in self.q_table.items():
            rows.append({
                'state': state,
                'action': ACTIONS[action],
                'q_value': round(q_val, 4)
            })
        return sorted(rows, key=lambda r: r['q_value'], reverse=True)[:20]


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TRAINING LOOP (run this file directly to pre-train)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Q-Learning Scheduler training loop...")
    print("Press Ctrl+C to stop.\n")

    agent = QLearningScheduler()
    episode = 0

    try:
        while True:
            episode += 1
            print(f"\n── Episode {episode}  (ε={agent.epsilon:.3f}) ──")
            decisions = agent.train_step()
            for d in decisions:
                print(f"  [{d['action']:20s}] PID {d['pid']:6d} {d['name'][:20]:20s} "
                      f"CPU {d['cpu_before']:5.1f}%→{d['cpu_after']:5.1f}%  "
                      f"Reward: {d['reward']:+.3f}  Nice: {d['nice_before']}→{d['nice_after']}")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n[Scheduler] Training stopped. Q-table saved.")
