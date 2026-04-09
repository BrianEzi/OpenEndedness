# Emergent Communication in Bottleneck Environments

This project explores the development of **emergent communication protocols** between autonomous agents using Multi-Agent Reinforcement Learning (MARL). Specifically, it focuses on a "Seer-Doer" architecture where a global observer (the Seer) must guide physical actors (the Doers) through a complex coordination task.

---

## 🎯 Project Objective

The primary goal is to solve a **coordination deadlock** in a grid-world environment. Two agents (Doers) must swap sides through a narrow, single-tile-wide corridor. This requires precise timing and spatial awareness.

### The Perception Challenge
To force the emergence of communication, the project utilizes a **curriculum** that gradually reduces the Doers' perception:
1.  **Level 1**: Doers have local vision and know their coordinates.
2.  **Level 2**: Doers have **no vision** but know their coordinates.
3.  **Level 3**: Doers have **no vision and no coordinates**.

At Level 3, the Doers are effectively "blind" and depends entirely on the **Seer's messages** to navigate and complete tasks.

---

## 🏗️ System Architecture

The system is built on a "Prefrontal Cortex vs. Motor Cortex" analogy.

### 1. The Seer (Prefrontal Cortex)
- **Role**: Global observer and navigator.
- **Input**: 
    - Full global map (grid).
    - Symbolic state (agent positions, goals, time).
    - Target object images.
- **Output**: 
    - A **discrete compositional message** $m_t$.
    - Navigation logits (used only during the pre-training "embodied" phase).
- **Architecture**: 
    - CNN for spatial maps.
    - MLP for symbolic data.
    - **LSTM** to maintain temporal context (crucial for sequencing commands).
    - **FSQ Head**: Quantizes continuous thoughts into discrete symbols.

### 2. The Doer (Motor Cortex)
- **Role**: Physical execution.
- **Input**: 
    - EGOCENTRIC 3x3 local view (often zeroed).
    - Proprioception (agent ID, carrying status).
    - **The Seer's message** $m_t$.
    - Menu images (for object selection).
- **Output**: Physical actions (Move N/S/E/W, Stay, Pick 0-3).
- **Architecture**: 
    - CNN for local views.
    - MLP for proprioception and message embedding.
    - **LSTM** to track execution state across time.

---

## 📡 Communication Protocol: Finite Scalar Quantization (FSQ)

Communication is discrete to simulate symbolic language. This is implemented via **FSQ** rather than Gumbel-Softmax or Vector Quantization (VQ).

> [!IMPORTANT]
> **Why FSQ?**
> FSQ avoids the "index collapse" common in VQ-VAE and doesn't require a commitment loss. It projects continuous vectors into a fixed coordinate grid and uses a **Straight-Through Estimator (STE)** to allow gradients to flow back to the Seer.

### The Quantization Flow:
1.  **Bounding**: Continuous vector $z$ is passed through `tanh`.
2.  **Scaling**: Mapped to a grid $[0, L-1]$.
3.  **Rounding**: Snapped to the nearest integer (quantized).
4.  **STE**: During backprop, the gradient of the rounding operation is treated as 1, allowing the Seer to learn which "thoughts" lead to successful Doer actions.

---

## 🎮 The Environment: `TwoDoerBottleneckEnv`

The environment is a custom JAX-based gridworld (`envs/two_doer_grid.py`).

### Two-Phase Task:
1.  **Navigation Phase**: Agents must move from their starting rooms, cross a central bottleneck corridor, and arrive at goal tiles.
2.  **Selection Phase**: Once at the goal, agents are presented with a "menu" of 4 objects. They must pick the specific object indicated by the environment's `target_items` (which only the Seer can see).

### Reward Structure (Team-Oriented):
- **Team Reward**: +1.0 for both agents successfully swapping and selecting correctly.
- **Progress Reward**: Based on Manhattan distance reduction to encourage movement.
- **Arrival Bonus**: +0.5 when both agents reach their goals.
- **Penalties**: Wall hits (-0.02), collisions (-0.05), and a step penalty (-0.03) to encourage efficiency.

---

## 🧠 Training Methodology: MAPPO

The agents are trained using **Multi-Agent Proximal Policy Optimization (MAPPO)**.

### Centralized Training, Decentralized Execution (CTDE)
- **Actor**: Each agent (Seer, Doers) has its own policy.
- **Critic**: A **Global Critic** observes the `global_map`. This provides a stable baseline for the team, as it has access to information that individual Doers lack during execution.

### Key Training Techniques:
- **Entropy Regularization**: Applied to both actions and **messages**. High message entropy prevents the Seer from collapsing into a single constant signal.
- **GAE (Generalized Advantage Estimation)**: Used to compute stable advantage targets across the sequence.
- **Action Masking**: Prevents Doers from attempting to "pick" objects until they have arrived at the goal and the menu is visible.

---

## 📈 Curriculum Learning Strategy

Training progresses through two main phases defined in `train.py`:

| Phase | Description | Key Mechanism |
| :--- | :--- | :--- |
| **Pick_Object** | Simplified task where agents start at the goals. | Focuses on learning the *naming* of objects/colors. |
| **Full_Training** | Full navigation + selection from random starts. | Focuses on *navigation commands* and *bottleneck coordination*. |

---

## 📂 Codebase Navigation

- **[`train.py`](file:///home/bezin/OpenEndedness/v2_JaxMarl/train.py)**: The main entry point. Orchestrates the training loop, curriculum, and logging.
- **[`models/`](file:///home/bezin/OpenEndedness/v2_JaxMarl/models/)**: Contains the neural architectures (`seer.py`, `doer.py`) and the `fsq.py` quantizer.
- **[`envs/`](file:///home/bezin/OpenEndedness/v2_JaxMarl/envs/)**: Contains the `two_doer_grid.py` definition.
- **[`agents/mappo.py`](file:///home/bezin/OpenEndedness/v2_JaxMarl/agents/mappo.py)**: The core RL update logic.
- **[`training/`](file:///home/bezin/OpenEndedness/v2_JaxMarl/training/)**: Utilities for trajectories, GAE, and bit masking for communication.

---

> [!TIP]
> To start training, run:
> ```bash
> python train.py
> ```
> This will automatically initialize WandB logging and begin the curriculum from Phase 1.
