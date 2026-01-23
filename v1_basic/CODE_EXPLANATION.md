# Heist: Multi-Agent Reinforcement Learning Project

This project implements a "Blind Fetch" task where two agents, a **Seer** and a **Doer**, must cooperate to reach a target in a grid world using Multi-Agent Reinforcement Learning (MARL).

## Project Structure

- `env.py`: Defines the `BlindFetchEnv` grid world environment.
- `models.py`: Contains the `ActorCritic` neural network architecture.
- `train.py`: Implements the PPO training loop using JAX and Flax.
- `analysis.py`: Handles data visualization and result plotting.
- `main.py`: The entry point for running training and evaluation.
- `debug_model.py`: Utility for checking model behavior.

---

## 1. Environment (`env.py`)

The environment is a **5x5 grid world**.

- **State**: The `EnvState` tracks the Doer's position, the target position, the last message sent by the Seer, and the current time step.
- **Agents**:
  - **Seer**: Receives the full state (Doer position and Target position).
  - **Doer**: Receives only the **Seer's message**. It does not know where the target is.
- **Actions**:
  - Seer: Sends a continuous 5-dimensional message.
  - Doer: Chooses one of 5 moves: `Stay`, `Up`, `Down`, `Left`, `Right`.
- **Rewards**:
  - `+1.0` for reaching the target.
  - `-0.01` penalty per step to encourage efficiency.
- **Observations**:
  - `seer`: Normalized Doer and Target positions.
  - `doer`: The last message received from the Seer.

## 2. Models (`models.py`)

The project uses an **Actor-Critic** architecture implemented in Flax.

- **Seer Head**: A dense network that outputs the parameters (mean) of a 5D Gaussian distribution for communication.
- **Doer Head**: A dense network that outputs logits for a categorical distribution over the 5 possible moves.
- **Centralized Critic**: A value head that takes both the Seer's and Doer's observations to estimate the expected return, enabling better variance reduction during training.

## 3. Training (`train.py`)

Training is performed using **Proximal Policy Optimization (PPO)**, optimized for JAX's high-performance computing capabilities.

- **Vectorized Environments**: Uses `jax.vmap` to run hundreds of environments in parallel.
- **Rollout**: Collects trajectories of experiences (observations, actions, rewards).
- **GAE (Generalized Advantage Estimation)**: Computes advantages to balance bias and variance in policy gradients.
- **PPO Update**: Performs policy and value updates using clipped gradients and Adam optimizer. It includes an entropy bonus to encourage exploration.
- **Metrics**: Tracks loss, reward, success rate, and "communication pulse" (message magnitude).

## 4. Analysis & Evaluation (`analysis.py` & `main.py`)

After training, the project performs a detailed analysis of the learned behavior.

- **Learning Curve**: Plots Mean Reward and Success Rate over training updates.
- **Communication Pulse**: Visualizes how the magnitude of the Seer's messages evolves.
- **Grounding Heatmap**: Uses **PCA (Principal Component Analysis)** on Seer messages to visualize how communication "grounds" to specific grid locations.
- **Trajectory Trace**: Visualizes the path taken by the Doer in a single evaluation episode, demonstrating the final learned cooperation.

## 5. Execution Flow

1. **Initialization**: Configures hyperparameters (learning rate, environment count, etc.).
2. **Training**: Executes the `make_train` loop in `train.py`.
3. **Data Collection**: Runs evaluation episodes to collect messages and trajectories.
4. **Visualization**: Generates `.png` plots using `analysis.py`.
5. **Logging**: Saves training metrics to `training_log.csv`.
