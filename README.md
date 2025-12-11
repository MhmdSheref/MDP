# 🏭 Multi-Supplier Perishable Inventory MDP

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art Reinforcement Learning (RL) environment for managing perishable pharmaceutical inventory. This project implements a complex Markov Decision Process (MDP) with multiple suppliers, stochastic demand, lead times, and spoilage dynamics, designed to benchmark RL agents against traditional operations research policies.

---

## 🚀 Key Features

### 🧠 Advanced RL Integration
- **Gymnasium Compatible**: Fully compliant `PerishableInventoryGymWrapper` for seamless integration with Stable Baselines3, Ray RLLib, etc.
- **Cost-Aware Observations**: Observation space includes normalized inventory, pipeline states, supplier costs, lead times, and crisis indicators.
- **Asymmetric Action Space**: Innovative action space design that favors ordering from cheaper/slower suppliers while allowing surge orders from expensive/fast ones.
- **Curriculum Learning**: Built-in support for progressive training from simple to extreme complexity scenarios.

### 🌍 Comprehensive Environment Suite
- **100+ Unique Environments**: A generated suite of environments ranging from simple deterministic settings to chaotic, high-variance scenarios.
- **Complexity Levels**:
    - **Simple**: Baseline scenarios where traditional policies excel.
    - **Moderate**: Introduces seasonality and stochastic lead times.
    - **Complex**: Composite demand patterns and multi-supplier constraints.
    - **Extreme**: Crisis dynamics, massive volatility, and tight constraints.

### 📦 Core MDP Logic
- **Multi-Item Support**: Manage inventory for multiple distinct products with shared or separate suppliers.
- **Complex Demand**: Support for Poisson, Negative Binomial, Seasonal, Trend, and Composite demand processes.
- **Crisis Dynamics**: Simulate supply chain disruptions and demand spikes.
- **Supplier Contracts**: Model volume discounts and minimum order quantities (MOQ).

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/MahmoudZah/Multi-Supplier-Perishable-Inventory.git
cd Multi-Supplier-Perishable-Inventory

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃‍♂️ Quick Start

### Training an RL Agent

We provide a production-ready training script `train_rl.py` that implements PPO with curriculum learning and benchmarking.

```bash
# Train with default settings (5M steps, 8 parallel envs)
python colab_training/train_rl.py

# Run a quick test
python colab_training/train_rl.py --test-mode
```

### Using the Environment

```python
from colab_training.gym_env import create_gym_env

# Create a complex environment
env = create_gym_env(
    shelf_life=5,
    mean_demand=20.0,
    enable_crisis=True,
    fast_lead_time=1,
    slow_lead_time=4
)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Your agent here
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

---

## 📊 Benchmarking

The system automatically benchmarks RL performance against the **Tailored Base-Surge (TBS)** policy, a known heuristic for dual-sourcing problems.

| Complexity | RL vs TBS Ratio | Interpretation |
|------------|-----------------|----------------|
| **Simple** | ~1.00 | RL matches optimal policy |
| **Moderate** | 0.95 - 1.05 | RL competitive with tuned heuristic |
| **Complex** | **< 0.90** | RL discovers superior strategies |
| **Extreme** | **< 0.85** | RL significantly outperforms in chaos |

---

## 📂 Project Structure

```
.
├── perishable_inventory_mdp/   # Core MDP implementation
│   ├── state.py                # Inventory and Pipeline state
│   ├── demand.py               # Demand process generation
│   ├── environment.py          # Main MDP logic
│   └── ...
├── colab_training/             # RL Training Infrastructure
│   ├── gym_env.py              # Gymnasium Wrapper
│   ├── train_rl.py             # Training Script (PPO)
│   ├── environment_suite.py    # 100+ Benchmark Environments
│   └── ...
├── tests/                      # Comprehensive Test Suite
└── README.md
```

---

## 📝 License

This project is licensed under the MIT License.
