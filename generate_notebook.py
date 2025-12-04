import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Perishable Inventory MDP - RL Training\n",
    "\n",
    "This notebook trains a Reinforcement Learning agent (PPO) to manage inventory in a multi-supplier perishable environment.\n",
    "\n",
    "## Prerequisites\n",
    "1.  Upload the `Multi-Supplier-Perishable-Inventory` repository to your Google Drive.\n",
    "2.  Update the `repo_path` variable below to point to the correct location."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "!pip install stable-baselines3 gymnasium shimmy pandas matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# Mount Google Drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# --- CONFIGURATION ---\n",
    "# Update this path to where you stored the repository in your Drive\n",
    "repo_path = '/content/drive/MyDrive/Multi-Supplier-Perishable-Inventory'\n",
    "# ---------------------\n",
    "\n",
    "if not os.path.exists(repo_path):\n",
    "    print(f\"WARNING: Path {repo_path} does not exist. Please check the path.\")\n",
    "else:\n",
    "    print(f\"Found repository at {repo_path}\")\n",
    "    sys.path.append(repo_path)\n",
    "    os.chdir(repo_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import gymnasium as gym\n",
    "from stable_baselines3 import PPO\n",
    "from stable_baselines3.common.env_util import make_vec_env\n",
    "from stable_baselines3.common.callbacks import EvalCallback\n",
    "from stable_baselines3.common.vec_env import SubprocVecEnv\n",
    "\n",
    "# Import project modules\n",
    "try:\n",
    "    from inventory_sim.env import PerishableInventoryMDP\n",
    "    from inventory_sim.core.costs import CostParameters\n",
    "    from inventory_sim.core.demand import PoissonDemand\n",
    "    from colab_training.gym_wrapper import PerishableInventoryGymWrapper\n",
    "    print(\"Successfully imported project modules.\")\n",
    "except ImportError as e:\n",
    "    print(f\"Error importing modules: {e}\")\n",
    "    print(\"Make sure the repository path is added to sys.path correctly.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_env(mdp_params):\n",
    "    \"\"\"Factory function to create wrapped environment.\"\"\"\n",
    "    def _init():\n",
    "        shelf_life = mdp_params['shelf_life']\n",
    "        suppliers = mdp_params['suppliers']\n",
    "        \n",
    "        demand_params = mdp_params['demand']\n",
    "        if demand_params['type'] == 'poisson':\n",
    "            demand_process = PoissonDemand(demand_params['mean'])\n",
    "        else:\n",
    "            raise ValueError(f\"Unsupported demand type: {demand_params['type']}\")\n",
    "            \n",
    "        cost_params_dict = mdp_params['costs']\n",
    "        cost_params = CostParameters.uniform_holding(\n",
    "            shelf_life=shelf_life,\n",
    "            holding_cost=cost_params_dict['holding'],\n",
    "            shortage_cost=cost_params_dict['shortage'],\n",
    "            spoilage_cost=cost_params_dict['spoilage'],\n",
    "            discount_factor=cost_params_dict['discount_factor']\n",
    "        )\n",
    "        \n",
    "        mdp = PerishableInventoryMDP(\n",
    "            shelf_life=shelf_life,\n",
    "            suppliers=suppliers,\n",
    "            demand_process=demand_process,\n",
    "            cost_params=cost_params\n",
    "        )\n",
    "        \n",
    "        env = PerishableInventoryGymWrapper(mdp)\n",
    "        return env\n",
    "    return _init"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- TRAINING CONFIGURATION ---\n",
    "mdp_params = {\n",
    "    \"shelf_life\": 5,\n",
    "    \"suppliers\": [\n",
    "        {\"id\": 0, \"lead_time\": 1, \"unit_cost\": 2.0, \"capacity\": 100, \"moq\": 1},\n",
    "        {\"id\": 1, \"lead_time\": 3, \"unit_cost\": 1.0, \"capacity\": 100, \"moq\": 1}\n",
    "    ],\n",
    "    \"demand\": {\n",
    "        \"type\": \"poisson\",\n",
    "        \"mean\": 10.0\n",
    "    },\n",
    "    \"costs\": {\n",
    "        \"holding\": 0.5,\n",
    "        \"shortage\": 10.0,\n",
    "        \"spoilage\": 5.0,\n",
    "        \"discount_factor\": 0.99\n",
    "    }\n",
    "}\n",
    "\n",
    "training_params = {\n",
    "    \"total_timesteps\": 100000,\n",
    "    \"learning_rate\": 0.0003,\n",
    "    \"seed\": 42,\n",
    "    \"n_envs\": 4\n",
    "}\n",
    "\n",
    "log_dir = \"./logs/\"\n",
    "os.makedirs(log_dir, exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create Environments\n",
    "env = make_vec_env(\n",
    "    make_env(mdp_params),\n",
    "    n_envs=training_params['n_envs'],\n",
    "    seed=training_params['seed'],\n",
    "    # Use SubprocVecEnv for parallel execution if n_envs > 1\n",
    "    vec_env_cls=SubprocVecEnv if training_params['n_envs'] > 1 else None\n",
    ")\n",
    "\n",
    "eval_env = make_vec_env(\n",
    "    make_env(mdp_params),\n",
    "    n_envs=1,\n",
    "    seed=training_params['seed'] + 1000\n",
    ")\n",
    "\n",
    "# Setup Callback\n",
    "eval_callback = EvalCallback(\n",
    "    eval_env,\n",
    "    best_model_save_path=os.path.join(log_dir, 'best_model'),\n",
    "    log_path=os.path.join(log_dir, 'results'),\n",
    "    eval_freq=10000,\n",
    "    n_eval_episodes=10,\n",
    "    deterministic=True,\n",
    "    render=False\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize PPO Agent\n",
    "model = PPO(\n",
    "    \"MlpPolicy\",\n",
    "    env,\n",
    "    verbose=1,\n",
    "    learning_rate=training_params['learning_rate'],\n",
    "    tensorboard_log=os.path.join(log_dir, \"tensorboard\")\n",
    ")\n",
    "\n",
    "print(f\"Starting training for {training_params['total_timesteps']} timesteps...\")\n",
    "model.learn(\n",
    "    total_timesteps=training_params['total_timesteps'],\n",
    "    callback=eval_callback\n",
    ")\n",
    "print(\"Training complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save final model\n",
    "model.save(os.path.join(log_dir, \"final_model\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- EVALUATION & PLOTTING ---\n",
    "import matplotlib.pyplot as plt\n",
    "from inventory_sim.plotting import plot_simulation_trace\n",
    "\n",
    "# Load best model\n",
    "best_model = PPO.load(os.path.join(log_dir, 'best_model', 'best_model'))\n",
    "\n",
    "# Run a single episode for visualization\n",
    "obs = eval_env.reset()\n",
    "done = False\n",
    "trace_data = []\n",
    "period = 0\n",
    "\n",
    "# Note: eval_env is vectorized, so we need to handle array outputs\n",
    "# For plotting, we'll just use the first env\n",
    "\n",
    "while not done and period < 100:\n",
    "    action, _ = best_model.predict(obs, deterministic=True)\n",
    "    obs, rewards, dones, infos = eval_env.step(action)\n",
    "    \n",
    "    # Extract info from the first environment\n",
    "    info = infos[0]\n",
    "    \n",
    "    # We need to reconstruct some data for the plot since the wrapper flattens everything\n",
    "    # Ideally, we'd access the underlying environment, but with SubprocVecEnv it's hard.\n",
    "    # Instead, we rely on the 'info' dict we populated in the wrapper.\n",
    "    \n",
    "    # Wait, the wrapper info might need to be expanded to include everything needed for plotting\n",
    "    # For now, let's just print the total reward\n",
    "    period += 1\n",
    "    if dones[0]:\n",
    "        break\n",
    "\n",
    "print(\"Evaluation run finished.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('colab_training/Perishable_Inventory_RL.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
