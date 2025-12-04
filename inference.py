#!/usr/bin/env python
"""
Inference Script for Trained RL Inventory Agent

This script loads a trained RL model and runs inference on the
perishable inventory MDP environment, comparing performance against
baseline policies.

Usage:
    python inference.py --model models/ppo_inventory_final.zip
    python inference.py --model models/ppo_inventory_final.zip --episodes 20 --render
    python inference.py --model models/ppo_inventory_final.zip --compare-baselines

Requirements:
    pip install stable-baselines3 gymnasium numpy scipy matplotlib
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError:
    print("Error: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

from inventory_sim.gym_env import PerishableInventoryGymEnv
from inventory_sim.env import create_simple_mdp
from inventory_sim.simulation import run_episode
from inventory_agents import (
    TailoredBaseSurgePolicy,
    BaseStockPolicy,
    DoNothingPolicy,
    ConstantOrderPolicy
)


# Default environment configuration
DEFAULT_ENV_CONFIG = {
    "shelf_life": 5,
    "num_suppliers": 2,
    "mean_demand": 10.0,
    "fast_lead_time": 1,
    "slow_lead_time": 3,
    "fast_cost": 2.0,
    "slow_cost": 1.0,
    "max_order_per_supplier": 30,
    "order_step": 5,
    "max_episode_steps": 200,
    "normalize_obs": True,
}


class RLAgent:
    """Wrapper class for RL agent inference."""
    
    def __init__(self, model_path: str, env_config: Optional[Dict] = None):
        """
        Load a trained RL model.
        
        Args:
            model_path: Path to the trained model (.zip file)
            env_config: Environment configuration (uses defaults if not provided)
        """
        self.model_path = Path(model_path)
        self.env_config = env_config or DEFAULT_ENV_CONFIG
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = PPO.load(str(self.model_path))
        print(f"Model loaded successfully!")
        
        # Create environment for inference
        self.env = PerishableInventoryGymEnv(**self.env_config)
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for given observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic prediction
        
        Returns:
            Tuple of (action, hidden_states)
        """
        action, states = self.model.predict(observation, deterministic=deterministic)
        return action, states
    
    def run_episode(
        self,
        seed: Optional[int] = None,
        render: bool = False,
        deterministic: bool = True
    ) -> Dict:
        """
        Run a single episode with the trained agent.
        
        Args:
            seed: Random seed for reproducibility
            render: Whether to print state at each step
            deterministic: Whether to use deterministic actions
        
        Returns:
            Dictionary with episode metrics
        """
        obs, info = self.env.reset(seed=seed)
        
        episode_reward = 0.0
        total_demand = 0.0
        total_sales = 0.0
        total_spoiled = 0.0
        total_purchase_cost = 0.0
        step_count = 0
        
        actions_taken = []
        
        while True:
            action, _ = self.predict(obs, deterministic=deterministic)
            
            if render:
                order_fast = action[0] * self.env_config["order_step"]
                order_slow = action[1] * self.env_config["order_step"]
                print(f"Step {step_count}: Action=[{order_fast}, {order_slow}], "
                      f"IP={info.get('inventory_position', 0):.0f}")
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            episode_reward += reward
            total_demand += info.get('demand', 0)
            total_sales += info.get('sales', 0)
            total_spoiled += info.get('spoiled', 0)
            total_purchase_cost += info.get('period_cost', 0) * 0.5  # Approximate
            step_count += 1
            
            actions_taken.append(action.tolist())
            
            if terminated or truncated:
                break
        
        fill_rate = total_sales / total_demand if total_demand > 0 else 1.0
        
        return {
            "episode_reward": episode_reward,
            "total_cost": -episode_reward,
            "steps": step_count,
            "total_demand": total_demand,
            "total_sales": total_sales,
            "total_spoiled": total_spoiled,
            "fill_rate": fill_rate,
            "actions": actions_taken,
        }
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate the agent over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to run
            deterministic: Whether to use deterministic actions
            verbose: Whether to print progress
        
        Returns:
            Dictionary with aggregate metrics
        """
        all_rewards = []
        all_fill_rates = []
        all_spoilage = []
        
        for ep in range(n_episodes):
            result = self.run_episode(seed=42 + ep, deterministic=deterministic)
            all_rewards.append(result["episode_reward"])
            all_fill_rates.append(result["fill_rate"])
            all_spoilage.append(result["total_spoiled"])
            
            if verbose:
                print(f"Episode {ep + 1}/{n_episodes}: "
                      f"Reward={result['episode_reward']:.2f}, "
                      f"Fill Rate={result['fill_rate']:.2%}")
        
        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_cost": -np.mean(all_rewards),
            "mean_fill_rate": np.mean(all_fill_rates),
            "std_fill_rate": np.std(all_fill_rates),
            "mean_spoilage": np.mean(all_spoilage),
            "n_episodes": n_episodes,
        }


def evaluate_baseline_policies(
    env_config: Dict,
    n_episodes: int = 10,
    n_periods: int = 200,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate baseline policies for comparison.
    
    Args:
        env_config: Environment configuration
        n_episodes: Number of episodes to run
        n_periods: Episode length
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping policy names to metrics
    """
    # Create MDP for baseline evaluation
    mdp = create_simple_mdp(
        shelf_life=env_config["shelf_life"],
        num_suppliers=env_config["num_suppliers"],
        mean_demand=env_config["mean_demand"],
        fast_lead_time=env_config["fast_lead_time"],
        slow_lead_time=env_config["slow_lead_time"],
        fast_cost=env_config["fast_cost"],
        slow_cost=env_config["slow_cost"],
    )
    
    # Define baseline policies
    baselines = {
        "Do Nothing": DoNothingPolicy(),
        "Constant (10/period)": ConstantOrderPolicy({0: 0.0, 1: 10.0}),
        "Base Stock (S=60)": BaseStockPolicy(target_level=60.0, supplier_id=1),
        "TBS (Base=50, Reorder=25)": TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=50.0,
            reorder_point=25.0
        ),
    }
    
    results = {}
    
    for name, policy in baselines.items():
        if verbose:
            print(f"\nEvaluating {name}...")
        
        all_rewards = []
        all_fill_rates = []
        all_spoilage_rates = []
        
        for ep in range(n_episodes):
            initial_inv = np.full(mdp.shelf_life, 20.0)
            state = mdp.create_initial_state(initial_inventory=initial_inv)
            
            episode_results, total_reward = run_episode(
                mdp, policy, 
                num_periods=n_periods, 
                seed=42 + ep, 
                initial_state=state
            )
            metrics = mdp.compute_inventory_metrics(episode_results)
            
            all_rewards.append(total_reward)
            all_fill_rates.append(metrics["fill_rate"])
            all_spoilage_rates.append(metrics["spoilage_rate"])
        
        results[name] = {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_cost": -np.mean(all_rewards),
            "mean_fill_rate": np.mean(all_fill_rates),
            "mean_spoilage_rate": np.mean(all_spoilage_rates),
            "n_episodes": n_episodes,
        }
        
        if verbose:
            print(f"  Mean Reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
            print(f"  Fill Rate: {results[name]['mean_fill_rate']:.2%}")
            print(f"  Spoilage Rate: {results[name]['mean_spoilage_rate']:.2%}")
    
    return results


def print_comparison_table(rl_results: Dict, baseline_results: Dict[str, Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Policy':<35} {'Mean Reward':>15} {'Fill Rate':>12} {'Spoilage':>12}")
    print("-" * 80)
    
    # Print baseline results
    for name, metrics in baseline_results.items():
        print(f"{name:<35} {metrics['mean_reward']:>12.2f} ± {metrics['std_reward']:<5.2f} "
              f"{metrics['mean_fill_rate']:>10.2%} {metrics.get('mean_spoilage_rate', 0):>10.2%}")
    
    # Print RL results
    print("-" * 80)
    print(f"{'RL Agent (PPO)':<35} {rl_results['mean_reward']:>12.2f} ± {rl_results['std_reward']:<5.2f} "
          f"{rl_results['mean_fill_rate']:>10.2%} {rl_results.get('mean_spoilage', 0)/200:>10.2%}")
    
    print("=" * 80)
    
    # Find best baseline
    best_baseline_name = max(baseline_results.keys(), 
                            key=lambda k: baseline_results[k]['mean_reward'])
    best_baseline = baseline_results[best_baseline_name]
    
    improvement = rl_results['mean_reward'] - best_baseline['mean_reward']
    
    print(f"\nRL Agent vs Best Baseline ({best_baseline_name}):")
    if improvement > 0:
        print(f"  ✅ RL Agent outperforms by {improvement:.2f} reward units ({100*improvement/abs(best_baseline['mean_reward']):.1f}% improvement)")
    else:
        print(f"  📈 RL Agent underperforms by {-improvement:.2f} reward units")
        print(f"     Consider training for more steps or tuning hyperparameters.")


def generate_visualization(
    rl_agent: RLAgent,
    baseline_results: Dict[str, Dict],
    output_dir: str = "."
):
    """Generate comparison visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    # Run RL evaluation for plot data
    rl_results = rl_agent.evaluate(n_episodes=10, verbose=False)
    
    # Prepare data
    all_policies = list(baseline_results.keys()) + ["RL Agent (PPO)"]
    all_results = list(baseline_results.values()) + [rl_results]
    
    rewards = [r["mean_reward"] for r in all_results]
    reward_stds = [r["std_reward"] for r in all_results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['gray', 'steelblue', 'forestgreen', 'orange', 'crimson']
    y_pos = np.arange(len(all_policies))
    
    bars = ax.barh(y_pos, rewards, xerr=reward_stds, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_policies)
    ax.set_xlabel("Mean Episode Reward (higher is better)")
    ax.set_title("RL Agent vs Baseline Policies")
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, reward in zip(bars, rewards):
        offset = -30 if reward < 0 else 5
        ax.text(reward + offset, bar.get_y() + bar.get_height()/2, 
                f'{reward:.0f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "inference_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for trained RL inventory agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --model models/ppo_inventory_final.zip
  python inference.py --model models/ppo_inventory_final.zip --episodes 20
  python inference.py --model models/ppo_inventory_final.zip --compare-baselines --visualize
  python inference.py --model models/ppo_inventory_final.zip --render --episodes 1
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)"
    )
    
    parser.add_argument(
        "--compare-baselines", "-c",
        action="store_true",
        help="Compare against baseline policies"
    )
    
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render episode steps (for single episode debugging)"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate comparison visualization"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)"
    )
    
    parser.add_argument(
        "--config", "-cfg",
        type=str,
        default=None,
        help="Path to JSON config file for environment"
    )
    
    parser.add_argument(
        "--deterministic", "-d",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)"
    )
    
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    env_config = DEFAULT_ENV_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            env_config.update(custom_config)
    
    # Determine if deterministic
    deterministic = not args.stochastic
    
    print("=" * 60)
    print("RL INVENTORY AGENT INFERENCE")
    print("=" * 60)
    
    # Load RL agent
    try:
        rl_agent = RLAgent(args.model, env_config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Run single rendered episode if requested
    if args.render:
        print("\n" + "-" * 40)
        print("RUNNING RENDERED EPISODE")
        print("-" * 40)
        result = rl_agent.run_episode(seed=42, render=True, deterministic=deterministic)
        print(f"\nEpisode Summary:")
        print(f"  Total Reward: {result['episode_reward']:.2f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Fill Rate: {result['fill_rate']:.2%}")
        return
    
    # Evaluate RL agent
    print("\n" + "-" * 40)
    print("EVALUATING RL AGENT")
    print("-" * 40)
    rl_results = rl_agent.evaluate(
        n_episodes=args.episodes,
        deterministic=deterministic,
        verbose=True
    )
    
    print(f"\nAggregate Results ({args.episodes} episodes):")
    print(f"  Mean Reward: {rl_results['mean_reward']:.2f} ± {rl_results['std_reward']:.2f}")
    print(f"  Mean Cost: {rl_results['mean_cost']:.2f}")
    print(f"  Mean Fill Rate: {rl_results['mean_fill_rate']:.2%}")
    
    # Compare with baselines if requested
    baseline_results = None
    if args.compare_baselines:
        print("\n" + "-" * 40)
        print("EVALUATING BASELINE POLICIES")
        print("-" * 40)
        baseline_results = evaluate_baseline_policies(
            env_config,
            n_episodes=args.episodes,
            n_periods=env_config["max_episode_steps"],
            verbose=True
        )
        
        print_comparison_table(rl_results, baseline_results)
    
    # Generate visualization if requested
    if args.visualize:
        if baseline_results is None:
            print("\nGenerating baselines for visualization...")
            baseline_results = evaluate_baseline_policies(
                env_config,
                n_episodes=args.episodes,
                verbose=False
            )
        generate_visualization(rl_agent, baseline_results, args.output_dir)
    
    # Save results to JSON
    output_path = Path(args.output_dir) / "inference_results.json"
    results = {
        "model_path": str(args.model),
        "n_episodes": args.episodes,
        "deterministic": deterministic,
        "rl_agent": rl_results,
    }
    if baseline_results:
        results["baselines"] = baseline_results
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to {output_path}")
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
