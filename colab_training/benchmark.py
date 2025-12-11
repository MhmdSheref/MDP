"""
Benchmark Module for RL vs TBS Performance Comparison.

Provides tools for:
1. Evaluating individual policies on environments
2. Comparing RL agents against baseline policies (TBS, BaseStock)
3. Generating performance reports by complexity level
4. Visualization of comparative results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path

from perishable_inventory_mdp.policies import (
    TailoredBaseSurgePolicy,
    BaseStockPolicy,
    DoNothingPolicy
)
from perishable_inventory_mdp.state import InventoryState


@dataclass
class EvaluationResult:
    """Results from evaluating a policy on an environment."""
    env_id: str
    complexity: str
    policy_name: str
    
    # Core metrics
    mean_cost: float
    std_cost: float
    mean_fill_rate: float
    std_fill_rate: float
    
    # Detailed metrics
    mean_procurement_cost: float = 0.0
    mean_holding_cost: float = 0.0
    mean_shortage_cost: float = 0.0
    mean_spoilage: float = 0.0
    
    n_episodes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "env_id": self.env_id,
            "complexity": self.complexity,
            "policy_name": self.policy_name,
            "mean_cost": self.mean_cost,
            "std_cost": self.std_cost,
            "mean_fill_rate": self.mean_fill_rate,
            "std_fill_rate": self.std_fill_rate,
            "mean_procurement_cost": self.mean_procurement_cost,
            "mean_holding_cost": self.mean_holding_cost,
            "mean_shortage_cost": self.mean_shortage_cost,
            "mean_spoilage": self.mean_spoilage,
            "n_episodes": self.n_episodes
        }


@dataclass
class ComparisonReport:
    """Comparison report between RL and baseline policies."""
    results: List[EvaluationResult] = field(default_factory=list)
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def get_summary_by_complexity(self) -> pd.DataFrame:
        """Get aggregated summary by complexity and policy."""
        df = self.to_dataframe()
        if df.empty:
            return df
        
        summary = df.groupby(['complexity', 'policy_name']).agg({
            'mean_cost': ['mean', 'std'],
            'mean_fill_rate': ['mean', 'std'],
            'mean_spoilage': 'mean'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        return summary.reset_index()
    
    def get_rl_vs_tbs_ratio(self) -> pd.DataFrame:
        """Compute RL/TBS cost ratio by complexity."""
        df = self.to_dataframe()
        if df.empty:
            return df
        
        rl_df = df[df['policy_name'] == 'RL'].set_index(['env_id', 'complexity'])
        tbs_df = df[df['policy_name'] == 'TBS'].set_index(['env_id', 'complexity'])
        
        merged = rl_df[['mean_cost']].join(
            tbs_df[['mean_cost']], 
            lsuffix='_rl', 
            rsuffix='_tbs'
        )
        merged['cost_ratio'] = merged['mean_cost_rl'] / merged['mean_cost_tbs'].replace(0, np.nan)
        
        return merged.reset_index().groupby('complexity')['cost_ratio'].agg(['mean', 'std', 'min', 'max'])
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        data = {
            "results": [r.to_dict() for r in self.results]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ComparisonReport':
        """Load report from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        report = cls()
        for r in data['results']:
            report.add_result(EvaluationResult(**r))
        return report


def evaluate_policy(
    policy: Any,
    env: Any,
    n_episodes: int = 10,
    max_steps: int = 500,
    policy_name: str = "policy",
    env_id: str = "",
    complexity: str = ""
) -> EvaluationResult:
    """Evaluate a single policy on an environment.
    
    Args:
        policy: Policy to evaluate (SB3 model or custom policy)
        env: Gym environment or wrapped MDP
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        policy_name: Name for the policy
        env_id: Environment identifier
        complexity: Complexity level
    
    Returns:
        EvaluationResult with aggregated metrics
    """
    is_sb3_model = hasattr(policy, 'predict')
    has_gym_interface = hasattr(env, 'step') and hasattr(env, 'reset')
    
    costs = []
    fill_rates = []
    procurement_costs = []
    holding_costs = []
    shortage_costs = []
    spoilages = []
    
    for _ in range(n_episodes):
        if has_gym_interface:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        else:
            state = env.reset()
        
        episode_cost = 0.0
        episode_procurement = 0.0
        episode_holding = 0.0
        episode_shortage = 0.0
        episode_demand = 0.0
        episode_sales = 0.0
        episode_spoilage = 0.0
        
        for step in range(max_steps):
            if has_gym_interface:
                if is_sb3_model:
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    # Custom policy with gym env - need underlying state
                    if hasattr(env, 'current_state') and hasattr(env, 'mdp'):
                        action_dict = policy.get_action(env.current_state, env.mdp)
                        action = _encode_action_for_env(env, action_dict)
                    elif hasattr(env, 'envs') and len(env.envs) > 0:
                        inner_env = env.envs[0]
                        if hasattr(inner_env, 'current_state') and hasattr(inner_env, 'mdp'):
                            action_dict = policy.get_action(inner_env.current_state, inner_env.mdp)
                            action = _encode_action_for_env(inner_env, action_dict)
                        else:
                            action = env.action_space.sample()
                    else:
                        action = env.action_space.sample()
                
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]
                if isinstance(done, np.ndarray):
                    done = done[0]
                
                episode_cost += info.get('total_cost', 0)
                episode_procurement += info.get('procurement_cost', 0)
                episode_holding += info.get('holding_cost', 0)
                episode_shortage += info.get('shortage_cost', 0)
                episode_demand += info.get('demand', 0)
                episode_sales += info.get('sales', 0)
                episode_spoilage += info.get('spoilage', 0)
                
                if done:
                    break
            else:
                # Direct MDP interface
                action = policy.get_action(state, env)
                result = env.step(state, action)
                state = result.next_state
                
                episode_cost += result.costs.total_cost
                episode_procurement += result.costs.purchase_cost
                episode_holding += result.costs.holding_cost
                episode_shortage += result.costs.shortage_cost
                episode_demand += result.demand_realized
                episode_sales += result.sales
                episode_spoilage += result.spoiled
        
        costs.append(episode_cost)
        fill_rate = episode_sales / max(episode_demand, 1e-6)
        fill_rates.append(fill_rate)
        procurement_costs.append(episode_procurement)
        holding_costs.append(episode_holding)
        shortage_costs.append(episode_shortage)
        spoilages.append(episode_spoilage)
    
    return EvaluationResult(
        env_id=env_id,
        complexity=complexity,
        policy_name=policy_name,
        mean_cost=np.mean(costs),
        std_cost=np.std(costs),
        mean_fill_rate=np.mean(fill_rates),
        std_fill_rate=np.std(fill_rates),
        mean_procurement_cost=np.mean(procurement_costs),
        mean_holding_cost=np.mean(holding_costs),
        mean_shortage_cost=np.mean(shortage_costs),
        mean_spoilage=np.mean(spoilages),
        n_episodes=n_episodes
    )


def _encode_action_for_env(env: Any, action_dict: Dict[int, float]) -> np.ndarray:
    """Convert policy action dict to environment action array."""
    if not hasattr(env, 'supplier_order') or not hasattr(env, 'supplier_action_bins'):
        # Fallback: return zeros
        return np.zeros(len(action_dict), dtype=np.int64)
    
    action = []
    for sid in env.supplier_order:
        qty = action_dict.get(sid, 0)
        bins = env.supplier_action_bins[sid]
        idx = np.argmin(np.abs(np.array(bins) - qty))
        action.append(idx)
    return np.array(action)


def get_tbs_policy_for_env(env: Any) -> TailoredBaseSurgePolicy:
    """Create TBS policy configured for the given environment.
    
    Args:
        env: Gym wrapper or MDP environment
    
    Returns:
        Configured TailoredBaseSurgePolicy
    """
    # Extract underlying MDP
    if hasattr(env, 'mdp'):
        mdp = env.mdp
    elif hasattr(env, 'envs') and len(env.envs) > 0:
        mdp = env.envs[0].mdp if hasattr(env.envs[0], 'mdp') else env.envs[0]
    else:
        mdp = env
    
    # Get supplier info
    suppliers = mdp.suppliers if hasattr(mdp, 'suppliers') else []
    if len(suppliers) < 2:
        raise ValueError("TBS requires at least 2 suppliers")
    
    # Identify slow (cheap) and fast (expensive) suppliers
    sorted_suppliers = sorted(suppliers, key=lambda s: s.get('unit_cost', 1.0))
    slow_supplier = sorted_suppliers[0]  # Cheapest
    fast_supplier = sorted_suppliers[-1]  # Most expensive
    
    # Get demand info
    demand_process = mdp.demand_process if hasattr(mdp, 'demand_process') else None
    if demand_process is None:
        mean_demand = 10.0
        std_demand = np.sqrt(10.0)
    elif hasattr(demand_process, 'base_rate'):
        mean_demand = demand_process.base_rate
        std_demand = np.sqrt(mean_demand)  # Poisson approximation
    elif hasattr(demand_process, 'mean_demand'):
        mean_demand = demand_process.mean_demand
        std_demand = np.sqrt(mean_demand)
    else:
        mean_demand = 10.0
        std_demand = np.sqrt(10.0)
    
    return TailoredBaseSurgePolicy.from_demand_forecast(
        slow_supplier_id=slow_supplier['id'],
        fast_supplier_id=fast_supplier['id'],
        mean_demand=mean_demand,
        std_demand=std_demand,
        slow_lead_time=slow_supplier.get('lead_time', 3),
        fast_lead_time=fast_supplier.get('lead_time', 1),
        service_level=0.95
    )


def get_basestock_policy_for_env(env: Any, target_level: float = 60.0) -> BaseStockPolicy:
    """Create BaseStock policy for the environment.
    
    Args:
        env: Environment
        target_level: Order-up-to level
    
    Returns:
        Configured BaseStockPolicy
    """
    # Use primary (cheapest) supplier
    if hasattr(env, 'mdp'):
        mdp = env.mdp
    elif hasattr(env, 'envs') and len(env.envs) > 0:
        mdp = env.envs[0].mdp if hasattr(env.envs[0], 'mdp') else env.envs[0]
    else:
        mdp = env
    
    suppliers = mdp.suppliers if hasattr(mdp, 'suppliers') else []
    if suppliers:
        sorted_suppliers = sorted(suppliers, key=lambda s: s.get('unit_cost', 1.0))
        supplier_id = sorted_suppliers[0]['id']
    else:
        supplier_id = 0
    
    return BaseStockPolicy(target_level=target_level, supplier_id=supplier_id)


def compare_policies(
    rl_model: Any,
    env_configs: List[Any],
    env_factory: Any,
    n_episodes: int = 10,
    include_baselines: List[str] = None
) -> ComparisonReport:
    """Compare RL agent against baseline policies across environments.
    
    Args:
        rl_model: Trained RL model (SB3)
        env_configs: List of EnvironmentConfig objects
        env_factory: Function to build gym env from config
        n_episodes: Episodes per evaluation
        include_baselines: List of baseline names to include
    
    Returns:
        ComparisonReport with all results
    """
    include_baselines = include_baselines or ["TBS", "BaseStock"]
    report = ComparisonReport()
    
    for config in env_configs:
        env = env_factory(config)
        env_id = config.env_id if hasattr(config, 'env_id') else str(hash(str(config)))
        complexity = config.complexity if hasattr(config, 'complexity') else "unknown"
        
        # Evaluate RL
        rl_result = evaluate_policy(
            policy=rl_model,
            env=env,
            n_episodes=n_episodes,
            policy_name="RL",
            env_id=env_id,
            complexity=complexity
        )
        report.add_result(rl_result)
        
        # Evaluate baselines
        if "TBS" in include_baselines:
            try:
                tbs_policy = get_tbs_policy_for_env(env)
                tbs_result = evaluate_policy(
                    policy=tbs_policy,
                    env=env,
                    n_episodes=n_episodes,
                    policy_name="TBS",
                    env_id=env_id,
                    complexity=complexity
                )
                report.add_result(tbs_result)
            except (ValueError, AttributeError) as e:
                print(f"Skipping TBS for {env_id}: {e}")
        
        if "BaseStock" in include_baselines:
            try:
                bs_policy = get_basestock_policy_for_env(env)
                bs_result = evaluate_policy(
                    policy=bs_policy,
                    env=env,
                    n_episodes=n_episodes,
                    policy_name="BaseStock",
                    env_id=env_id,
                    complexity=complexity
                )
                report.add_result(bs_result)
            except (ValueError, AttributeError) as e:
                print(f"Skipping BaseStock for {env_id}: {e}")
        
        if "DoNothing" in include_baselines:
            dn_policy = DoNothingPolicy()
            dn_result = evaluate_policy(
                policy=dn_policy,
                env=env,
                n_episodes=n_episodes,
                policy_name="DoNothing",
                env_id=env_id,
                complexity=complexity
            )
            report.add_result(dn_result)
        
        env.close() if hasattr(env, 'close') else None
    
    return report


def generate_performance_report(report: ComparisonReport) -> str:
    """Generate human-readable performance report.
    
    Args:
        report: ComparisonReport with evaluation results
    
    Returns:
        Formatted string report
    """
    summary = report.get_summary_by_complexity()
    ratio = report.get_rl_vs_tbs_ratio()
    
    lines = [
        "=" * 60,
        "RL vs TBS Performance Comparison Report",
        "=" * 60,
        "",
        "Summary by Complexity Level:",
        "-" * 40,
    ]
    
    if not summary.empty:
        lines.append(summary.to_string())
    else:
        lines.append("No results available")
    
    lines.extend([
        "",
        "RL/TBS Cost Ratio by Complexity:",
        "-" * 40,
    ])
    
    if not ratio.empty:
        lines.append(ratio.to_string())
        lines.append("")
        
        # Interpretation
        for complexity, row in ratio.iterrows():
            mean_ratio = row['mean']
            if mean_ratio < 1.0:
                interpretation = f"RL outperforms TBS by {(1-mean_ratio)*100:.1f}%"
            elif mean_ratio > 1.0:
                interpretation = f"TBS outperforms RL by {(mean_ratio-1)*100:.1f}%"
            else:
                interpretation = "RL and TBS perform equally"
            lines.append(f"  {complexity}: {interpretation}")
    else:
        lines.append("No comparison available")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def visualize_comparison(
    report: ComparisonReport,
    save_path: Optional[str] = None
) -> Any:
    """Generate visualization of RL vs TBS comparison.
    
    Args:
        report: ComparisonReport
        save_path: Optional path to save figure
    
    Returns:
        matplotlib figure object (if matplotlib available)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None
    
    df = report.to_dataframe()
    if df.empty:
        print("No data to visualize")
        return None
    
    # Group by complexity and policy
    summary = df.groupby(['complexity', 'policy_name']).agg({
        'mean_cost': 'mean',
        'mean_fill_rate': 'mean'
    }).reset_index()
    
    complexities = ["simple", "moderate", "complex", "extreme"]
    policies = summary['policy_name'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost comparison
    ax1 = axes[0]
    x = np.arange(len(complexities))
    width = 0.25
    
    for i, policy in enumerate(policies):
        policy_data = summary[summary['policy_name'] == policy]
        costs = []
        for c in complexities:
            row = policy_data[policy_data['complexity'] == c]
            costs.append(row['mean_cost'].values[0] if len(row) > 0 else 0)
        ax1.bar(x + i*width, costs, width, label=policy)
    
    ax1.set_xlabel('Complexity')
    ax1.set_ylabel('Mean Cost')
    ax1.set_title('Cost by Complexity Level')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(complexities)
    ax1.legend()
    
    # Fill rate comparison
    ax2 = axes[1]
    for i, policy in enumerate(policies):
        policy_data = summary[summary['policy_name'] == policy]
        fill_rates = []
        for c in complexities:
            row = policy_data[policy_data['complexity'] == c]
            fill_rates.append(row['mean_fill_rate'].values[0] if len(row) > 0 else 0)
        ax2.bar(x + i*width, fill_rates, width, label=policy)
    
    ax2.set_xlabel('Complexity')
    ax2.set_ylabel('Fill Rate')
    ax2.set_title('Fill Rate by Complexity Level')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(complexities)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig
