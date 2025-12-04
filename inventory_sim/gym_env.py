"""
Gymnasium-compatible wrapper for the Perishable Inventory MDP.

This module provides a Gym environment that can be used with standard
RL libraries like Stable Baselines3 for training agents.

TRAINING TIPS:
- Use reward_shaping=True for faster learning
- Start with shorter episodes (100 steps) and increase
- Use normalize_reward=True to keep rewards in reasonable range
- Consider curriculum learning: start with higher initial inventory
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List

from .env import PerishableInventoryMDP, create_simple_mdp, TransitionResult
from .core.state import InventoryState


class PerishableInventoryGymEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Perishable Inventory MDP.
    
    This environment wraps the PerishableInventoryMDP to be compatible with
    Gymnasium and Stable Baselines3 for reinforcement learning training.
    
    Key Features:
    - Reward normalization to keep values in reasonable range
    - Reward shaping to guide learning (bonus for maintaining inventory)
    - Configurable action space (discrete or continuous)
    - Proper observation normalization
    
    Observation Space:
        - Inventory by expiry bucket (shelf_life floats)
        - Pipeline totals per supplier (num_suppliers floats)
        - Backorders (1 float)
        - Inventory position (1 float)
        - Mean demand (1 float) - helps agent learn demand level
    
    Action Space:
        MultiDiscrete: Order quantities for each supplier
        Each action dimension represents units to order from that supplier
    
    Reward:
        Shaped reward combining cost minimization and service level
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        shelf_life: int = 5,
        num_suppliers: int = 2,
        mean_demand: float = 10.0,
        fast_lead_time: int = 1,
        slow_lead_time: int = 3,
        fast_cost: float = 2.0,
        slow_cost: float = 1.0,
        max_order_per_supplier: int = 30,
        order_step: int = 5,
        max_episode_steps: int = 200,
        initial_inventory: Optional[np.ndarray] = None,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        reward_shaping: bool = True,
        target_service_level: float = 0.95,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Gymnasium environment.
        
        Args:
            shelf_life: Number of expiry buckets
            num_suppliers: Number of suppliers
            mean_demand: Mean demand per period (Poisson)
            fast_lead_time: Lead time for fast supplier
            slow_lead_time: Lead time for slow supplier
            fast_cost: Unit cost for fast supplier
            slow_cost: Unit cost for slow supplier
            max_order_per_supplier: Maximum order quantity per supplier
            order_step: Step size for discrete actions (e.g., 5 = order 0, 5, 10, ...)
            max_episode_steps: Maximum steps per episode
            initial_inventory: Initial inventory levels (if None, uses safety stock)
            normalize_obs: Whether to normalize observations to [0, 1]
            normalize_reward: Whether to normalize rewards to reasonable range
            reward_shaping: Whether to add shaping rewards for better learning
            target_service_level: Target service level for shaping reward
            render_mode: Render mode ("human" or "ansi")
        """
        super().__init__()
        
        self.shelf_life = shelf_life
        self.num_suppliers = num_suppliers
        self.mean_demand = mean_demand
        self.max_order = max_order_per_supplier
        self.order_step = order_step
        self.max_episode_steps = max_episode_steps
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.reward_shaping = reward_shaping
        self.target_service_level = target_service_level
        self.render_mode = render_mode
        
        # Compute target inventory position (lead time demand + safety stock)
        self.target_inventory_position = mean_demand * (slow_lead_time + 2)  # Approx base-stock level
        
        # Store initial inventory for reset
        if initial_inventory is not None:
            self._initial_inventory = initial_inventory.copy()
        else:
            # Default: start with good inventory level to help exploration
            # Distribute across freshness buckets
            self._initial_inventory = np.full(shelf_life, mean_demand * 1.5)
        
        # Create the underlying MDP
        self.mdp = create_simple_mdp(
            shelf_life=shelf_life,
            num_suppliers=num_suppliers,
            mean_demand=mean_demand,
            fast_lead_time=fast_lead_time,
            slow_lead_time=slow_lead_time,
            fast_cost=fast_cost,
            slow_cost=slow_cost
        )
        
        # Define action space: MultiDiscrete for each supplier
        # Number of actions per supplier = max_order // order_step + 1
        num_actions_per_supplier = (max_order_per_supplier // order_step) + 1
        self.action_space = spaces.MultiDiscrete(
            [num_actions_per_supplier] * num_suppliers
        )
        
        # Define observation space
        # [inventory_buckets, pipeline_totals, backorders, inventory_position, demand_info]
        obs_dim = shelf_life + num_suppliers + 3  # +3 for backorders, IP, and demand level
        
        if normalize_obs:
            self.observation_space = spaces.Box(
                low=-1.0,  # Allow negative for normalized backorders
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
        
        # Normalization constants (based on expected ranges)
        self._inv_scale = mean_demand * 10  # Scale inventory values
        self._pipeline_scale = mean_demand * 5
        self._backorder_scale = mean_demand * 3
        self._ip_scale = mean_demand * 15
        
        # Reward normalization (approximate expected cost per step)
        # Expected cost ≈ holding + some purchase cost
        self._reward_scale = mean_demand * slow_cost * 2 + mean_demand * 0.5  # Approx per-step cost
        
        # State tracking
        self._state: Optional[InventoryState] = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_result: Optional[TransitionResult] = None
        self._cumulative_demand = 0.0
        self._cumulative_sales = 0.0
    
    def _get_obs(self) -> np.ndarray:
        """Convert InventoryState to observation array."""
        if self._state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Inventory by bucket (normalized)
        if self.normalize_obs:
            inventory = self._state.inventory / self._inv_scale
            
            # Pipeline totals per supplier (normalized)
            pipeline_totals = np.array([
                p.total_in_pipeline() / self._pipeline_scale
                for p in self._state.pipelines.values()
            ])
            
            # Other features (normalized)
            backorders = self._state.backorders / self._backorder_scale
            ip = self._state.inventory_position / self._ip_scale
            demand_level = self.mean_demand / (self.mean_demand * 2)  # Constant, normalized
        else:
            inventory = self._state.inventory.copy()
            pipeline_totals = np.array([
                p.total_in_pipeline() 
                for p in self._state.pipelines.values()
            ])
            backorders = self._state.backorders
            ip = self._state.inventory_position
            demand_level = self.mean_demand
        
        # Combine into observation
        obs = np.concatenate([
            inventory,
            pipeline_totals,
            [backorders],
            [ip],
            [demand_level]
        ])
        
        if self.normalize_obs:
            obs = np.clip(obs, -1.0, 1.0)
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current state."""
        info = {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "inventory_position": self._state.inventory_position if self._state else 0,
            "total_inventory": self._state.total_inventory if self._state else 0,
            "backorders": self._state.backorders if self._state else 0,
        }
        
        if self._last_result is not None:
            info.update({
                "demand": self._last_result.demand_realized,
                "sales": self._last_result.sales,
                "spoiled": self._last_result.spoiled,
                "period_cost": self._last_result.costs.total_cost,
            })
        
        # Add cumulative metrics
        if self._cumulative_demand > 0:
            info["fill_rate"] = self._cumulative_sales / self._cumulative_demand
        else:
            info["fill_rate"] = 1.0
            
        return info
    
    def _action_to_orders(self, action: np.ndarray) -> Dict[int, float]:
        """Convert discrete action to order dictionary."""
        orders = {}
        for i, a in enumerate(action):
            orders[i] = float(a * self.order_step)
        return orders
    
    def _compute_reward(self, result: TransitionResult) -> float:
        """
        Compute shaped reward for the transition.
        
        The raw MDP reward is very negative (costs), which makes learning hard.
        We add shaping rewards to guide the agent:
        1. Base reward: negative of normalized cost
        2. Service reward: bonus for fulfilling demand
        3. Inventory reward: small bonus for maintaining good inventory position
        4. Spoilage penalty: additional penalty for spoilage
        """
        # Base reward: use the MDP cost but normalized
        base_cost = result.costs.total_cost
        
        if self.normalize_reward:
            # Normalize to roughly [-2, 0] range in typical operation
            normalized_cost = base_cost / self._reward_scale
            base_reward = -normalized_cost
        else:
            base_reward = result.reward
        
        if not self.reward_shaping:
            return base_reward
        
        # === SHAPING REWARDS ===
        shaped_reward = base_reward
        
        # 1. Service reward: big bonus for meeting demand
        if result.demand_realized > 0:
            fill_rate = result.sales / result.demand_realized
            # Reward for achieving target service level
            service_bonus = 1.0 * (fill_rate - 0.5)  # +0.5 for 100% fill, -0.5 for 0%
            shaped_reward += service_bonus
        
        # 2. Inventory position reward: small bonus for being near target
        ip = result.next_state.inventory_position
        ip_error = abs(ip - self.target_inventory_position) / self.target_inventory_position
        ip_reward = 0.2 * max(0, 1 - ip_error)  # 0 to 0.2 bonus
        shaped_reward += ip_reward
        
        # 3. Penalty for stockouts (additional to shortage cost)
        if result.new_backorders > 0:
            stockout_penalty = -0.5 * min(result.new_backorders / self.mean_demand, 1.0)
            shaped_reward += stockout_penalty
        
        # 4. Small penalty for excessive spoilage
        if result.spoiled > self.mean_demand * 0.1:  # More than 10% of demand spoiled
            spoilage_penalty = -0.3 * min(result.spoiled / self.mean_demand, 1.0)
            shaped_reward += spoilage_penalty
        
        return shaped_reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options:
                - 'initial_inventory': Override initial inventory
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Allow options to override initial inventory
        initial_inv = self._initial_inventory.copy()
        if options and 'initial_inventory' in options:
            initial_inv = options['initial_inventory']
        
        # Create initial state
        self._state = self.mdp.create_initial_state(
            initial_inventory=initial_inv
        )
        
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_result = None
        self._cumulative_demand = 0.0
        self._cumulative_sales = 0.0
        
        return self._get_obs(), self._get_info()
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: MultiDiscrete action (order quantities per supplier)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Convert action to order dictionary
        orders = self._action_to_orders(action)
        
        # Execute step in MDP
        result = self.mdp.step(self._state, orders)
        self._last_result = result
        
        # Update state
        self._state = result.next_state
        self._step_count += 1
        
        # Track cumulative metrics
        self._cumulative_demand += result.demand_realized
        self._cumulative_sales += result.sales
        
        # Compute shaped reward
        reward = self._compute_reward(result)
        self._episode_reward += reward
        
        # Check termination
        terminated = False  # No natural termination condition
        truncated = self._step_count >= self.max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self.render_mode is None:
            return None
        
        if self._state is None:
            return "Environment not initialized"
        
        lines = [
            f"Step: {self._step_count}",
            f"Inventory: {self._state.inventory.astype(int)}",
            f"Total Inventory: {self._state.total_inventory:.0f}",
            f"Backorders: {self._state.backorders:.0f}",
            f"Inventory Position: {self._state.inventory_position:.0f}",
        ]
        
        for sid, pipeline in self._state.pipelines.items():
            lines.append(f"Pipeline {sid}: {pipeline.pipeline.astype(int)} (total: {pipeline.total_in_pipeline():.0f})")
        
        if self._last_result:
            lines.extend([
                f"Last Demand: {self._last_result.demand_realized:.0f}",
                f"Last Sales: {self._last_result.sales:.0f}",
                f"Last Spoiled: {self._last_result.spoiled:.0f}",
                f"Last Cost: {self._last_result.costs.total_cost:.2f}",
            ])
        
        if self._cumulative_demand > 0:
            fill_rate = self._cumulative_sales / self._cumulative_demand
            lines.append(f"Episode Fill Rate: {fill_rate:.1%}")
        
        lines.append(f"Episode Reward: {self._episode_reward:.2f}")
        
        output = "\n".join(lines)
        
        if self.render_mode == "human":
            print(output)
        
        return output
    
    def close(self):
        """Clean up resources."""
        pass


class PerishableInventoryGymEnvContinuous(PerishableInventoryGymEnv):
    """
    Continuous action space version of the inventory environment.
    
    Uses Box action space where each dimension represents normalized
    order quantity [0, 1] that gets scaled to [0, max_order].
    
    Good for SAC, TD3, or PPO with continuous actions.
    """
    
    def __init__(self, **kwargs):
        # Remove order_step if provided (not used in continuous version)
        kwargs.pop('order_step', None)
        super().__init__(**kwargs)
        
        # Override action space to continuous
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_suppliers,),
            dtype=np.float32
        )
    
    def _action_to_orders(self, action: np.ndarray) -> Dict[int, float]:
        """Convert continuous action to order dictionary."""
        orders = {}
        for i, a in enumerate(action):
            # Scale from [0, 1] to [0, max_order], round to nearest integer
            orders[i] = float(round(np.clip(a, 0, 1) * self.max_order))
        return orders


def make_inventory_env(
    env_type: str = "discrete",
    **kwargs
) -> gym.Env:
    """
    Factory function to create inventory environments.
    
    Args:
        env_type: "discrete" for MultiDiscrete actions, "continuous" for Box actions
        **kwargs: Arguments passed to environment constructor
    
    Returns:
        Gymnasium environment
    
    Recommended settings for training:
        - Use normalize_reward=True and reward_shaping=True
        - Start with max_episode_steps=100, increase later
        - Use order_step=5 for discrete, gives 7 actions per supplier
    """
    if env_type == "discrete":
        return PerishableInventoryGymEnv(**kwargs)
    elif env_type == "continuous":
        return PerishableInventoryGymEnvContinuous(**kwargs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
