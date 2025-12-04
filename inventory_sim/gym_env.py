"""
Gymnasium-compatible wrapper for the Perishable Inventory MDP.

This module provides a Gym environment that can be used with standard
RL libraries like Stable Baselines3 for training agents.
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
    
    Observation Space:
        - Inventory by expiry bucket (shelf_life floats)
        - Pipeline totals per supplier (num_suppliers floats)
        - Backorders (1 float)
        - Inventory position (1 float)
    
    Action Space:
        MultiDiscrete: Order quantities for each supplier (0 to max_order)
        Each action dimension represents units to order from that supplier
    
    Reward:
        Negative of total period costs (purchase + holding + shortage + spoilage)
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
            initial_inventory: Initial inventory levels (if None, starts empty)
            normalize_obs: Whether to normalize observations to [0, 1]
            render_mode: Render mode ("human" or "ansi")
        """
        super().__init__()
        
        self.shelf_life = shelf_life
        self.num_suppliers = num_suppliers
        self.max_order = max_order_per_supplier
        self.order_step = order_step
        self.max_episode_steps = max_episode_steps
        self.normalize_obs = normalize_obs
        self.render_mode = render_mode
        
        # Store initial inventory for reset
        if initial_inventory is not None:
            self._initial_inventory = initial_inventory.copy()
        else:
            # Default: distributed inventory across buckets
            self._initial_inventory = np.full(shelf_life, mean_demand * 2)
        
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
        # [inventory_buckets, pipeline_totals, backorders, inventory_position]
        obs_dim = shelf_life + num_suppliers + 2
        
        if normalize_obs:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
        
        # Normalization constants
        self._obs_max = np.array(
            [100.0] * shelf_life +  # inventory buckets
            [200.0] * num_suppliers +  # pipeline totals
            [50.0] +  # backorders
            [300.0]   # inventory position
        )
        
        # State tracking
        self._state: Optional[InventoryState] = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_result: Optional[TransitionResult] = None
    
    def _get_obs(self) -> np.ndarray:
        """Convert InventoryState to observation array."""
        if self._state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Inventory by bucket
        inventory = self._state.inventory.copy()
        
        # Pipeline totals per supplier
        pipeline_totals = np.array([
            p.total_in_pipeline() 
            for p in self._state.pipelines.values()
        ])
        
        # Combine into observation
        obs = np.concatenate([
            inventory,
            pipeline_totals,
            [self._state.backorders],
            [self._state.inventory_position]
        ])
        
        if self.normalize_obs:
            obs = np.clip(obs / self._obs_max, 0.0, 1.0)
        
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
        
        return info
    
    def _action_to_orders(self, action: np.ndarray) -> Dict[int, float]:
        """Convert discrete action to order dictionary."""
        orders = {}
        for i, a in enumerate(action):
            orders[i] = float(a * self.order_step)
        return orders
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used)
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create initial state
        self._state = self.mdp.create_initial_state(
            initial_inventory=self._initial_inventory.copy()
        )
        
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_result = None
        
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
        
        # Get reward (negative cost)
        reward = result.reward
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
            orders[i] = float(round(a * self.max_order))
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
    """
    if env_type == "discrete":
        return PerishableInventoryGymEnv(**kwargs)
    elif env_type == "continuous":
        return PerishableInventoryGymEnvContinuous(**kwargs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
