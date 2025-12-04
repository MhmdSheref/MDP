"""
Simulation runner for Perishable Inventory MDP.

Separates the simulation loop from the environment and agent implementations.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from .interfaces import InventoryEnvironment, InventoryAgent
from .env import TransitionResult
from .core.state import InventoryState

def run_episode(
    env: InventoryEnvironment,
    agent: InventoryAgent,
    num_periods: int,
    seed: Optional[int] = None,
    initial_state: Optional[InventoryState] = None
) -> Tuple[List[TransitionResult], float]:
    """
    Run a single simulation episode.
    
    Args:
        env: The inventory environment
        agent: The agent/policy to evaluate
        num_periods: Number of periods to simulate
        seed: Random seed for reproducibility
        initial_state: Optional starting state (if None, env.reset() is used)
        
    Returns:
        Tuple of (list of TransitionResults, total discounted reward)
    """
    if seed is not None:
        np.random.seed(seed)
        
    if initial_state is None:
        state = env.reset(seed=seed)
    else:
        state = initial_state.copy()
        
    results = []
    total_reward = 0.0
    discount = 1.0
    
    # Get discount factor if available (specific to PerishableInventoryMDP)
    if hasattr(env, 'cost_params'):
        discount_factor = env.cost_params.discount_factor
    else:
        discount_factor = 1.0
    
    for _ in range(num_periods):
        # Agent selects action
        action = agent.act(state, env)
        
        # Environment executes step
        result = env.step(state, action)
        results.append(result)
        
        # Accumulate reward
        # Check if result has reward property, otherwise assume 0 or extract from info
        reward = getattr(result, 'reward', 0.0)
        total_reward += discount * reward
        
        discount *= discount_factor
        
        # Update state
        state = result.next_state
        
    return results, total_reward
