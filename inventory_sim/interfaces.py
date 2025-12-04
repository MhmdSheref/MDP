"""
Interfaces for the Perishable Inventory MDP.

Defines the contract between the Environment and Agents/Solvers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .core.state import InventoryState


class InventoryEnvironment(ABC):
    """
    Abstract base class for Inventory Environments.
    """
    
    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> InventoryState:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional configuration options
            
        Returns:
            Initial InventoryState
        """
        pass
    
    @abstractmethod
    def step(
        self,
        state: InventoryState,
        action: Any,
        **kwargs
    ) -> Any:
        """
        Execute one step of the environment.
        
        Args:
            state: Current state
            action: Action to take
            **kwargs: Additional arguments (e.g. fixed demand for testing)
            
        Returns:
            Transition result (next_state, reward, done, info) or similar object
        """
        pass
    
    @abstractmethod
    def get_feasible_actions(self, state: InventoryState) -> List[Any]:
        """Get list of feasible actions from the current state."""
        pass


class InventoryAgent(ABC):
    """
    Abstract base class for Inventory Agents/Policies.
    """
    
    @abstractmethod
    def act(
        self,
        state: InventoryState,
        env: InventoryEnvironment
    ) -> Any:
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            env: The environment (for accessing model parameters if needed)
            
        Returns:
            Selected action
        """
        pass
