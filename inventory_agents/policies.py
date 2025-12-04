"""
Policies for the Perishable Inventory MDP

Implements various ordering policies including:
- Base-stock (order-up-to) policies
- Tailored Base-Surge (TBS) policies for two suppliers
- Myopic policies
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inventory_sim.core.state import InventoryState
    from inventory_sim.env import PerishableInventoryMDP
from inventory_sim.interfaces import InventoryAgent, InventoryEnvironment
from inventory_sim.exceptions import InvalidParameterError, SupplierNotFoundError


class BasePolicy(InventoryAgent):
    """
    Abstract base class for ordering policies.
    
    A policy π: X → A maps states to actions.
    """
    
    @abstractmethod
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        """
        Get the action to take in the given state.
        
        Args:
            state: Current inventory state X_t
            mdp: The MDP environment
        
        Returns:
            Action dictionary {supplier_id: order_quantity}
        """
        pass
    
    def act(
        self,
        state: 'InventoryState',
        env: 'InventoryEnvironment'
    ) -> Dict[int, float]:
        """Alias for get_action to satisfy InventoryAgent interface"""
        return self.get_action(state, env)  # type: ignore

    def __call__(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        return self.get_action(state, mdp)


class DoNothingPolicy(BasePolicy):
    """Policy that never orders anything"""
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        return {s: 0.0 for s in state.pipelines.keys()}


class ConstantOrderPolicy(BasePolicy):
    """
    Policy that orders a constant amount each period.
    
    Useful for baseline comparisons.
    """
    
    def __init__(self, order_quantities: Dict[int, float]):
        self.order_quantities = order_quantities
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        return self.order_quantities.copy()


class BaseStockPolicy(BasePolicy):
    """
    Base-stock (order-up-to) policy.
    
    Orders to bring inventory position up to target level S*.
    
    a_t = max(0, S* - IP_t)
    
    where IP_t is the inventory position (on-hand + pipeline - backorders).
    
    Attributes:
        target_level: Order-up-to level S*
        supplier_id: Supplier to order from (if multiple suppliers)
    """
    
    def __init__(
        self,
        target_level: float,
        supplier_id: int = 0,
        respect_moq: bool = True
    ):
        if target_level < 0:
            raise InvalidParameterError(f"Target level must be non-negative, got {target_level}")
        self.target_level = target_level
        self.supplier_id = supplier_id
        self.respect_moq = respect_moq
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        # Calculate order-up-to quantity
        inventory_position = state.inventory_position
        order_qty = max(0, self.target_level - inventory_position)
        
        # Respect capacity first
        if self.supplier_id in state.pipelines:
            capacity = state.pipelines[self.supplier_id].capacity
            order_qty = min(order_qty, capacity)
        
        # Then apply MOQ rounding
        if self.respect_moq and self.supplier_id in state.pipelines:
            moq = state.pipelines[self.supplier_id].moq
            if order_qty > 0 and order_qty < moq:
                # If order is positive but less than MOQ, round up to MOQ
                order_qty = moq
                # Re-check capacity after rounding up
                if self.supplier_id in state.pipelines:
                    capacity = state.pipelines[self.supplier_id].capacity
                    if order_qty > capacity:
                        # Can't meet MOQ given capacity - order 0
                        order_qty = 0.0
            elif order_qty > 0:
                # Round up to nearest MOQ multiple
                order_qty = float(np.ceil(order_qty / moq) * moq)
                # Re-check capacity after rounding
                if self.supplier_id in state.pipelines:
                    capacity = state.pipelines[self.supplier_id].capacity
                    if order_qty > capacity:
                        # Round down to largest MOQ multiple that fits
                        order_qty = float(np.floor(capacity / moq) * moq)
        
        # Validate supplier exists
        if self.supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.supplier_id, available)
        
        # Return action for all suppliers (0 for non-target suppliers)
        action = {s: 0.0 for s in state.pipelines.keys()}
        action[self.supplier_id] = order_qty
        return action


class MultiSupplierBaseStockPolicy(BasePolicy):
    """
    Base-stock policy that allocates orders across multiple suppliers.
    
    Uses "cheapest effective arrival" logic:
    Orders from suppliers in order of increasing effective cost,
    considering lead time and expected spoilage.
    """
    
    def __init__(
        self,
        target_level: float,
        allocation_strategy: str = "cheapest_first"
    ):
        if target_level < 0:
            raise InvalidParameterError(f"Target level must be non-negative, got {target_level}")
        self.target_level = target_level
        self.allocation_strategy = allocation_strategy
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        inventory_position = state.inventory_position
        order_qty = max(0, self.target_level - inventory_position)
        
        if order_qty <= 0:
            return {s: 0.0 for s in state.pipelines.keys()}
        
        # Sort suppliers by cost
        sorted_suppliers = sorted(
            state.pipelines.items(),
            key=lambda x: x[1].unit_cost
        )
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        remaining = order_qty
        
        for supplier_id, pipeline in sorted_suppliers:
            if remaining <= 0:
                break
            
            # Allocate up to capacity
            alloc = min(remaining, pipeline.capacity)
            
            # Round to MOQ
            moq = pipeline.moq
            if alloc > 0 and alloc < moq:
                alloc = moq
            elif alloc > 0:
                alloc = np.ceil(alloc / moq) * moq
            
            action[supplier_id] = alloc
            remaining -= alloc
        
        return action


@dataclass
class TailoredBaseSurgePolicy(BasePolicy):
    """
    Tailored Base-Surge (TBS) policy for two suppliers.
    
    Allocates base demand to slow (cheap) supplier and 
    surge demand to fast (expensive) supplier.
    
    - Slow supplier: orders to meet base-stock level S_slow
    - Fast supplier: surge orders when inventory falls below reorder point r
    
    Under certain conditions (large cost/lead-time differences),
    TBS is optimal or near-optimal.
    
    Attributes:
        slow_supplier_id: ID of slow (cheap) supplier
        fast_supplier_id: ID of fast (expensive) supplier
        base_stock_level: Target level S* for slow supplier
        reorder_point: Reorder point r for fast supplier
        max_surge: Maximum surge order quantity
    """
    slow_supplier_id: int
    fast_supplier_id: int
    base_stock_level: float
    reorder_point: float
    max_surge: float = float('inf')
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        # Validate suppliers exist
        if self.slow_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.slow_supplier_id, available)
        if self.fast_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.fast_supplier_id, available)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        
        # Calculate relevant positions
        on_hand = state.total_inventory
        pipeline_slow = state.pipelines[self.slow_supplier_id].total_in_pipeline()
        pipeline_fast = state.pipelines[self.fast_supplier_id].total_in_pipeline()
        
        # Inventory position including only slow supplier pipeline
        ip_slow = on_hand + pipeline_slow - state.backorders
        
        # Total inventory position
        ip_total = state.inventory_position
        
        # Base order to slow supplier: order up to S*
        base_order = max(0, self.base_stock_level - ip_slow)
        
        # Apply MOQ and capacity for slow supplier
        slow_pipeline = state.pipelines[self.slow_supplier_id]
        if base_order > 0:
            moq = slow_pipeline.moq
            if base_order < moq:
                base_order = moq
            else:
                base_order = np.ceil(base_order / moq) * moq
            base_order = min(base_order, slow_pipeline.capacity)
        
        action[self.slow_supplier_id] = base_order
        
        # Surge order to fast supplier if inventory is low
        if ip_total < self.reorder_point:
            surge_order = min(
                self.reorder_point - ip_total,
                self.max_surge
            )
            
            # Apply MOQ and capacity for fast supplier
            fast_pipeline = state.pipelines[self.fast_supplier_id]
            if surge_order > 0:
                moq = fast_pipeline.moq
                if surge_order < moq:
                    surge_order = moq
                else:
                    surge_order = np.ceil(surge_order / moq) * moq
                surge_order = min(surge_order, fast_pipeline.capacity)
            
            action[self.fast_supplier_id] = surge_order
        
        return action
    
    @classmethod
    def from_demand_forecast(
        cls,
        slow_supplier_id: int,
        fast_supplier_id: int,
        mean_demand: float,
        std_demand: float,
        slow_lead_time: int,
        fast_lead_time: int,
        service_level: float = 0.95
    ) -> 'TailoredBaseSurgePolicy':
        """
        Create TBS policy from demand forecast.
        
        Sets:
        - Base stock level = slow lead time demand + safety stock
        - Reorder point = fast lead time demand + safety stock
        """
        from scipy import stats
        z_alpha = stats.norm.ppf(service_level)
        
        # Base stock for slow supplier
        slow_demand_mean = mean_demand * slow_lead_time
        slow_demand_std = std_demand * np.sqrt(slow_lead_time)
        base_stock = slow_demand_mean + z_alpha * slow_demand_std
        
        # Reorder point for fast supplier
        fast_demand_mean = mean_demand * fast_lead_time
        fast_demand_std = std_demand * np.sqrt(fast_lead_time)
        reorder_point = fast_demand_mean + z_alpha * fast_demand_std
        
        return cls(
            slow_supplier_id=slow_supplier_id,
            fast_supplier_id=fast_supplier_id,
            base_stock_level=base_stock,
            reorder_point=reorder_point
        )


class MyopicPolicy(BasePolicy):
    """
    Myopic (one-step lookahead) policy.
    
    At each step, chooses the action that minimizes expected
    one-step cost without considering future costs.
    
    Useful as a baseline and for verifying optimal policy structure.
    """
    
    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        feasible_actions = mdp.get_feasible_actions(state)
        
        best_action = feasible_actions[0]
        best_cost = float('inf')
        
        for action in feasible_actions:
            expected_cost = mdp.expected_cost(state, action, self.num_samples)
            if expected_cost < best_cost:
                best_cost = expected_cost
                best_action = action
        
        return best_action


class SurvivalAdjustedPolicy(BasePolicy):
    """
    Policy that uses survival-adjusted inventory position.
    
    Accounts for probability that inventory will be consumed
    before expiry, ordering more aggressively for perishable items.
    
    IP_t^surv = Σ_n ρ_n * I_t^(n)
    """
    
    def __init__(
        self,
        target_level: float,
        survival_probs: np.ndarray,
        supplier_id: int = 0
    ):
        self.target_level = target_level
        self.survival_probs = survival_probs
        self.supplier_id = supplier_id
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        # Use survival-adjusted inventory position
        ip_surv = state.survival_adjusted_inventory_position(self.survival_probs)
        
        # Add pipeline (assume all pipeline will arrive fresh)
        ip_surv += state.total_pipeline - state.backorders
        
        order_qty = max(0, self.target_level - ip_surv)
        
        # Apply constraints
        if self.supplier_id in state.pipelines:
            pipeline = state.pipelines[self.supplier_id]
            moq = pipeline.moq
            if order_qty > 0 and order_qty < moq:
                order_qty = moq
            elif order_qty > 0:
                order_qty = np.ceil(order_qty / moq) * moq
            order_qty = min(order_qty, pipeline.capacity)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        action[self.supplier_id] = order_qty
        return action
    
    @classmethod
    def compute_survival_probs(
        cls,
        shelf_life: int,
        mean_demand: float,
        inventory_level: float
    ) -> np.ndarray:
        """
        Compute survival probabilities ρ_n.
        
        Probability that inventory with n periods remaining
        will be consumed before expiry.
        
        Simple approximation: ρ_n = min(1, n * mean_demand / inventory_level)
        """
        if inventory_level <= 0:
            return np.ones(shelf_life)
        
        probs = np.array([
            min(1.0, n * mean_demand / inventory_level)
            for n in range(1, shelf_life + 1)
        ])
        return probs

