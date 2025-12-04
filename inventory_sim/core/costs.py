"""
Cost Parameters and Calculations for the Perishable Inventory MDP

Implements the cost structure:
- Purchase costs (unit + fixed)
- Holding costs (age-dependent)
- Shortage/backorder costs
- Spoilage costs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class CostParameters:
    """
    Cost parameters for the MDP.
    
    Attributes:
        holding_costs: Array h_n of holding costs per unit per expiry bucket
                      (can be age-dependent, higher for items near expiry)
        shortage_cost: Backorder penalty cost b per unit
        spoilage_cost: Wastage cost w per expired unit
        safety_penalty: Penalty weight η for violating safety threshold
        discount_factor: Discount factor γ for infinite-horizon MDP
    """
    holding_costs: np.ndarray
    shortage_cost: float = 10.0
    spoilage_cost: float = 5.0
    safety_penalty: float = 0.0
    discount_factor: float = 0.99
    
    def __post_init__(self):
        self.holding_costs = np.array(self.holding_costs, dtype=np.float64)
    
    @classmethod
    def uniform_holding(
        cls,
        shelf_life: int,
        holding_cost: float = 1.0,
        shortage_cost: float = 10.0,
        spoilage_cost: float = 5.0,
        discount_factor: float = 0.99
    ) -> 'CostParameters':
        """Create cost parameters with uniform holding cost across all buckets"""
        return cls(
            holding_costs=np.full(shelf_life, holding_cost),
            shortage_cost=shortage_cost,
            spoilage_cost=spoilage_cost,
            discount_factor=discount_factor
        )
    
    @classmethod
    def age_dependent_holding(
        cls,
        shelf_life: int,
        base_holding: float = 1.0,
        age_premium: float = 0.5,
        shortage_cost: float = 10.0,
        spoilage_cost: float = 5.0,
        discount_factor: float = 0.99
    ) -> 'CostParameters':
        """
        Create cost parameters with age-dependent holding costs.
        Older inventory (closer to expiry) has higher holding cost.
        
        h_n = base_holding + age_premium * (N - n) / N
        """
        holding_costs = np.array([
            base_holding + age_premium * (shelf_life - n) / shelf_life
            for n in range(1, shelf_life + 1)
        ])
        return cls(
            holding_costs=holding_costs,
            shortage_cost=shortage_cost,
            spoilage_cost=spoilage_cost,
            discount_factor=discount_factor
        )


@dataclass
class PeriodCosts:
    """
    Breakdown of costs incurred in a single period.
    
    c_t = C_t^purchase + C_t^hold + C_t^short + w * Spoiled_t
    """
    purchase_cost: float = 0.0
    fixed_order_cost: float = 0.0
    holding_cost: float = 0.0
    shortage_cost: float = 0.0
    spoilage_cost: float = 0.0
    safety_violation_cost: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total cost for the period"""
        return (
            self.purchase_cost +
            self.fixed_order_cost +
            self.holding_cost +
            self.shortage_cost +
            self.spoilage_cost +
            self.safety_violation_cost
        )
    
    @property
    def reward(self) -> float:
        """Reward (negative cost) for RL formulation"""
        return -self.total_cost
    
    def __add__(self, other: 'PeriodCosts') -> 'PeriodCosts':
        """Add two PeriodCosts together"""
        return PeriodCosts(
            purchase_cost=self.purchase_cost + other.purchase_cost,
            fixed_order_cost=self.fixed_order_cost + other.fixed_order_cost,
            holding_cost=self.holding_cost + other.holding_cost,
            shortage_cost=self.shortage_cost + other.shortage_cost,
            spoilage_cost=self.spoilage_cost + other.spoilage_cost,
            safety_violation_cost=self.safety_violation_cost + other.safety_violation_cost
        )


def calculate_purchase_costs(
    actions: Dict[int, float],
    pipelines: Dict[int, 'SupplierPipeline']
) -> PeriodCosts:
    """
    Calculate purchase costs for an action.
    
    C_t^purchase = Σ_s (v_s * a_t^(s) + K_s * 1_{a_t^(s) > 0})
    
    Args:
        actions: Dictionary {supplier_id: order_quantity}
        pipelines: Dictionary of supplier pipelines with cost info
    
    Returns:
        PeriodCosts with purchase and fixed costs filled in
    """
    purchase_cost = 0.0
    fixed_cost = 0.0
    
    for supplier_id, order_qty in actions.items():
        if order_qty > 0:
            pipeline = pipelines[supplier_id]
            purchase_cost += pipeline.unit_cost * order_qty
            fixed_cost += pipeline.fixed_cost
    
    return PeriodCosts(purchase_cost=purchase_cost, fixed_order_cost=fixed_cost)


def calculate_holding_cost(
    inventory: np.ndarray,
    holding_costs: np.ndarray
) -> float:
    """
    Calculate holding cost for current inventory.
    
    C_t^hold = Σ_{n=1}^N h_n * Î_t^(n)
    
    Args:
        inventory: Inventory by expiry bucket after serving demand
        holding_costs: Array of per-unit holding costs by bucket
    
    Returns:
        Total holding cost
    """
    return np.dot(holding_costs, inventory)


def calculate_shortage_cost(
    new_backorders: float,
    shortage_penalty: float
) -> float:
    """
    Calculate shortage/backorder cost.
    
    C_t^short = b * B_t^new
    
    Args:
        new_backorders: New backorders created this period
        shortage_penalty: Per-unit shortage cost b
    
    Returns:
        Total shortage cost
    """
    return shortage_penalty * new_backorders


def calculate_spoilage_cost(
    spoiled_qty: float,
    spoilage_penalty: float
) -> float:
    """
    Calculate spoilage/wastage cost.
    
    C_t^spoil = w * Spoiled_t
    
    Args:
        spoiled_qty: Quantity of inventory that expired
        spoilage_penalty: Per-unit spoilage cost w
    
    Returns:
        Total spoilage cost
    """
    return spoilage_penalty * spoiled_qty


def calculate_safety_violation_cost(
    inventory_position: float,
    safe_threshold: float,
    safety_penalty: float
) -> float:
    """
    Calculate cost of violating safety inventory threshold.
    
    C_t^safety = η * max(0, S_t^safe - IP_t)
    
    Args:
        inventory_position: Current inventory position
        safe_threshold: Safety threshold S_t^safe
        safety_penalty: Penalty weight η
    
    Returns:
        Safety violation cost
    """
    violation = max(0, safe_threshold - inventory_position)
    return safety_penalty * violation


def calculate_safe_threshold(
    mean_demand: float,
    std_demand: float,
    service_level: float = 0.95,
    horizon: int = 1
) -> float:
    """
    Calculate safe inventory threshold.
    
    S_t^safe = μ_{t:H} + z_α * σ_{t:H}
    
    Args:
        mean_demand: Forecasted mean cumulative demand μ_{t:H}
        std_demand: Standard deviation of cumulative demand σ_{t:H}
        service_level: Target service level α (default 0.95)
        horizon: Forecast horizon H
    
    Returns:
        Safe inventory threshold S_t^safe
    """
    from scipy import stats
    z_alpha = stats.norm.ppf(service_level)
    return mean_demand + z_alpha * std_demand

