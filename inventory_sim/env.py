"""
Perishable Inventory MDP Environment

Implements the complete MDP environment with the sequence of events:
1. Arrivals
2. Serve demand (FIFO)
3. Calculate costs
4. Aging and spoilage
5. Pipeline shifts and new orders
6. Backorder update
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy

from .core.state import InventoryState, SupplierPipeline, create_state_from_config
from .core.demand import DemandProcess, PoissonDemand, StochasticLeadTime
from .core.costs import (
    CostParameters, PeriodCosts,
    calculate_purchase_costs, calculate_holding_cost,
    calculate_shortage_cost, calculate_spoilage_cost,
    calculate_safety_violation_cost, calculate_safe_threshold
)
from .interfaces import InventoryEnvironment
from .exceptions import (
    ActionValidationError, SupplierNotFoundError,
    CapacityViolationError, MOQViolationError, InvalidParameterError
)



@dataclass
class TransitionResult:
    """
    Result of a state transition.
    
    Contains the new state, costs, and diagnostic information.
    """
    next_state: InventoryState
    costs: PeriodCosts
    demand_realized: float
    sales: float
    new_backorders: float
    spoiled: float
    arrivals: float
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def reward(self) -> float:
        """Reward (negative cost) for RL"""
        return self.costs.reward


class PerishableInventoryMDP(InventoryEnvironment):
    """
    Complete MDP environment for perishable inventory management.
    
    Implements the Bellman equation:
    V(X) = max_{a∈A(X)} { -c(X,a) + γ * E[V(X') | X, a] }
    
    Attributes:
        shelf_life: Number of expiry buckets N
        demand_process: Stochastic demand generator
        cost_params: Cost parameters
        suppliers: List of supplier configurations
        stochastic_lead_times: Optional dict of StochasticLeadTime per supplier
        lost_sales: If True, use lost-sales model; if False, use backorders
    """
    
    def __init__(
        self,
        shelf_life: int,
        suppliers: List[Dict],
        demand_process: DemandProcess,
        cost_params: CostParameters,
        stochastic_lead_times: Optional[Dict[int, StochasticLeadTime]] = None,
        lost_sales: bool = False
    ):
        self.shelf_life = shelf_life
        self.suppliers = suppliers
        self.demand_process = demand_process
        self.cost_params = cost_params
        self.stochastic_lead_times = stochastic_lead_times or {}
        self.lost_sales = lost_sales
        
        # Validate shelf life matches cost parameters
        if len(cost_params.holding_costs) != shelf_life:
            raise ValueError(
                f"Holding costs length {len(cost_params.holding_costs)} != shelf_life {shelf_life}"
            )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> InventoryState:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Dictionary containing initial state configuration:
                - initial_inventory
                - initial_backorders
                - initial_exogenous
        
        Returns:
            Initial InventoryState
        """
        if seed is not None:
            np.random.seed(seed)
            
        options = options or {}
        
        return create_state_from_config(
            shelf_life=self.shelf_life,
            suppliers=self.suppliers,
            initial_inventory=options.get('initial_inventory'),
            initial_backorders=options.get('initial_backorders', 0.0),
            initial_exogenous=options.get('initial_exogenous')
        )

    def create_initial_state(
        self,
        initial_inventory: Optional[np.ndarray] = None,
        initial_backorders: float = 0.0,
        initial_exogenous: Optional[np.ndarray] = None
    ) -> InventoryState:
        """
        Create an initial state for the MDP.
        
        Deprecated: Use reset() instead.
        """
        return self.reset(options={
            'initial_inventory': initial_inventory,
            'initial_backorders': initial_backorders,
            'initial_exogenous': initial_exogenous
        })
    
    def get_feasible_actions(self, state: InventoryState) -> List[Dict[int, float]]:
        """
        Get all feasible actions from a state.
        
        Constraints:
        - a_t^(s) ≤ U_s (capacity)
        - a_t^(s) ∈ M_s * Z_≥0 (MOQ multiples)
        
        For simplicity, returns a list of discrete action combinations.
        """
        feasible = [{}]  # Empty action (hold off) is always feasible
        
        # Get all supplier action combinations
        for supplier_id, pipeline in state.pipelines.items():
            new_feasible = []
            max_qty = min(pipeline.capacity, 100)  # Cap for discretization
            moq = pipeline.moq
            
            for action in feasible:
                # Add zero order option
                new_action = action.copy()
                new_action[supplier_id] = 0.0
                new_feasible.append(new_action)
                
                # Add positive order options (MOQ multiples)
                qty = moq
                while qty <= max_qty:
                    new_action = action.copy()
                    new_action[supplier_id] = float(qty)
                    new_feasible.append(new_action)
                    qty += moq
            
            feasible = new_feasible
        
        return feasible
    
    def is_action_feasible(self, state: InventoryState, action: Dict[int, float]) -> bool:
        """
        Check if an action is feasible given the current state.
        
        Returns True if feasible, False otherwise (does not raise exceptions).
        For detailed validation with exceptions, use validate_action().
        """
        try:
            self.validate_action(state, action)
            return True
        except ActionValidationError:
            return False
    
    def validate_action(self, state: InventoryState, action: Dict[int, float]) -> None:
        """
        Validate an action and raise descriptive exceptions if invalid.
        
        Args:
            state: Current inventory state
            action: Action to validate
        
        Raises:
            SupplierNotFoundError: If action references unknown supplier
            CapacityViolationError: If order exceeds capacity
            MOQViolationError: If order violates MOQ constraints
            InvalidParameterError: If order quantity is negative
        """
        for supplier_id, order_qty in action.items():
            # Check supplier exists
            if supplier_id not in state.pipelines:
                available = list(state.pipelines.keys())
                raise SupplierNotFoundError(supplier_id, available)
            
            pipeline = state.pipelines[supplier_id]
            
            # Check for negative orders
            if order_qty < 0:
                raise InvalidParameterError(
                    f"Order quantity must be non-negative for supplier {supplier_id}, got {order_qty}"
                )
            
            # Check capacity
            if order_qty > pipeline.capacity:
                raise CapacityViolationError(supplier_id, order_qty, pipeline.capacity)
            
            # Check MOQ (must be 0 or multiple of MOQ)
            if order_qty > 0 and order_qty % pipeline.moq != 0:
                raise MOQViolationError(supplier_id, order_qty, pipeline.moq)

    
    def step(
        self,
        state: InventoryState,
        action: Dict[int, float],
        demand: Optional[float] = None
    ) -> TransitionResult:
        """
        Execute one step of the MDP.
        
        Implements the sequence of events:
        1. Arrivals
        2. Serve demand (FIFO)
        3. Calculate costs
        4. Aging and spoilage
        5. Pipeline shifts and new orders
        6. Backorder update
        
        Args:
            state: Current state X_t
            action: Order action a_t = {supplier_id: quantity}
            demand: Optional fixed demand (if None, sampled from process)
        
        Returns:
            TransitionResult containing next state, costs, and diagnostics
        """
        # Validate action before proceeding
        self.validate_action(state, action)
        
        # Create copy to avoid mutating original state
        next_state = state.copy()
        
        # Sample demand if not provided
        if demand is None:
            demand = self.demand_process.sample(state.exogenous_state)
        
        # ========== 1. ARRIVALS ==========
        # A_t = Σ_{s∈S} (P_t^(s,1) + P̃_t^(s,1))
        arrivals = next_state.get_arriving_inventory()
        
        # I_t^(N) ← I_t^(N) + A_t
        next_state.add_arrivals(arrivals)
        
        # ========== 2. SERVE DEMAND (FIFO) ==========
        # Also serve any existing backorders first if we have inventory
        total_demand = demand
        if not self.lost_sales:
            # In backorder model, try to fulfill backorders first
            total_demand += next_state.backorders
        
        sales, new_backorders = next_state.serve_demand_fifo(total_demand)
        
        # Snapshot inventory for holding cost calculation
        inventory_snapshot = next_state.inventory.copy()
        
        # ========== 3. CALCULATE COSTS ==========
        costs = PeriodCosts()
        
        # Purchase costs: C_t^purchase = Σ_s (v_s * a_t^(s) + K_s * 1_{a > 0})
        costs += calculate_purchase_costs(action, next_state.pipelines)
        
        # Holding costs: C_t^hold = Σ_n h_n * Î_t^(n)
        costs.holding_cost = calculate_holding_cost(
            inventory_snapshot,
            self.cost_params.holding_costs
        )
        
        # Shortage costs: C_t^short = b * B_t^new
        costs.shortage_cost = calculate_shortage_cost(
            new_backorders,
            self.cost_params.shortage_cost
        )
        
        # ========== 4. AGING AND SPOILAGE ==========
        # Spoiled_t = I_t^(1)
        # I_{t+1}^(n) = I_t^(n+1), I_{t+1}^(N) = 0
        spoiled = next_state.age_inventory()
        
        # Add spoilage cost: c_t ← c_t + w * Spoiled_t
        costs.spoilage_cost = calculate_spoilage_cost(
            spoiled,
            self.cost_params.spoilage_cost
        )
        
        # ========== 5. PIPELINE SHIFTS AND NEW ORDERS ==========
        for supplier_id, pipeline in next_state.pipelines.items():
            order_qty = action.get(supplier_id, 0.0)
            
            # Handle stochastic lead times
            if supplier_id in self.stochastic_lead_times:
                slt = self.stochastic_lead_times[supplier_id]
                if not slt.sample_advancement():
                    # Pipeline doesn't advance - just add new order
                    # This is simplified; full implementation would shift partially
                    order_qty = action.get(supplier_id, 0.0)
            
            # P_{t+1}^(s,ℓ) = P_t^(s,ℓ+1), P_{t+1}^(s,L_s) = a_t^(s)
            pipeline.shift_and_add_order(order_qty)
            
            # P̃_{t+1}^(s,ℓ) = P̃_t^(s,ℓ+1), P̃_{t+1}^(s,L_s) = 0
            pipeline.shift_scheduled()
        
        # ========== 6. BACKORDER UPDATE ==========
        if self.lost_sales:
            # Lost sales model: B_t = 0 always
            next_state.backorders = 0.0
        else:
            # Backorder model: B_{t+1} = B_t^new
            # (Existing backorders were included in total_demand and potentially fulfilled)
            next_state.backorders = new_backorders
        
        # Update time step
        next_state.time_step = state.time_step + 1
        
        # Update exogenous state
        next_state.exogenous_state = self.demand_process.update_exogenous_state(
            state.exogenous_state
        )
        
        return TransitionResult(
            next_state=next_state,
            costs=costs,
            demand_realized=demand,
            sales=sales,
            new_backorders=new_backorders,
            spoiled=spoiled,
            arrivals=arrivals,
            info={
                "inventory_position": next_state.inventory_position,
                "total_inventory": next_state.total_inventory
            }
        )
    
    # simulate_episode removed. Use simulation.run_episode instead.
    
    def expected_cost(
        self,
        state: InventoryState,
        action: Dict[int, float],
        num_samples: int = 100
    ) -> float:
        """
        Estimate expected one-step cost via Monte Carlo.
        
        E[c(X, a)] ≈ (1/N) * Σ_i c(X, a, D_i)
        """
        total_cost = 0.0
        for _ in range(num_samples):
            result = self.step(state.copy(), action)
            total_cost += result.costs.total_cost
        return total_cost / num_samples
    
    def compute_inventory_metrics(
        self,
        results: List[TransitionResult]
    ) -> Dict[str, float]:
        """
        Compute performance metrics from simulation results.
        
        Returns:
            Dictionary of metrics including:
            - fill_rate: Fraction of demand fulfilled
            - spoilage_rate: Fraction of inventory spoiled
            - average_inventory: Average inventory level
            - service_level: Probability of no stockout
        """
        total_demand = sum(r.demand_realized for r in results)
        total_sales = sum(r.sales for r in results)
        total_spoiled = sum(r.spoiled for r in results)
        total_arrivals = sum(r.arrivals for r in results)
        
        fill_rate = total_sales / total_demand if total_demand > 0 else 1.0
        spoilage_rate = total_spoiled / total_arrivals if total_arrivals > 0 else 0.0
        
        avg_inventory = np.mean([r.next_state.total_inventory for r in results])
        
        stockout_periods = sum(1 for r in results if r.new_backorders > 0)
        service_level = 1 - stockout_periods / len(results)
        
        total_cost = sum(r.costs.total_cost for r in results)
        avg_cost = total_cost / len(results)
        
        return {
            "fill_rate": fill_rate,
            "spoilage_rate": spoilage_rate,
            "average_inventory": avg_inventory,
            "service_level": service_level,
            "total_cost": total_cost,
            "average_cost": avg_cost,
            "total_demand": total_demand,
            "total_sales": total_sales,
            "total_spoiled": total_spoiled
        }


def create_simple_mdp(
    shelf_life: int = 5,
    num_suppliers: int = 2,
    mean_demand: float = 10.0,
    fast_lead_time: int = 1,
    slow_lead_time: int = 3,
    fast_cost: float = 2.0,
    slow_cost: float = 1.0
) -> PerishableInventoryMDP:
    """
    Factory function to create a simple two-supplier MDP.
    
    Creates a typical TBS (Tailored Base-Surge) scenario with:
    - Fast, expensive supplier
    - Slow, cheap supplier
    """
    suppliers = [
        {
            "id": 0,
            "lead_time": fast_lead_time,
            "unit_cost": fast_cost,
            "fixed_cost": 0.0,
            "capacity": 100,
            "moq": 1
        },
        {
            "id": 1,
            "lead_time": slow_lead_time,
            "unit_cost": slow_cost,
            "fixed_cost": 0.0,
            "capacity": 100,
            "moq": 1
        }
    ]
    
    if num_suppliers == 1:
        suppliers = [suppliers[0]]
    
    demand_process = PoissonDemand(mean_demand)
    cost_params = CostParameters.age_dependent_holding(
        shelf_life=shelf_life,
        base_holding=0.5,
        age_premium=0.5,
        shortage_cost=10.0,
        spoilage_cost=5.0
    )
    
    return PerishableInventoryMDP(
        shelf_life=shelf_life,
        suppliers=suppliers,
        demand_process=demand_process,
        cost_params=cost_params
    )

