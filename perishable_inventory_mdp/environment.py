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

from .state import InventoryState, SupplierPipeline, create_state_from_config
from .demand import DemandProcess, PoissonDemand, StochasticLeadTime
from .costs import (
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


class EnhancedPerishableInventoryMDP(PerishableInventoryMDP):
    """
    Enhanced MDP with supplier rejection and crisis support.
    
    Extends PerishableInventoryMDP to add:
    - Supplier order rejection based on crisis state
    - Crisis state transitions during simulation
    - Contract support for discounts/penalties
    
    Attributes:
        crisis_process: Optional CrisisProcess for crisis state management
        rejection_probs: Dict of base rejection probabilities per supplier
    """
    
    def __init__(
        self,
        shelf_life: int,
        suppliers: List[Dict],
        demand_process: DemandProcess,
        cost_params: CostParameters,
        stochastic_lead_times: Optional[Dict[int, StochasticLeadTime]] = None,
        lost_sales: bool = False,
        crisis_process: Optional['CrisisProcess'] = None,
        enable_rejection: bool = False
    ):
        super().__init__(
            shelf_life=shelf_life,
            suppliers=suppliers,
            demand_process=demand_process,
            cost_params=cost_params,
            stochastic_lead_times=stochastic_lead_times,
            lost_sales=lost_sales
        )
        
        self.crisis_process = crisis_process
        self.enable_rejection = enable_rejection
        
        # Extract rejection probabilities from suppliers
        self.rejection_probs = {}
        for supplier in suppliers:
            sid = supplier['id']
            self.rejection_probs[sid] = supplier.get('rejection_prob', 0.0)
    
    def _should_reject_order(self, supplier_id: int) -> bool:
        """
        Determine if a supplier order should be rejected.
        
        Rejection probability combines base supplier probability with
        crisis-induced disruption.
        """
        if not self.enable_rejection:
            return False
        
        base_prob = self.rejection_probs.get(supplier_id, 0.0)
        
        if self.crisis_process is not None:
            # Crisis increases rejection probability
            crisis_prob = self.crisis_process.get_supply_disruption_prob()
            # Combined probability: P(reject) = base + crisis - base*crisis
            combined = base_prob + crisis_prob - base_prob * crisis_prob
            combined = min(0.95, combined)  # Cap at 95%
            return np.random.random() < combined
        
        return np.random.random() < base_prob
    
    def step(
        self,
        state: InventoryState,
        action: Dict[int, float],
        demand: Optional[float] = None
    ) -> TransitionResult:
        """
        Execute one step with enhanced features.
        
        Extends base step() to:
        1. Apply supplier rejection to orders
        2. Transition crisis state
        3. Modulate demand based on crisis
        """
        # Update crisis state if process exists
        if self.crisis_process is not None:
            self.crisis_process.sample_transition()
            
            # Apply crisis demand multiplier if using CompositeDemand
            if state.exogenous_state is not None and len(state.exogenous_state) >= 2:
                state.exogenous_state[1] = float(self.crisis_process.current_state)
        
        # Apply rejection logic to modify action
        effective_action = {}
        rejected_orders = {}
        
        for supplier_id, order_qty in action.items():
            if order_qty > 0 and self._should_reject_order(supplier_id):
                rejected_orders[supplier_id] = order_qty
                effective_action[supplier_id] = 0.0
            else:
                effective_action[supplier_id] = order_qty
        
        # Call parent step with effective action
        result = super().step(state, effective_action, demand)
        
        # Track rejected orders in result (could extend TransitionResult later)
        # For now, add to costs as "emergency sourcing cost" if needed
        if rejected_orders:
            # Could add rejection penalty here
            pass
        
        return result
    
    def create_initial_state(
        self,
        initial_inventory: Optional[np.ndarray] = None,
        initial_backorders: float = 0.0
    ) -> InventoryState:
        """Create initial state with crisis initialization."""
        state = super().create_initial_state(initial_inventory, initial_backorders)
        
        # Initialize exogenous state with crisis level
        if self.crisis_process is not None:
            crisis_level = self.crisis_process.current_state
            if state.exogenous_state is None:
                state.exogenous_state = np.array([0.0, float(crisis_level)])
            elif len(state.exogenous_state) < 2:
                state.exogenous_state = np.array([
                    state.exogenous_state[0] if len(state.exogenous_state) > 0 else 0.0,
                    float(crisis_level)
                ])
            else:
                state.exogenous_state[1] = float(crisis_level)
        
        return state


def create_enhanced_mdp(
    shelf_life: int = 5,
    num_suppliers: int = 2,
    mean_demand: float = 10.0,
    fast_lead_time: int = 1,
    slow_lead_time: int = 3,
    fast_cost: float = 2.0,
    slow_cost: float = 1.0,
    enable_crisis: bool = False,
    crisis_probability: float = 0.05,
    enable_rejection: bool = False,
    fast_rejection_prob: float = 0.0,
    slow_rejection_prob: float = 0.0,
    demand_type: str = "stationary"
) -> EnhancedPerishableInventoryMDP:
    """
    Factory function to create an enhanced MDP with crisis/rejection support.
    
    Args:
        shelf_life: Number of expiry buckets
        num_suppliers: 1 or 2 suppliers
        mean_demand: Mean demand rate
        fast_lead_time: Lead time for fast supplier
        slow_lead_time: Lead time for slow supplier
        fast_cost: Unit cost for fast supplier
        slow_cost: Unit cost for slow supplier
        enable_crisis: Whether to enable crisis state transitions
        crisis_probability: Probability of transitioning to crisis
        enable_rejection: Whether to enable supplier rejection
        fast_rejection_prob: Base rejection probability for fast supplier
        slow_rejection_prob: Base rejection probability for slow supplier
        demand_type: "stationary", "seasonal", "spiky", "composite"
    
    Returns:
        Configured EnhancedPerishableInventoryMDP
    """
    from .demand import create_demand_scenario, CompositeDemand
    from .crisis import create_crisis_process
    
    suppliers = [
        {
            "id": 0,
            "lead_time": fast_lead_time,
            "unit_cost": fast_cost,
            "fixed_cost": 0.0,
            "capacity": 100,
            "moq": 1,
            "rejection_prob": fast_rejection_prob
        },
        {
            "id": 1,
            "lead_time": slow_lead_time,
            "unit_cost": slow_cost,
            "fixed_cost": 0.0,
            "capacity": 100,
            "moq": 1,
            "rejection_prob": slow_rejection_prob
        }
    ]
    
    if num_suppliers == 1:
        suppliers = [suppliers[0]]
    
    # Create demand process
    if demand_type == "composite" and enable_crisis:
        demand_process = CompositeDemand(
            mean_demand,
            seasonality_amplitude=0.2,
            spike_prob=0.05,
            spike_multiplier=2.0
        )
    else:
        demand_process = create_demand_scenario(mean_demand, demand_type)
    
    cost_params = CostParameters.age_dependent_holding(
        shelf_life=shelf_life,
        base_holding=0.5,
        age_premium=0.5,
        shortage_cost=10.0,
        spoilage_cost=5.0
    )
    
    # Create crisis process if enabled
    crisis_process = None
    if enable_crisis:
        crisis_process = create_crisis_process(
            crisis_probability=crisis_probability,
            recovery_rate=0.2,
            severity="moderate"
        )
    
    return EnhancedPerishableInventoryMDP(
        shelf_life=shelf_life,
        suppliers=suppliers,
        demand_process=demand_process,
        cost_params=cost_params,
        crisis_process=crisis_process,
        enable_rejection=enable_rejection
    )

