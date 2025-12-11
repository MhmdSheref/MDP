"""
State Variables for the Perishable Inventory MDP

Implements the state representation X_t = (I_t, {P_t^(s)}_{s∈S}, B_t, z_t)
as defined in the mathematical formulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

from .exceptions import (
    StateValidationError, NegativeInventoryError, InvalidParameterError
)



@dataclass
class SupplierPipeline:
    """
    Pipeline for a single supplier tracking incoming orders.
    
    P_t^(s) = (P_t^(s,1), ..., P_t^(s,L_s))
    where P_t^(s,1) arrives at the start of period t.
    
    Attributes:
        supplier_id: Identifier for this supplier
        lead_time: Number of lead-time buckets L_s
        pipeline: Array of quantities in transit, indexed by arrival time
        scheduled: Non-decision based incoming supply (committed orders)
        unit_cost: Per-unit purchase cost v_s
        fixed_cost: Fixed ordering cost K_s
        capacity: Maximum order size U_s
        moq: Minimum order quantity M_s
    """
    supplier_id: int
    lead_time: int
    pipeline: np.ndarray = field(default_factory=lambda: np.array([]))
    scheduled: np.ndarray = field(default_factory=lambda: np.array([]))
    unit_cost: float = 1.0
    fixed_cost: float = 0.0
    capacity: float = float('inf')
    moq: int = 1
    
    def __post_init__(self):
        # Validation of parameters
        if self.lead_time <= 0:
            raise InvalidParameterError(f"Lead time must be positive, got {self.lead_time}")
        if self.unit_cost < 0:
            raise InvalidParameterError(f"Unit cost must be non-negative, got {self.unit_cost}")
        if self.fixed_cost < 0:
            raise InvalidParameterError(f"Fixed cost must be non-negative, got {self.fixed_cost}")
        if self.capacity <= 0:
            raise InvalidParameterError(f"Capacity must be positive, got {self.capacity}")
        if self.moq <= 0:
            raise InvalidParameterError(f"MOQ must be positive, got {self.moq}")
        
        # Initialize arrays if empty
        init_pipeline = len(self.pipeline) == 0
        init_scheduled = len(self.scheduled) == 0
        
        if init_pipeline:
            self.pipeline = np.zeros(self.lead_time, dtype=np.float64)
        if init_scheduled:
            self.scheduled = np.zeros(self.lead_time, dtype=np.float64)
        
        # Validate array lengths only if they were provided (not initialized)
        if not init_pipeline and len(self.pipeline) != self.lead_time:
            raise InvalidParameterError(
                f"Pipeline length {len(self.pipeline)} != lead_time {self.lead_time}"
            )
        if not init_scheduled and len(self.scheduled) != self.lead_time:
            raise InvalidParameterError(
                f"Scheduled length {len(self.scheduled)} != lead_time {self.lead_time}"
            )
        
        # Check for negative values
        if np.any(self.pipeline < 0):
            raise NegativeInventoryError("Pipeline contains negative values")
        if np.any(self.scheduled < 0):
            raise NegativeInventoryError("Scheduled contains negative values")
    
    def get_arriving(self) -> float:
        """Get quantity arriving this period (P_t^(s,1) + P̃_t^(s,1))"""
        return self.pipeline[0] + self.scheduled[0]
    
    def shift_and_add_order(self, order_qty: float) -> float:
        """
        Shift pipeline forward and add new order at the end.
        Returns the quantity that arrived (was at position 0).
        
        Pipeline evolution:
        P_{t+1}^(s,ℓ) = P_t^(s,ℓ+1) for ℓ < L_s
        P_{t+1}^(s,L_s) = a_t^(s)
        """
        if order_qty < 0:
            raise InvalidParameterError(f"Order quantity must be non-negative, got {order_qty}")
        
        arrived = self.pipeline[0]
        self.pipeline = np.roll(self.pipeline, -1)
        self.pipeline[-1] = order_qty
        return arrived
    
    def shift_scheduled(self) -> float:
        """
        Shift scheduled supply forward.
        
        P̃_{t+1}^(s,ℓ) = P̃_t^(s,ℓ+1)
        P̃_{t+1}^(s,L_s) = 0
        """
        arrived = self.scheduled[0]
        self.scheduled = np.roll(self.scheduled, -1)
        self.scheduled[-1] = 0
        return arrived
    
    def total_in_pipeline(self) -> float:
        """Total quantity in transit"""
        return np.sum(self.pipeline) + np.sum(self.scheduled)
    
    def copy(self) -> 'SupplierPipeline':
        """Create a deep copy of this pipeline"""
        return SupplierPipeline(
            supplier_id=self.supplier_id,
            lead_time=self.lead_time,
            pipeline=self.pipeline.copy(),
            scheduled=self.scheduled.copy(),
            unit_cost=self.unit_cost,
            fixed_cost=self.fixed_cost,
            capacity=self.capacity,
            moq=self.moq
        )


@dataclass
class InventoryState:
    """
    Complete state representation for the Perishable Inventory MDP.
    
    X_t = (I_t, {P_t^(s)}_{s∈S}, B_t, z_t)
    
    Attributes:
        inventory: On-hand inventory by expiry buckets I_t = (I_t^(1), ..., I_t^(N))
                   where I_t^(1) expires soonest, I_t^(N) is freshest
        shelf_life: Number of expiry buckets N
        pipelines: Dictionary of supplier pipelines {s: P_t^(s)}
        backorders: Unfulfilled demand B_t
        exogenous_state: External state z_t (seasonality, trends, etc.)
        time_step: Current time period t
    """
    shelf_life: int
    inventory: np.ndarray = field(default_factory=lambda: np.array([]))
    pipelines: Dict[int, SupplierPipeline] = field(default_factory=dict)
    backorders: float = 0.0
    exogenous_state: Optional[np.ndarray] = None
    time_step: int = 0
    
    def __post_init__(self):
        # Validate shelf life
        if self.shelf_life <= 0:
            raise InvalidParameterError(f"Shelf life must be positive, got {self.shelf_life}")
        
        # Validate backorders
        if self.backorders < 0:
            raise InvalidParameterError(f"Backorders cannot be negative, got {self.backorders}")
        
        # Initialize or validate inventory
        if len(self.inventory) == 0:
            self.inventory = np.zeros(self.shelf_life, dtype=np.float64)
        elif len(self.inventory) != self.shelf_life:
            raise InvalidParameterError(
                f"Inventory length {len(self.inventory)} != shelf_life {self.shelf_life}"
            )
        
        # Check for negative inventory
        if np.any(self.inventory < 0):
            neg_idx = np.where(self.inventory < 0)[0]
            raise NegativeInventoryError(
                bucket_idx=int(neg_idx[0]), 
                value=float(self.inventory[neg_idx[0]])
            )
    
    @property
    def total_inventory(self) -> float:
        """Total on-hand inventory across all expiry buckets"""
        return np.sum(self.inventory)
    
    @property
    def total_pipeline(self) -> float:
        """Total inventory in all supplier pipelines"""
        return sum(p.total_in_pipeline() for p in self.pipelines.values())
    
    @property
    def inventory_position(self) -> float:
        """
        Inventory position: on-hand + pipeline - backorders
        """
        return self.total_inventory + self.total_pipeline - self.backorders
    
    def survival_adjusted_inventory_position(self, survival_probs: np.ndarray) -> float:
        """
        Survival-adjusted inventory position IP_t^surv.
        
        Accounts for probability that inventory will be consumed before expiry.
        
        Args:
            survival_probs: Array ρ_n of survival probabilities for each bucket
        
        Returns:
            IP_t^surv = Σ_n ρ_n * I_t^(n)
        """
        if len(survival_probs) != self.shelf_life:
            raise ValueError("Survival probabilities must match shelf life")
        return np.dot(survival_probs, self.inventory)
    
    def get_arriving_inventory(self) -> float:
        """
        Get total inventory arriving this period from all pipelines.
        
        A_t = Σ_{s∈S} (P_t^(s,1) + P̃_t^(s,1))
        """
        return sum(p.get_arriving() for p in self.pipelines.values())
    
    def add_arrivals(self, arrivals: float) -> None:
        """
        Add arriving inventory to freshest bucket.
        
        I_t^(N) ← I_t^(N) + A_t
        """
        if arrivals < 0:
            raise InvalidParameterError(f"Arrivals must be non-negative, got {arrivals}")
        self.inventory[-1] += arrivals
    
    def serve_demand_fifo(self, demand: float) -> Tuple[float, float]:
        """
        Serve demand using FIFO (oldest first) policy.
        
        For n=1,...,N:
            take_n = min(I_t^(n), R)
            I_t^(n) ← I_t^(n) - take_n
            R ← R - take_n
        
        Args:
            demand: Total demand D_t to fulfill
        
        Returns:
            Tuple of (sales x_t, new_backorders B_t^new)
        """
        if demand < 0:
            raise InvalidParameterError(f"Demand must be non-negative, got {demand}")
        
        remaining = demand
        for n in range(self.shelf_life):
            take = min(self.inventory[n], remaining)
            self.inventory[n] -= take
            remaining -= take
            if remaining <= 0:
                break
        
        sales = demand - remaining
        new_backorders = max(remaining, 0)
        return sales, new_backorders
    
    def age_inventory(self) -> float:
        """
        Age inventory by one period and return spoiled quantity.
        
        Spoiled_t = I_t^(1)
        I_{t+1}^(n) = I_t^(n+1) for n=1,...,N-1
        I_{t+1}^(N) = 0
        
        Returns:
            Quantity of spoiled inventory
        """
        spoiled = self.inventory[0]
        self.inventory = np.roll(self.inventory, -1)
        self.inventory[-1] = 0
        return spoiled
    
    def get_aging_matrix(self) -> np.ndarray:
        """
        Get the aging shift matrix A_age for vectorized operations.
        
        A_age is an NxN matrix where:
        A_age[i,i+1] = 1 for i = 0,...,N-2
        All other entries are 0
        
        Then: I_{t+1}^aged = A_age @ I_t
        """
        N = self.shelf_life
        A_age = np.zeros((N, N), dtype=np.float64)
        for i in range(N - 1):
            A_age[i, i + 1] = 1
        return A_age
    
    def copy(self) -> 'InventoryState':
        """Create a deep copy of this state"""
        return InventoryState(
            shelf_life=self.shelf_life,
            inventory=self.inventory.copy(),
            pipelines={s: p.copy() for s, p in self.pipelines.items()},
            backorders=self.backorders,
            exogenous_state=self.exogenous_state.copy() if self.exogenous_state is not None else None,
            time_step=self.time_step
        )
    
    def to_tuple(self) -> tuple:
        """Convert state to hashable tuple for use as dictionary key"""
        inv_tuple = tuple(self.inventory.round(2))
        pipeline_tuples = tuple(
            (s, tuple(p.pipeline.round(2)), tuple(p.scheduled.round(2)))
            for s, p in sorted(self.pipelines.items())
        )
        exog = tuple(self.exogenous_state.round(2)) if self.exogenous_state is not None else ()
        return (inv_tuple, pipeline_tuples, round(self.backorders, 2), exog)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, InventoryState):
            return False
        return self.to_tuple() == other.to_tuple()


def create_state_from_config(
    shelf_life: int,
    suppliers: List[Dict],
    initial_inventory: Optional[np.ndarray] = None,
    initial_backorders: float = 0.0,
    initial_exogenous: Optional[np.ndarray] = None
) -> InventoryState:
    """
    Factory function to create an InventoryState from configuration.
    
    Args:
        shelf_life: Number of expiry buckets N
        suppliers: List of supplier configurations, each containing:
            - id: Supplier identifier
            - lead_time: Lead time L_s
            - unit_cost: Per-unit cost v_s
            - fixed_cost: Fixed ordering cost K_s (optional)
            - capacity: Maximum order U_s (optional)
            - moq: Minimum order quantity M_s (optional)
            - initial_pipeline: Initial pipeline quantities (optional)
        initial_inventory: Initial inventory by expiry bucket
        initial_backorders: Initial backorder level
        initial_exogenous: Initial exogenous state
    
    Returns:
        Configured InventoryState
    """
    state = InventoryState(
        shelf_life=shelf_life,
        inventory=initial_inventory if initial_inventory is not None else np.zeros(shelf_life),
        backorders=initial_backorders,
        exogenous_state=initial_exogenous
    )
    
    for supplier in suppliers:
        pipeline = SupplierPipeline(
            supplier_id=supplier['id'],
            lead_time=supplier['lead_time'],
            unit_cost=supplier.get('unit_cost', 1.0),
            fixed_cost=supplier.get('fixed_cost', 0.0),
            capacity=supplier.get('capacity', float('inf')),
            moq=supplier.get('moq', 1)
        )
        if 'initial_pipeline' in supplier:
            pipeline.pipeline = np.array(supplier['initial_pipeline'], dtype=np.float64)
        state.pipelines[supplier['id']] = pipeline
    
    return state


# =============================================================================
# MULTI-ITEM STATE EXTENSIONS
# =============================================================================


@dataclass
class SupplierConfig:
    """
    Extended supplier configuration with item coverage.
    
    Attributes:
        supplier_id: Unique supplier identifier
        lead_time: Number of periods for delivery
        unit_cost: Per-unit cost for orders
        fixed_cost: Fixed cost per order
        capacity: Maximum order quantity per period
        moq: Minimum order quantity
        items_supplied: Set of item IDs this supplier can provide
        rejection_prob: Base probability of order rejection
    """
    supplier_id: int
    lead_time: int
    unit_cost: float = 1.0
    fixed_cost: float = 0.0
    capacity: float = float('inf')
    moq: int = 1
    items_supplied: Optional[List[int]] = None  # None = all items
    rejection_prob: float = 0.0
    
    def __post_init__(self):
        if self.lead_time <= 0:
            raise InvalidParameterError(f"Lead time must be positive, got {self.lead_time}")
        if self.unit_cost < 0:
            raise InvalidParameterError(f"Unit cost must be non-negative, got {self.unit_cost}")
        if not (0 <= self.rejection_prob <= 1):
            raise InvalidParameterError(
                f"Rejection probability must be in [0, 1], got {self.rejection_prob}"
            )
    
    def supplies_item(self, item_id: int) -> bool:
        """Check if this supplier supplies the given item."""
        if self.items_supplied is None:
            return True  # Supplies all items
        return item_id in self.items_supplied
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility with existing API."""
        return {
            'id': self.supplier_id,
            'lead_time': self.lead_time,
            'unit_cost': self.unit_cost,
            'fixed_cost': self.fixed_cost,
            'capacity': self.capacity,
            'moq': self.moq,
            'items_supplied': self.items_supplied,
            'rejection_prob': self.rejection_prob
        }


@dataclass
class ItemConfig:
    """
    Configuration for a single item.
    
    Attributes:
        item_id: Unique item identifier
        shelf_life: Number of expiry buckets for this item
        name: Human-readable item name
        demand_multiplier: Multiplier for base demand (for item-specific demand)
    """
    item_id: int
    shelf_life: int
    name: str = ""
    demand_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.shelf_life <= 0:
            raise InvalidParameterError(f"Shelf life must be positive, got {self.shelf_life}")
        if self.demand_multiplier < 0:
            raise InvalidParameterError(
                f"Demand multiplier must be non-negative, got {self.demand_multiplier}"
            )


@dataclass
class MultiItemInventoryState:
    """
    State for multi-item perishable inventory.
    
    Extends the single-item InventoryState to support multiple products.
    
    Attributes:
        items: Dictionary of item configurations {item_id: ItemConfig}
        inventory: Dictionary of inventory by item {item_id: np.ndarray by expiry}
        pipelines: Dict of pipelines by (item_id, supplier_id)
        backorders: Dictionary of backorders by item {item_id: float}
        
        suppliers: Dictionary of supplier configurations {supplier_id: SupplierConfig}
        
        crisis_state: Current crisis level (0=normal, 1=elevated, 2=crisis)
        time_step: Current time period
        exogenous_state: External state array [time, crisis_level, ...]
    """
    items: Dict[int, ItemConfig] = field(default_factory=dict)
    inventory: Dict[int, np.ndarray] = field(default_factory=dict)
    pipelines: Dict[Tuple[int, int], SupplierPipeline] = field(default_factory=dict)
    backorders: Dict[int, float] = field(default_factory=dict)
    
    suppliers: Dict[int, SupplierConfig] = field(default_factory=dict)
    
    crisis_state: int = 0
    time_step: int = 0
    exogenous_state: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Initialize backorders for all items
        for item_id in self.items:
            if item_id not in self.backorders:
                self.backorders[item_id] = 0.0
            if item_id not in self.inventory:
                shelf_life = self.items[item_id].shelf_life
                self.inventory[item_id] = np.zeros(shelf_life, dtype=np.float64)
    
    @property
    def num_items(self) -> int:
        """Number of items in the system."""
        return len(self.items)
    
    @property
    def num_suppliers(self) -> int:
        """Number of suppliers in the system."""
        return len(self.suppliers)
    
    def get_total_inventory(self, item_id: int) -> float:
        """Get total on-hand inventory for an item."""
        if item_id not in self.inventory:
            return 0.0
        return np.sum(self.inventory[item_id])
    
    def get_total_pipeline(self, item_id: int) -> float:
        """Get total inventory in transit for an item."""
        total = 0.0
        for (iid, sid), pipeline in self.pipelines.items():
            if iid == item_id:
                total += pipeline.total_in_pipeline()
        return total
    
    def get_inventory_position(self, item_id: int) -> float:
        """Get inventory position for an item."""
        return (
            self.get_total_inventory(item_id) +
            self.get_total_pipeline(item_id) -
            self.backorders.get(item_id, 0.0)
        )
    
    def get_suppliers_for_item(self, item_id: int) -> List[int]:
        """Get list of supplier IDs that supply the given item."""
        return [
            sid for sid, config in self.suppliers.items()
            if config.supplies_item(item_id)
        ]
    
    def add_arrivals(self, item_id: int, arrivals: float) -> None:
        """Add arriving inventory to the freshest bucket for an item."""
        if arrivals < 0:
            raise InvalidParameterError(f"Arrivals must be non-negative, got {arrivals}")
        if item_id in self.inventory:
            self.inventory[item_id][-1] += arrivals
    
    def serve_demand_fifo(self, item_id: int, demand: float) -> Tuple[float, float]:
        """
        Serve demand for an item using FIFO policy.
        
        Returns:
            Tuple of (sales, new_backorders)
        """
        if demand < 0:
            raise InvalidParameterError(f"Demand must be non-negative, got {demand}")
        
        if item_id not in self.inventory:
            return 0.0, demand
        
        inv = self.inventory[item_id]
        remaining = demand
        
        for n in range(len(inv)):
            take = min(inv[n], remaining)
            inv[n] -= take
            remaining -= take
            if remaining <= 0:
                break
        
        sales = demand - remaining
        return sales, max(remaining, 0)
    
    def age_inventory(self, item_id: int) -> float:
        """Age inventory for an item and return spoiled quantity."""
        if item_id not in self.inventory:
            return 0.0
        
        inv = self.inventory[item_id]
        spoiled = inv[0]
        self.inventory[item_id] = np.roll(inv, -1)
        self.inventory[item_id][-1] = 0
        return spoiled
    
    def copy(self) -> 'MultiItemInventoryState':
        """Create a deep copy of this state."""
        return MultiItemInventoryState(
            items={k: ItemConfig(
                item_id=v.item_id,
                shelf_life=v.shelf_life,
                name=v.name,
                demand_multiplier=v.demand_multiplier
            ) for k, v in self.items.items()},
            inventory={k: v.copy() for k, v in self.inventory.items()},
            pipelines={k: v.copy() for k, v in self.pipelines.items()},
            backorders=self.backorders.copy(),
            suppliers={k: SupplierConfig(
                supplier_id=v.supplier_id,
                lead_time=v.lead_time,
                unit_cost=v.unit_cost,
                fixed_cost=v.fixed_cost,
                capacity=v.capacity,
                moq=v.moq,
                items_supplied=v.items_supplied.copy() if v.items_supplied else None,
                rejection_prob=v.rejection_prob
            ) for k, v in self.suppliers.items()},
            crisis_state=self.crisis_state,
            time_step=self.time_step,
            exogenous_state=self.exogenous_state.copy() if self.exogenous_state is not None else None
        )


def create_multi_item_state(
    items: List[Dict],
    suppliers: List[Dict],
    initial_inventory: Optional[Dict[int, np.ndarray]] = None,
    initial_crisis: int = 0
) -> MultiItemInventoryState:
    """
    Factory function to create a multi-item state.
    
    Args:
        items: List of item configurations with keys:
            - id: Item identifier
            - shelf_life: Number of expiry buckets
            - name: Optional item name
            - demand_multiplier: Optional demand multiplier
        suppliers: List of supplier configurations with keys:
            - id: Supplier identifier
            - lead_time: Delivery lead time
            - unit_cost: Per-unit cost
            - items_supplied: Optional list of item IDs (None = all)
            - rejection_prob: Base rejection probability
        initial_inventory: Optional {item_id: inventory_array}
        initial_crisis: Initial crisis level (0, 1, or 2)
    
    Returns:
        Configured MultiItemInventoryState
    """
    state = MultiItemInventoryState(crisis_state=initial_crisis)
    
    # Add items
    for item in items:
        item_id = item['id']
        state.items[item_id] = ItemConfig(
            item_id=item_id,
            shelf_life=item['shelf_life'],
            name=item.get('name', f'Item_{item_id}'),
            demand_multiplier=item.get('demand_multiplier', 1.0)
        )
        
        # Initialize inventory
        if initial_inventory and item_id in initial_inventory:
            state.inventory[item_id] = np.array(initial_inventory[item_id], dtype=np.float64)
        else:
            state.inventory[item_id] = np.zeros(item['shelf_life'], dtype=np.float64)
        
        state.backorders[item_id] = 0.0
    
    # Add suppliers
    for supplier in suppliers:
        supplier_id = supplier['id']
        state.suppliers[supplier_id] = SupplierConfig(
            supplier_id=supplier_id,
            lead_time=supplier['lead_time'],
            unit_cost=supplier.get('unit_cost', 1.0),
            fixed_cost=supplier.get('fixed_cost', 0.0),
            capacity=supplier.get('capacity', float('inf')),
            moq=supplier.get('moq', 1),
            items_supplied=supplier.get('items_supplied', None),
            rejection_prob=supplier.get('rejection_prob', 0.0)
        )
        
        # Create pipelines for each item-supplier pair
        for item_id in state.items:
            if state.suppliers[supplier_id].supplies_item(item_id):
                pipeline = SupplierPipeline(
                    supplier_id=supplier_id,
                    lead_time=supplier['lead_time'],
                    unit_cost=supplier.get('unit_cost', 1.0),
                    fixed_cost=supplier.get('fixed_cost', 0.0),
                    capacity=supplier.get('capacity', float('inf')),
                    moq=supplier.get('moq', 1)
                )
                state.pipelines[(item_id, supplier_id)] = pipeline
    
    # Initialize exogenous state
    state.exogenous_state = np.array([0.0, float(initial_crisis)])
    
    return state
