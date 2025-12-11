"""
Supplier Contracts for the Perishable Inventory MDP

Models long-term contracts with suppliers that provide:
- Discounted unit prices
- Guaranteed lead times
- Minimum order commitments with penalties for under-ordering
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

from .exceptions import InvalidParameterError


class ContractStatus(IntEnum):
    """Status of a supplier contract."""
    PENDING = 0    # Offered but not yet signed
    ACTIVE = 1     # Currently active
    EXPIRED = 2    # Contract duration completed
    BROKEN = 3     # Terminated early with penalty


@dataclass
class SupplierContract:
    """
    Long-term contract with a supplier.
    
    Provides price discounts in exchange for committed order quantities.
    
    Attributes:
        contract_id: Unique identifier for this contract
        supplier_id: ID of the supplier this contract is with
        item_id: ID of the item covered (optional, None = all items)
        
        committed_quantity: Minimum order quantity per period
        discount_rate: Discount off regular unit price (0.15 = 15% off)
        lead_time_reduction: Periods reduced from normal lead time
        
        penalty_rate: Penalty per unit for ordering less than committed
        breaking_penalty: One-time penalty for terminating early
        
        duration: Total contract duration in periods
        periods_remaining: Periods left until expiration
        status: Current contract status
        
        total_ordered: Running total of units ordered under this contract
        total_penalty_incurred: Running total of penalties paid
    """
    contract_id: int
    supplier_id: int
    item_id: Optional[int] = None
    
    # Contract terms
    committed_quantity: float = 10.0
    discount_rate: float = 0.10
    lead_time_reduction: int = 0
    
    # Penalties
    penalty_rate: float = 0.5  # Per unit under-ordering penalty
    breaking_penalty: float = 100.0  # Early termination penalty
    
    # Duration
    duration: int = 20
    periods_remaining: int = -1  # -1 means use duration
    status: ContractStatus = ContractStatus.PENDING
    
    # Tracking
    total_ordered: float = 0.0
    total_penalty_incurred: float = 0.0
    
    def __post_init__(self):
        # Set periods_remaining to duration if not specified
        if self.periods_remaining < 0:
            self.periods_remaining = self.duration
        
        # Validation
        if self.committed_quantity < 0:
            raise InvalidParameterError(
                f"Committed quantity must be non-negative, got {self.committed_quantity}"
            )
        if not (0 <= self.discount_rate <= 1):
            raise InvalidParameterError(
                f"Discount rate must be in [0, 1], got {self.discount_rate}"
            )
        if self.lead_time_reduction < 0:
            raise InvalidParameterError(
                f"Lead time reduction must be non-negative, got {self.lead_time_reduction}"
            )
        if self.penalty_rate < 0:
            raise InvalidParameterError(
                f"Penalty rate must be non-negative, got {self.penalty_rate}"
            )
        if self.breaking_penalty < 0:
            raise InvalidParameterError(
                f"Breaking penalty must be non-negative, got {self.breaking_penalty}"
            )
        if self.duration <= 0:
            raise InvalidParameterError(
                f"Duration must be positive, got {self.duration}"
            )
    
    def sign(self) -> None:
        """Activate the contract."""
        if self.status != ContractStatus.PENDING:
            raise InvalidParameterError(
                f"Can only sign pending contracts, status is {self.status.name}"
            )
        self.status = ContractStatus.ACTIVE
    
    def is_active(self) -> bool:
        """Check if contract is currently active."""
        return self.status == ContractStatus.ACTIVE
    
    def get_discounted_price(self, base_price: float) -> float:
        """
        Calculate discounted unit price.
        
        Args:
            base_price: Original unit price from supplier
        
        Returns:
            Discounted price under this contract
        """
        return base_price * (1 - self.discount_rate)
    
    def get_effective_lead_time(self, base_lead_time: int) -> int:
        """
        Calculate effective lead time under contract.
        
        Args:
            base_lead_time: Supplier's normal lead time
        
        Returns:
            Reduced lead time (minimum 1)
        """
        return max(1, base_lead_time - self.lead_time_reduction)
    
    def calculate_period_penalty(self, order_quantity: float) -> float:
        """
        Calculate penalty for under-ordering in a period.
        
        Args:
            order_quantity: Actual order quantity this period
        
        Returns:
            Penalty amount (0 if order meets commitment)
        """
        if not self.is_active():
            return 0.0
        
        shortfall = max(0, self.committed_quantity - order_quantity)
        return shortfall * self.penalty_rate
    
    def process_period(self, order_quantity: float) -> Tuple[float, float]:
        """
        Process one period of the contract.
        
        Args:
            order_quantity: Quantity ordered from this supplier this period
        
        Returns:
            Tuple of (discount_savings, penalty_incurred)
        """
        if not self.is_active():
            return 0.0, 0.0
        
        # Track order
        self.total_ordered += order_quantity
        
        # Calculate penalty
        penalty = self.calculate_period_penalty(order_quantity)
        self.total_penalty_incurred += penalty
        
        # Decrement remaining periods
        self.periods_remaining -= 1
        
        # Check for expiration
        if self.periods_remaining <= 0:
            self.status = ContractStatus.EXPIRED
        
        return 0.0, penalty  # Discount applied at order time
    
    def break_contract(self) -> float:
        """
        Terminate the contract early.
        
        Returns:
            Breaking penalty amount
        """
        if not self.is_active():
            return 0.0
        
        self.status = ContractStatus.BROKEN
        self.total_penalty_incurred += self.breaking_penalty
        return self.breaking_penalty
    
    def copy(self) -> 'SupplierContract':
        """Create a copy of this contract."""
        return SupplierContract(
            contract_id=self.contract_id,
            supplier_id=self.supplier_id,
            item_id=self.item_id,
            committed_quantity=self.committed_quantity,
            discount_rate=self.discount_rate,
            lead_time_reduction=self.lead_time_reduction,
            penalty_rate=self.penalty_rate,
            breaking_penalty=self.breaking_penalty,
            duration=self.duration,
            periods_remaining=self.periods_remaining,
            status=self.status,
            total_ordered=self.total_ordered,
            total_penalty_incurred=self.total_penalty_incurred
        )


@dataclass
class ContractManager:
    """
    Manages all contracts for the inventory system.
    
    Attributes:
        contracts: Dictionary of active contracts by contract_id
        available_contracts: Contracts available for signing
        next_contract_id: Counter for contract IDs
    """
    contracts: Dict[int, SupplierContract] = field(default_factory=dict)
    available_contracts: List[SupplierContract] = field(default_factory=list)
    next_contract_id: int = 0
    
    def add_available_contract(self, contract: SupplierContract) -> None:
        """Add a contract to the available pool."""
        self.available_contracts.append(contract)
    
    def sign_contract(self, contract_id: int) -> Optional[SupplierContract]:
        """
        Sign an available contract.
        
        Args:
            contract_id: ID of the contract to sign
        
        Returns:
            The signed contract, or None if not found
        """
        for i, contract in enumerate(self.available_contracts):
            if contract.contract_id == contract_id:
                contract.sign()
                self.contracts[contract_id] = contract
                self.available_contracts.pop(i)
                return contract
        return None
    
    def get_active_contracts(self, supplier_id: Optional[int] = None) -> List[SupplierContract]:
        """
        Get all active contracts, optionally filtered by supplier.
        
        Args:
            supplier_id: If provided, filter to this supplier only
        
        Returns:
            List of active contracts
        """
        active = [c for c in self.contracts.values() if c.is_active()]
        if supplier_id is not None:
            active = [c for c in active if c.supplier_id == supplier_id]
        return active
    
    def get_contract_for_order(
        self,
        supplier_id: int,
        item_id: Optional[int] = None
    ) -> Optional[SupplierContract]:
        """
        Get the applicable contract for an order.
        
        Args:
            supplier_id: Supplier being ordered from
            item_id: Item being ordered (optional)
        
        Returns:
            The applicable contract, or None if no contract
        """
        for contract in self.contracts.values():
            if not contract.is_active():
                continue
            if contract.supplier_id != supplier_id:
                continue
            # Check item match (None means all items)
            if contract.item_id is None or contract.item_id == item_id:
                return contract
        return None
    
    def process_period_all(
        self,
        orders: Dict[int, float]
    ) -> Tuple[float, float]:
        """
        Process all active contracts for one period.
        
        Args:
            orders: Dictionary of {supplier_id: order_quantity}
        
        Returns:
            Tuple of (total_discounts, total_penalties)
        """
        total_discounts = 0.0
        total_penalties = 0.0
        
        for contract in list(self.contracts.values()):
            if not contract.is_active():
                continue
            
            order_qty = orders.get(contract.supplier_id, 0.0)
            discounts, penalties = contract.process_period(order_qty)
            total_discounts += discounts
            total_penalties += penalties
        
        return total_discounts, total_penalties
    
    def generate_available_contracts(
        self,
        suppliers: List[Dict],
        num_contracts: int = 2,
        seed: Optional[int] = None
    ) -> None:
        """
        Generate random available contracts.
        
        Args:
            suppliers: List of supplier configurations
            num_contracts: Number of contracts to generate
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.available_contracts = []
        
        for i in range(num_contracts):
            supplier = np.random.choice(suppliers)
            supplier_id = supplier.get('id', i)
            
            contract = SupplierContract(
                contract_id=self.next_contract_id,
                supplier_id=supplier_id,
                committed_quantity=np.random.uniform(5, 20),
                discount_rate=np.random.uniform(0.05, 0.25),
                lead_time_reduction=np.random.randint(0, 2),
                penalty_rate=np.random.uniform(0.3, 0.8),
                breaking_penalty=np.random.uniform(50, 200),
                duration=np.random.randint(10, 30),
                status=ContractStatus.PENDING
            )
            
            self.available_contracts.append(contract)
            self.next_contract_id += 1
    
    def copy(self) -> 'ContractManager':
        """Create a copy of the contract manager."""
        return ContractManager(
            contracts={k: v.copy() for k, v in self.contracts.items()},
            available_contracts=[c.copy() for c in self.available_contracts],
            next_contract_id=self.next_contract_id
        )


def calculate_contract_costs(
    contracts: Dict[int, SupplierContract],
    orders: Dict[int, float],
    base_prices: Dict[int, float]
) -> Dict[str, float]:
    """
    Calculate contract-related costs and savings.
    
    Args:
        contracts: Active contracts by ID
        orders: Orders by supplier_id
        base_prices: Base unit prices by supplier_id
    
    Returns:
        Dictionary with:
        - 'discount_savings': Total discount savings
        - 'under_order_penalty': Total penalty for under-ordering
        - 'net_contract_benefit': Net benefit (savings - penalties)
    """
    discount_savings = 0.0
    under_order_penalty = 0.0
    
    for contract in contracts.values():
        if not contract.is_active():
            continue
        
        supplier_id = contract.supplier_id
        order_qty = orders.get(supplier_id, 0.0)
        base_price = base_prices.get(supplier_id, 1.0)
        
        # Calculate discount savings
        discount_per_unit = base_price * contract.discount_rate
        discount_savings += discount_per_unit * order_qty
        
        # Calculate penalty
        penalty = contract.calculate_period_penalty(order_qty)
        under_order_penalty += penalty
    
    return {
        'discount_savings': discount_savings,
        'under_order_penalty': under_order_penalty,
        'net_contract_benefit': discount_savings - under_order_penalty
    }
