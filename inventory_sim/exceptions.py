"""
Custom exceptions for the Perishable Inventory MDP.

Provides descriptive error types for better error handling and debugging.
"""

from typing import Any, Dict, Optional


class PerishableInventoryError(Exception):
    """Base exception for all perishable inventory MDP errors."""
    pass


class StateValidationError(PerishableInventoryError):
    """Raised when state validation fails."""
    
    def __init__(self, message: str, state: Optional[Any] = None):
        super().__init__(message)
        self.state = state


class ActionValidationError(PerishableInventoryError):
    """Raised when action validation fails."""
    
    def __init__(self, message: str, action: Optional[Dict[int, float]] = None, state: Optional[Any] = None):
        super().__init__(message)
        self.action = action
        self.state = state


class InvalidParameterError(PerishableInventoryError):
    """Raised when invalid parameters are provided."""
    pass


class SupplierNotFoundError(PerishableInventoryError):
    """Raised when a supplier ID is not found in the state."""
    
    def __init__(self, supplier_id: int, available_suppliers: Optional[list] = None):
        msg = f"Supplier {supplier_id} not found"
        if available_suppliers:
            msg += f". Available suppliers: {available_suppliers}"
        super().__init__(msg)
        self.supplier_id = supplier_id
        self.available_suppliers = available_suppliers


class CapacityViolationError(ActionValidationError):
    """Raised when an order exceeds supplier capacity."""
    
    def __init__(self, supplier_id: int, order_qty: float, capacity: float):
        msg = f"Order quantity {order_qty} for supplier {supplier_id} exceeds capacity {capacity}"
        super().__init__(msg)
        self.supplier_id = supplier_id
        self.order_qty = order_qty
        self.capacity = capacity


class MOQViolationError(ActionValidationError):
    """Raised when an order violates minimum order quantity constraints."""
    
    def __init__(self, supplier_id: int, order_qty: float, moq: float):
        msg = f"Order quantity {order_qty} for supplier {supplier_id} must be 0 or a multiple of MOQ {moq}"
        super().__init__(msg)
        self.supplier_id = supplier_id
        self.order_qty = order_qty
        self.moq = moq


class NegativeInventoryError(StateValidationError):
    """Raised when inventory becomes negative (should never happen)."""
    
    def __init__(self, bucket_idx: Optional[int] = None, value: Optional[float] = None):
        msg = "Negative inventory detected"
        if bucket_idx is not None and value is not None:
            msg += f" in bucket {bucket_idx}: {value}"
        super().__init__(msg)
        self.bucket_idx = bucket_idx
        self.value = value


class InvalidDemandError(PerishableInventoryError):
    """Raised when demand parameters are invalid."""
    pass


class SolverConvergenceError(PerishableInventoryError):
    """Raised when a solver fails to converge."""
    
    def __init__(self, message: str, iterations: Optional[int] = None, tolerance: Optional[float] = None):
        super().__init__(message)
        self.iterations = iterations
        self.tolerance = tolerance
