from .env import PerishableInventoryMDP
from .core.state import InventoryState, SupplierPipeline
from .core.demand import DemandProcess, PoissonDemand, NegativeBinomialDemand
from .core.costs import CostParameters
from .interfaces import InventoryEnvironment
from .simulation import run_episode

# Gymnasium wrapper (optional - only if gymnasium is installed)
try:
    from .gym_env import (
        PerishableInventoryGymEnv,
        PerishableInventoryGymEnvContinuous,
        make_inventory_env
    )
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

__all__ = [
    "PerishableInventoryMDP",
    "InventoryState",
    "SupplierPipeline",
    "DemandProcess",
    "PoissonDemand",
    "NegativeBinomialDemand",
    "CostParameters",
    "InventoryEnvironment",
    "run_episode",
]

# Add Gym exports if available
if _GYM_AVAILABLE:
    __all__.extend([
        "PerishableInventoryGymEnv",
        "PerishableInventoryGymEnvContinuous",
        "make_inventory_env",
    ])
