"""
Tests for Multi-Item State Extensions.

Tests SupplierConfig, ItemConfig, MultiItemInventoryState, and factory functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.state import (
    SupplierConfig, ItemConfig, MultiItemInventoryState,
    create_multi_item_state
)
from perishable_inventory_mdp.exceptions import InvalidParameterError


class TestSupplierConfig:
    """Tests for SupplierConfig dataclass"""
    
    def test_basic_construction(self):
        """Test basic construction"""
        config = SupplierConfig(
            supplier_id=0,
            lead_time=3,
            unit_cost=2.0
        )
        
        assert config.supplier_id == 0
        assert config.lead_time == 3
        assert config.unit_cost == 2.0
        assert config.items_supplied is None  # All items
        assert config.rejection_prob == 0.0
    
    def test_supplies_item_all(self):
        """Test supplies_item when items_supplied is None (all items)"""
        config = SupplierConfig(supplier_id=0, lead_time=2)
        
        assert config.supplies_item(0)
        assert config.supplies_item(1)
        assert config.supplies_item(999)
    
    def test_supplies_item_limited(self):
        """Test supplies_item when items_supplied is specified"""
        config = SupplierConfig(
            supplier_id=0,
            lead_time=2,
            items_supplied=[0, 2, 5]
        )
        
        assert config.supplies_item(0)
        assert config.supplies_item(2)
        assert config.supplies_item(5)
        assert not config.supplies_item(1)
        assert not config.supplies_item(3)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = SupplierConfig(
            supplier_id=1,
            lead_time=3,
            unit_cost=1.5
        )
        
        d = config.to_dict()
        
        assert d['id'] == 1
        assert d['lead_time'] == 3
        assert d['unit_cost'] == 1.5
    
    def test_invalid_lead_time(self):
        """Test validation of lead time"""
        with pytest.raises(InvalidParameterError):
            SupplierConfig(supplier_id=0, lead_time=0)
    
    def test_invalid_rejection_prob(self):
        """Test validation of rejection probability"""
        with pytest.raises(InvalidParameterError):
            SupplierConfig(supplier_id=0, lead_time=1, rejection_prob=1.5)


class TestItemConfig:
    """Tests for ItemConfig dataclass"""
    
    def test_basic_construction(self):
        """Test basic construction"""
        config = ItemConfig(item_id=0, shelf_life=5, name="Test Item")
        
        assert config.item_id == 0
        assert config.shelf_life == 5
        assert config.name == "Test Item"
        assert config.demand_multiplier == 1.0
    
    def test_invalid_shelf_life(self):
        """Test validation of shelf life"""
        with pytest.raises(InvalidParameterError):
            ItemConfig(item_id=0, shelf_life=0)
    
    def test_invalid_demand_multiplier(self):
        """Test validation of demand multiplier"""
        with pytest.raises(InvalidParameterError):
            ItemConfig(item_id=0, shelf_life=5, demand_multiplier=-0.5)


class TestMultiItemInventoryState:
    """Tests for MultiItemInventoryState"""
    
    @pytest.fixture
    def simple_state(self):
        """Create a simple 2-item, 2-supplier state"""
        return create_multi_item_state(
            items=[
                {'id': 0, 'shelf_life': 4},
                {'id': 1, 'shelf_life': 3}
            ],
            suppliers=[
                {'id': 0, 'lead_time': 1, 'unit_cost': 2.0},
                {'id': 1, 'lead_time': 3, 'unit_cost': 1.0}
            ]
        )
    
    def test_basic_properties(self, simple_state):
        """Test basic properties"""
        assert simple_state.num_items == 2
        assert simple_state.num_suppliers == 2
        assert simple_state.crisis_state == 0
    
    def test_inventory_initialization(self, simple_state):
        """Test inventory is initialized correctly"""
        assert len(simple_state.inventory[0]) == 4  # shelf_life 4
        assert len(simple_state.inventory[1]) == 3  # shelf_life 3
        assert np.sum(simple_state.inventory[0]) == 0
    
    def test_pipelines_created(self, simple_state):
        """Test pipelines are created for each item-supplier pair"""
        # Both suppliers supply all items
        assert (0, 0) in simple_state.pipelines
        assert (0, 1) in simple_state.pipelines
        assert (1, 0) in simple_state.pipelines
        assert (1, 1) in simple_state.pipelines
    
    def test_get_total_inventory(self, simple_state):
        """Test getting total inventory"""
        simple_state.inventory[0] = np.array([5.0, 10.0, 15.0, 20.0])
        
        assert simple_state.get_total_inventory(0) == 50.0
        assert simple_state.get_total_inventory(1) == 0.0
    
    def test_get_suppliers_for_item(self):
        """Test getting suppliers for item with limited coverage"""
        state = create_multi_item_state(
            items=[
                {'id': 0, 'shelf_life': 4},
                {'id': 1, 'shelf_life': 3}
            ],
            suppliers=[
                {'id': 0, 'lead_time': 1, 'items_supplied': [0]},  # Only item 0
                {'id': 1, 'lead_time': 3, 'items_supplied': [1]}   # Only item 1
            ]
        )
        
        suppliers_0 = state.get_suppliers_for_item(0)
        suppliers_1 = state.get_suppliers_for_item(1)
        
        assert suppliers_0 == [0]
        assert suppliers_1 == [1]
    
    def test_add_arrivals(self, simple_state):
        """Test adding arrivals to freshest bucket"""
        simple_state.add_arrivals(0, 10.0)
        
        assert simple_state.inventory[0][-1] == 10.0  # Freshest bucket
    
    def test_serve_demand_fifo(self, simple_state):
        """Test serving demand with FIFO"""
        simple_state.inventory[0] = np.array([5.0, 10.0, 15.0, 20.0])
        
        sales, backorders = simple_state.serve_demand_fifo(0, 12.0)
        
        assert sales == 12.0
        assert backorders == 0.0
        # Should consume 5 from bucket 0, 7 from bucket 1
        assert_array_almost_equal(
            simple_state.inventory[0],
            [0.0, 3.0, 15.0, 20.0]
        )
    
    def test_serve_demand_creates_backorders(self, simple_state):
        """Test backorders created when demand exceeds inventory"""
        simple_state.inventory[0] = np.array([5.0, 5.0, 5.0, 5.0])
        
        sales, backorders = simple_state.serve_demand_fifo(0, 30.0)
        
        assert sales == 20.0
        assert backorders == 10.0
    
    def test_age_inventory(self, simple_state):
        """Test aging inventory and spoilage"""
        simple_state.inventory[0] = np.array([8.0, 10.0, 15.0, 20.0])
        
        spoiled = simple_state.age_inventory(0)
        
        assert spoiled == 8.0
        assert_array_almost_equal(
            simple_state.inventory[0],
            [10.0, 15.0, 20.0, 0.0]
        )
    
    def test_copy(self, simple_state):
        """Test copying state"""
        simple_state.inventory[0] = np.array([1.0, 2.0, 3.0, 4.0])
        simple_state.crisis_state = 2
        
        copied = simple_state.copy()
        
        assert copied is not simple_state
        assert np.array_equal(copied.inventory[0], simple_state.inventory[0])
        assert copied.crisis_state == 2
        
        # Modifying copy doesn't affect original
        copied.inventory[0][0] = 99.0
        assert simple_state.inventory[0][0] == 1.0


class TestCreateMultiItemState:
    """Tests for create_multi_item_state factory"""
    
    def test_basic_creation(self):
        """Test basic state creation"""
        state = create_multi_item_state(
            items=[
                {'id': 0, 'shelf_life': 5, 'name': 'Item A'},
                {'id': 1, 'shelf_life': 3, 'name': 'Item B'}
            ],
            suppliers=[
                {'id': 0, 'lead_time': 2}
            ]
        )
        
        assert state.num_items == 2
        assert state.items[0].name == 'Item A'
        assert state.items[1].shelf_life == 3
    
    def test_with_initial_inventory(self):
        """Test creation with initial inventory"""
        state = create_multi_item_state(
            items=[
                {'id': 0, 'shelf_life': 3}
            ],
            suppliers=[
                {'id': 0, 'lead_time': 1}
            ],
            initial_inventory={
                0: np.array([5.0, 10.0, 15.0])
            }
        )
        
        assert state.get_total_inventory(0) == 30.0
    
    def test_with_initial_crisis(self):
        """Test creation with initial crisis state"""
        state = create_multi_item_state(
            items=[{'id': 0, 'shelf_life': 3}],
            suppliers=[{'id': 0, 'lead_time': 1}],
            initial_crisis=2
        )
        
        assert state.crisis_state == 2
        assert state.exogenous_state[1] == 2.0
    
    def test_supplier_item_coverage(self):
        """Test limited supplier-item coverage"""
        state = create_multi_item_state(
            items=[
                {'id': 0, 'shelf_life': 4},
                {'id': 1, 'shelf_life': 3},
                {'id': 2, 'shelf_life': 5}
            ],
            suppliers=[
                {'id': 0, 'lead_time': 1, 'items_supplied': [0, 1]},
                {'id': 1, 'lead_time': 2, 'items_supplied': [1, 2]}
            ]
        )
        
        # Check pipelines were created correctly
        assert (0, 0) in state.pipelines  # Item 0 from Supplier 0
        assert (1, 0) in state.pipelines  # Item 1 from Supplier 0
        assert (0, 1) not in state.pipelines  # Item 0 NOT from Supplier 1
        assert (1, 1) in state.pipelines  # Item 1 from Supplier 1
        assert (2, 1) in state.pipelines  # Item 2 from Supplier 1
        assert (2, 0) not in state.pipelines  # Item 2 NOT from Supplier 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
