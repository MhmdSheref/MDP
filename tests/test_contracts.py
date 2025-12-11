"""
Tests for the Contracts module.

Tests SupplierContract, ContractManager, and contract cost calculations.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.contracts import (
    SupplierContract, ContractManager, ContractStatus,
    calculate_contract_costs
)
from perishable_inventory_mdp.exceptions import InvalidParameterError


class TestContractStatus:
    """Tests for ContractStatus enum"""
    
    def test_enum_values(self):
        """Test enum values"""
        assert ContractStatus.PENDING == 0
        assert ContractStatus.ACTIVE == 1
        assert ContractStatus.EXPIRED == 2
        assert ContractStatus.BROKEN == 3


class TestSupplierContract:
    """Tests for SupplierContract dataclass"""
    
    def test_basic_construction(self):
        """Test basic construction"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=15.0,
            discount_rate=0.10,
            duration=20
        )
        
        assert contract.contract_id == 1
        assert contract.supplier_id == 0
        assert contract.committed_quantity == 15.0
        assert contract.discount_rate == 0.10
        assert contract.periods_remaining == 20
        assert contract.status == ContractStatus.PENDING
    
    def test_sign_contract(self):
        """Test signing a contract"""
        contract = SupplierContract(contract_id=1, supplier_id=0)
        
        assert contract.status == ContractStatus.PENDING
        contract.sign()
        assert contract.status == ContractStatus.ACTIVE
    
    def test_sign_already_active(self):
        """Test signing already active contract raises error"""
        contract = SupplierContract(contract_id=1, supplier_id=0)
        contract.sign()
        
        with pytest.raises(InvalidParameterError):
            contract.sign()
    
    def test_is_active(self):
        """Test is_active check"""
        contract = SupplierContract(contract_id=1, supplier_id=0)
        
        assert not contract.is_active()
        contract.sign()
        assert contract.is_active()
    
    def test_get_discounted_price(self):
        """Test discounted price calculation"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            discount_rate=0.15
        )
        
        base_price = 10.0
        discounted = contract.get_discounted_price(base_price)
        
        assert_almost_equal(discounted, 8.5)  # 10 * (1 - 0.15)
    
    def test_get_effective_lead_time(self):
        """Test effective lead time calculation"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            lead_time_reduction=2
        )
        
        assert contract.get_effective_lead_time(5) == 3
        assert contract.get_effective_lead_time(2) == 1  # Minimum 1
        assert contract.get_effective_lead_time(1) == 1  # Minimum 1
    
    def test_calculate_period_penalty_meets_commitment(self):
        """Test no penalty when meeting commitment"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=10.0,
            penalty_rate=0.5
        )
        contract.sign()
        
        # Order exactly commitment
        penalty = contract.calculate_period_penalty(10.0)
        assert penalty == 0.0
        
        # Order more than commitment
        penalty = contract.calculate_period_penalty(15.0)
        assert penalty == 0.0
    
    def test_calculate_period_penalty_under_orders(self):
        """Test penalty for under-ordering"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=10.0,
            penalty_rate=0.5
        )
        contract.sign()
        
        # Order less than commitment
        penalty = contract.calculate_period_penalty(7.0)
        expected = 3.0 * 0.5  # shortfall * rate
        assert_almost_equal(penalty, expected)
    
    def test_calculate_penalty_inactive_contract(self):
        """Test no penalty for inactive contract"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=10.0,
            penalty_rate=0.5
        )
        # Not signed
        
        penalty = contract.calculate_period_penalty(0.0)
        assert penalty == 0.0
    
    def test_process_period(self):
        """Test processing a period"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=10.0,
            penalty_rate=0.5,
            duration=3
        )
        contract.sign()
        
        # Process period with under-ordering
        _, penalty = contract.process_period(8.0)
        
        assert_almost_equal(penalty, 1.0)  # (10-8) * 0.5
        assert contract.total_ordered == 8.0
        assert contract.periods_remaining == 2
    
    def test_process_period_expiration(self):
        """Test contract expires after duration"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            duration=2
        )
        contract.sign()
        
        contract.process_period(10.0)
        assert contract.is_active()
        assert contract.periods_remaining == 1
        
        contract.process_period(10.0)
        assert not contract.is_active()
        assert contract.status == ContractStatus.EXPIRED
    
    def test_break_contract(self):
        """Test breaking a contract early"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            breaking_penalty=100.0,
            duration=20
        )
        contract.sign()
        
        penalty = contract.break_contract()
        
        assert penalty == 100.0
        assert contract.status == ContractStatus.BROKEN
        assert not contract.is_active()
    
    def test_break_inactive_contract(self):
        """Test breaking inactive contract returns 0"""
        contract = SupplierContract(contract_id=1, supplier_id=0)
        
        penalty = contract.break_contract()
        assert penalty == 0.0
    
    def test_copy(self):
        """Test copying contract"""
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=15.0
        )
        contract.sign()
        contract.total_ordered = 50.0
        
        copied = contract.copy()
        
        assert copied.contract_id == 1
        assert copied.is_active()
        assert copied.total_ordered == 50.0
        assert copied is not contract
    
    def test_invalid_committed_quantity(self):
        """Test validation of committed quantity"""
        with pytest.raises(InvalidParameterError):
            SupplierContract(contract_id=1, supplier_id=0, committed_quantity=-5.0)
    
    def test_invalid_discount_rate(self):
        """Test validation of discount rate"""
        with pytest.raises(InvalidParameterError):
            SupplierContract(contract_id=1, supplier_id=0, discount_rate=1.5)
    
    def test_invalid_penalty_rate(self):
        """Test validation of penalty rate"""
        with pytest.raises(InvalidParameterError):
            SupplierContract(contract_id=1, supplier_id=0, penalty_rate=-0.1)
    
    def test_invalid_duration(self):
        """Test validation of duration"""
        with pytest.raises(InvalidParameterError):
            SupplierContract(contract_id=1, supplier_id=0, duration=0)


class TestContractManager:
    """Tests for ContractManager class"""
    
    def test_basic_construction(self):
        """Test basic construction"""
        manager = ContractManager()
        
        assert len(manager.contracts) == 0
        assert len(manager.available_contracts) == 0
    
    def test_add_available_contract(self):
        """Test adding available contracts"""
        manager = ContractManager()
        contract = SupplierContract(contract_id=1, supplier_id=0)
        
        manager.add_available_contract(contract)
        
        assert len(manager.available_contracts) == 1
    
    def test_sign_contract(self):
        """Test signing an available contract"""
        manager = ContractManager()
        contract = SupplierContract(contract_id=1, supplier_id=0)
        manager.add_available_contract(contract)
        
        signed = manager.sign_contract(1)
        
        assert signed is contract
        assert signed.is_active()
        assert 1 in manager.contracts
        assert len(manager.available_contracts) == 0
    
    def test_sign_nonexistent_contract(self):
        """Test signing nonexistent contract returns None"""
        manager = ContractManager()
        
        result = manager.sign_contract(999)
        assert result is None
    
    def test_get_active_contracts(self):
        """Test getting active contracts"""
        manager = ContractManager()
        
        c1 = SupplierContract(contract_id=1, supplier_id=0)
        c2 = SupplierContract(contract_id=2, supplier_id=1)
        c3 = SupplierContract(contract_id=3, supplier_id=0)
        
        c1.sign()
        c2.sign()
        # c3 not signed
        
        manager.contracts = {1: c1, 2: c2, 3: c3}
        
        active = manager.get_active_contracts()
        assert len(active) == 2
        
        # Filter by supplier
        supplier_0 = manager.get_active_contracts(supplier_id=0)
        assert len(supplier_0) == 1
        assert supplier_0[0].contract_id == 1
    
    def test_get_contract_for_order(self):
        """Test getting applicable contract for order"""
        manager = ContractManager()
        
        contract = SupplierContract(contract_id=1, supplier_id=0)
        contract.sign()
        manager.contracts[1] = contract
        
        # Found
        result = manager.get_contract_for_order(supplier_id=0)
        assert result is contract
        
        # Not found
        result = manager.get_contract_for_order(supplier_id=999)
        assert result is None
    
    def test_process_period_all(self):
        """Test processing all contracts for a period"""
        manager = ContractManager()
        
        c1 = SupplierContract(
            contract_id=1,
            supplier_id=0,
            committed_quantity=10.0,
            penalty_rate=0.5
        )
        c1.sign()
        
        c2 = SupplierContract(
            contract_id=2,
            supplier_id=1,
            committed_quantity=15.0,
            penalty_rate=0.4
        )
        c2.sign()
        
        manager.contracts = {1: c1, 2: c2}
        
        orders = {0: 8.0, 1: 10.0}  # Both under-order
        _, penalties = manager.process_period_all(orders)
        
        # c1: (10-8) * 0.5 = 1.0
        # c2: (15-10) * 0.4 = 2.0
        expected_penalty = 1.0 + 2.0
        assert_almost_equal(penalties, expected_penalty)
    
    def test_generate_available_contracts(self):
        """Test generating random contracts"""
        manager = ContractManager()
        suppliers = [{"id": 0}, {"id": 1}]
        
        manager.generate_available_contracts(suppliers, num_contracts=5, seed=42)
        
        assert len(manager.available_contracts) == 5
        for c in manager.available_contracts:
            assert c.status == ContractStatus.PENDING
            assert c.supplier_id in [0, 1]
    
    def test_copy(self):
        """Test copying manager"""
        manager = ContractManager()
        contract = SupplierContract(contract_id=1, supplier_id=0)
        contract.sign()
        manager.contracts[1] = contract
        
        copied = manager.copy()
        
        assert 1 in copied.contracts
        assert copied.contracts[1] is not contract


class TestContractCostCalculations:
    """Tests for contract cost calculation functions"""
    
    def test_calculate_contract_costs(self):
        """Test calculating contract costs"""
        contracts = {
            1: SupplierContract(
                contract_id=1,
                supplier_id=0,
                committed_quantity=10.0,
                discount_rate=0.20,
                penalty_rate=0.5
            )
        }
        contracts[1].sign()
        
        orders = {0: 8.0}
        base_prices = {0: 10.0}
        
        costs = calculate_contract_costs(contracts, orders, base_prices)
        
        # Discount: 8 units * 10 price * 0.20 rate = 16.0
        assert_almost_equal(costs['discount_savings'], 16.0)
        
        # Penalty: (10 - 8) * 0.5 = 1.0
        assert_almost_equal(costs['under_order_penalty'], 1.0)
        
        # Net: 16 - 1 = 15
        assert_almost_equal(costs['net_contract_benefit'], 15.0)
    
    def test_calculate_costs_inactive_contract(self):
        """Test costs for inactive contract"""
        contracts = {
            1: SupplierContract(contract_id=1, supplier_id=0)
            # Not signed
        }
        
        orders = {0: 5.0}
        base_prices = {0: 10.0}
        
        costs = calculate_contract_costs(contracts, orders, base_prices)
        
        assert costs['discount_savings'] == 0.0
        assert costs['under_order_penalty'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
