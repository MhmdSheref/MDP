"""
Basic Simulation Example

Demonstrates how to set up and run a perishable inventory MDP simulation
with multiple suppliers and different ordering policies, including visualizations.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from inventory_sim import (
    PerishableInventoryMDP,
    PoissonDemand,
    CostParameters,
    run_episode
)
from inventory_sim.env import create_simple_mdp
from inventory_agents import (
    BaseStockPolicy,
    TailoredBaseSurgePolicy,
    DoNothingPolicy,
    ConstantOrderPolicy
)

# Import plotting utilities
try:
    import matplotlib
    # Use Agg backend for file output (works without display)
    # Change to 'TkAgg' if you want interactive plots
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from inventory_sim.plotting import (
        plot_policy_comparison,
        plot_simulation_trace,
        plot_cost_breakdown,
        plot_seasonal_demand,
        plot_inventory_buckets
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Note: matplotlib not available. Plots will be skipped.")


def run_policy_comparison():
    """
    Compare different ordering policies on the same MDP.
    """
    print("=" * 60)
    print("POLICY COMPARISON SIMULATION")
    print("=" * 60)
    
    # Create a two-supplier MDP
    mdp = create_simple_mdp(
        shelf_life=5,
        num_suppliers=2,
        mean_demand=10.0,
        fast_lead_time=1,
        slow_lead_time=3,
        fast_cost=2.0,
        slow_cost=1.0
    )
    
    print("\nMDP Configuration:")
    print(f"  - Shelf life: 5 periods")
    print(f"  - Mean demand: 10 units/period (Poisson)")
    print(f"  - Supplier 0: Lead time=1, Cost=2.0 (fast, expensive)")
    print(f"  - Supplier 1: Lead time=3, Cost=1.0 (slow, cheap)")
    
    # Initial state
    initial_inventory = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    
    # Define policies to compare
    policies = {
        "Do Nothing": DoNothingPolicy(),
        "Constant (10/period from slow)": ConstantOrderPolicy({0: 0.0, 1: 10.0}),
        "Base Stock (S=60)": BaseStockPolicy(target_level=60.0, supplier_id=1),
        "Tailored Base-Surge": TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=50.0,
            reorder_point=25.0
        )
    }
    
    # Run simulations
    num_periods = 200
    seed = 42
    
    print(f"\nRunning {num_periods}-period simulations...\n")
    print("-" * 60)
    
    results_summary = []
    
    for name, policy in policies.items():
        state = mdp.create_initial_state(initial_inventory=initial_inventory.copy())
        results, total_reward = run_episode(
            mdp, policy, num_periods=num_periods, seed=seed, initial_state=state
        )
        metrics = mdp.compute_inventory_metrics(results)
        
        results_summary.append({
            "policy": name,
            "total_reward": total_reward,
            "fill_rate": metrics["fill_rate"],
            "service_level": metrics["service_level"],
            "spoilage_rate": metrics["spoilage_rate"],
            "avg_inventory": metrics["average_inventory"],
            "avg_cost": metrics["average_cost"]
        })
        
        print(f"Policy: {name}")
        print(f"  Fill Rate:      {metrics['fill_rate']:.2%}")
        print(f"  Service Level:  {metrics['service_level']:.2%}")
        print(f"  Spoilage Rate:  {metrics['spoilage_rate']:.2%}")
        print(f"  Avg Inventory:  {metrics['average_inventory']:.1f}")
        print(f"  Avg Cost/Period: {metrics['average_cost']:.2f}")
        print(f"  Total Reward:   {total_reward:.2f}")
        print()
    
    # Find best policy
    best = max(results_summary, key=lambda x: x["total_reward"])
    print("-" * 60)
    print(f"Best Policy: {best['policy']} (Total Reward: {best['total_reward']:.2f})")
    print("=" * 60)
    
    # Generate plot
    if PLOTTING_AVAILABLE:
        print("\nGenerating policy comparison plot...")
        fig = plot_policy_comparison(results_summary, save_path='plots/policy_comparison.png')
        plt.close(fig)
    
    return results_summary


def run_detailed_trace():
    """
    Run a detailed simulation trace showing Tailored Base-Surge (TBS) policy.
    
    This demonstrates the two-supplier TBS policy where:
    - Slow/cheap supplier (ID=1) handles base demand
    - Fast/expensive supplier (ID=0) handles surge demand
    """
    print("\n" + "=" * 60)
    print("DETAILED TBS SIMULATION TRACE (10 periods)")
    print("=" * 60)
    
    # Use same configuration as policy comparison for consistency
    mdp = create_simple_mdp(
        shelf_life=5,
        num_suppliers=2,
        mean_demand=10.0,
        fast_lead_time=1,
        slow_lead_time=3,
        fast_cost=2.0,
        slow_cost=1.0
    )
    
    state = mdp.create_initial_state(
        initial_inventory=np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    )
    
    # TBS policy: base stock level for slow supplier, reorder point triggers fast supplier
    policy = TailoredBaseSurgePolicy(
        slow_supplier_id=1,       # Cheap supplier for base demand
        fast_supplier_id=0,       # Expensive supplier for emergencies
        base_stock_level=50.0,    # Target level for slow supplier
        reorder_point=25.0        # Trigger fast supplier when below this
    )
    
    print(f"\nConfiguration:")
    print(f"  Shelf life: 5 periods")
    print(f"  Supplier 0 (Fast/Expensive): Lead time=1, Cost=2.0")
    print(f"  Supplier 1 (Slow/Cheap): Lead time=3, Cost=1.0")
    print(f"  TBS Base Stock Level: 50")
    print(f"  TBS Reorder Point: 25")
    print(f"\nInitial Inventory: {state.inventory} (Total: {state.total_inventory:.0f})")
    print()
    
    np.random.seed(123)
    
    # Collect trace data for plotting
    trace_data = []
    inventory_history = [state.inventory.copy()]
    
    for t in range(10):
        # Calculate positions for visibility
        ip_slow = state.total_inventory + state.pipelines[1].total_in_pipeline() - state.backorders
        ip_total = state.inventory_position
        
        action = policy.act(state, mdp)
        result = mdp.step(state, action)
        
        # Format inventory as integers for cleaner display
        inv_display = [int(x) for x in result.next_state.inventory]
        
        print(f"Period {t+1}:")
        print(f"  IP (slow only):   {ip_slow:.0f} | IP (total): {ip_total:.0f}")
        print(f"  Orders:           Slow(1)={int(action[1])}, Fast(0)={int(action[0])}")
        print(f"  Demand:           {int(result.demand_realized)}")
        print(f"  Sales:            {int(result.sales)}")
        print(f"  Arrivals:         {int(result.arrivals)}")
        print(f"  Spoiled:          {int(result.spoiled)}")
        print(f"  Inventory:        {inv_display}")
        print(f"  Period Cost:      {result.costs.total_cost:.2f}")
        print()
        
        # Store trace data for plotting
        trace_data.append({
            'period': t + 1,
            'demand': result.demand_realized,
            'sales': result.sales,
            'arrivals': result.arrivals,
            'spoiled': result.spoiled,
            'inventory_total': result.next_state.total_inventory,
            'order_slow': action[1],
            'order_fast': action[0],
            'ip_total': ip_total,
            'ip_slow': ip_slow,
            'cost': result.costs.total_cost,
            'base_stock': 50,
            'reorder_point': 25
        })
        inventory_history.append(result.next_state.inventory.copy())
        
        state = result.next_state
    
    # Generate plots
    if PLOTTING_AVAILABLE:
        print("Generating simulation trace plots...")
        fig1 = plot_simulation_trace(trace_data, save_path='plots/simulation_trace.png')
        plt.close(fig1)
        
        fig2 = plot_inventory_buckets(inventory_history, shelf_life=5, save_path='plots/inventory_buckets.png')
        plt.close(fig2)
    
    return trace_data


def run_seasonal_demand():
    """
    Demonstrate simulation with seasonal demand.
    """
    print("\n" + "=" * 60)
    print("SEASONAL DEMAND SIMULATION")
    print("=" * 60)
    
    from inventory_sim.core.demand import SeasonalDemand
    
    # Seasonal demand: higher in "summer" (periods 3-9 of 12-period cycle)
    demand_process = SeasonalDemand(
        base_rate=10.0,
        amplitude=0.4,  # 40% variation
        period=12,
        phase=0.0
    )
    
    suppliers = [
        {"id": 0, "lead_time": 2, "unit_cost": 1.5, "capacity": 50, "moq": 5}
    ]
    
    cost_params = CostParameters.age_dependent_holding(
        shelf_life=6,
        base_holding=0.5,
        age_premium=0.3,
        shortage_cost=15.0,
        spoilage_cost=8.0
    )
    
    mdp = PerishableInventoryMDP(
        shelf_life=6,
        suppliers=suppliers,
        demand_process=demand_process,
        cost_params=cost_params
    )
    
    state = mdp.create_initial_state(
        initial_inventory=np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0]),
        initial_exogenous=np.array([0.0])  # Start at period 0
    )
    
    # Use TBS-inspired policy adapted for single supplier
    policy = BaseStockPolicy(target_level=70.0, supplier_id=0)
    
    print("\nRunning 36-period simulation (3 seasonal cycles)...")
    
    results, total_reward = run_episode(mdp, policy, num_periods=36, seed=42, initial_state=state)
    
    # Analyze by season
    demands_by_period = [r.demand_realized for r in results]
    
    print("\nDemand Pattern (by period in cycle):")
    demand_by_cycle = []
    for cycle in range(3):
        start = cycle * 12
        cycle_demands = demands_by_period[start:start+12]
        demand_by_cycle.append(cycle_demands)
        print(f"  Cycle {cycle+1}: {[f'{d:.0f}' for d in cycle_demands]}")
    
    metrics = mdp.compute_inventory_metrics(results)
    print(f"\nOverall Metrics:")
    print(f"  Fill Rate:      {metrics['fill_rate']:.2%}")
    print(f"  Service Level:  {metrics['service_level']:.2%}")
    print(f"  Spoilage Rate:  {metrics['spoilage_rate']:.2%}")
    
    # Generate plot
    if PLOTTING_AVAILABLE:
        print("\nGenerating seasonal demand plot...")
        fig = plot_seasonal_demand(demand_by_cycle, period_length=12, save_path='plots/seasonal_demand.png')
        plt.close(fig)
    
    return demand_by_cycle


def run_cost_analysis():
    """
    Analyze cost breakdown over a simulation.
    """
    print("\n" + "=" * 60)
    print("COST BREAKDOWN ANALYSIS")
    print("=" * 60)
    
    mdp = create_simple_mdp(
        shelf_life=5,
        num_suppliers=2,
        mean_demand=12.0,
        fast_lead_time=1,
        slow_lead_time=3,
        fast_cost=2.5,
        slow_cost=1.0
    )
    
    state = mdp.create_initial_state(
        initial_inventory=np.array([15.0, 15.0, 15.0, 15.0, 15.0])
    )
    
    policy = TailoredBaseSurgePolicy(
        slow_supplier_id=1,
        fast_supplier_id=0,
        base_stock_level=60.0,
        reorder_point=20.0
    )
    
    results, _ = run_episode(mdp, policy, num_periods=100, seed=42, initial_state=state)
    
    # Aggregate costs
    total_purchase = sum(r.costs.purchase_cost for r in results)
    total_holding = sum(r.costs.holding_cost for r in results)
    total_shortage = sum(r.costs.shortage_cost for r in results)
    total_spoilage = sum(r.costs.spoilage_cost for r in results)
    total_cost = sum(r.costs.total_cost for r in results)
    
    print("\nCost Breakdown (100 periods):")
    print(f"  Purchase Costs:  {total_purchase:,.2f} ({100*total_purchase/total_cost:.1f}%)")
    print(f"  Holding Costs:   {total_holding:,.2f} ({100*total_holding/total_cost:.1f}%)")
    print(f"  Shortage Costs:  {total_shortage:,.2f} ({100*total_shortage/total_cost:.1f}%)")
    print(f"  Spoilage Costs:  {total_spoilage:,.2f} ({100*total_spoilage/total_cost:.1f}%)")
    print(f"  " + "-" * 40)
    print(f"  TOTAL:           {total_cost:,.2f}")
    
    print(f"\nAverage Cost per Period: {total_cost/100:.2f}")
    
    # Generate plot
    if PLOTTING_AVAILABLE:
        print("\nGenerating cost breakdown plot...")
        cost_data = {
            'purchase': total_purchase,
            'holding': total_holding,
            'shortage': total_shortage,
            'spoilage': total_spoilage
        }
        fig = plot_cost_breakdown(cost_data, save_path='plots/cost_breakdown.png')
        plt.close(fig)
    
    return cost_data


if __name__ == "__main__":
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Run all simulations
    policy_results = run_policy_comparison()
    trace_data = run_detailed_trace()
    seasonal_data = run_seasonal_demand()
    cost_data = run_cost_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    if PLOTTING_AVAILABLE:
        print("\nPlots saved to ./plots/ directory:")
        print("  - policy_comparison.png")
        print("  - simulation_trace.png")
        print("  - inventory_buckets.png")
        print("  - seasonal_demand.png")
        print("  - cost_breakdown.png")
    print("=" * 60)

