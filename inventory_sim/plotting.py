"""
Plotting utilities for the Perishable Inventory MDP.

Provides visualization functions for simulation results,
policy comparisons, and inventory dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# Set a modern style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28965A',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'purple': '#7B2D8E',
    'teal': '#1B998B',
    'gray': '#5C6B73'
}

POLICY_COLORS = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]


def plot_policy_comparison(
    results: List[Dict],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar charts comparing policy performance.
    
    Args:
        results: List of dicts with policy results containing:
            - policy: policy name
            - fill_rate, service_level, spoilage_rate, avg_inventory, avg_cost, total_reward
        metrics: List of metrics to plot (default: all)
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    if metrics is None:
        metrics = ['fill_rate', 'service_level', 'spoilage_rate', 'avg_inventory', 'avg_cost']
    
    policies = [r['policy'] for r in results]
    n_policies = len(policies)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    metric_configs = {
        'fill_rate': {'title': 'Fill Rate', 'format': '{:.1%}', 'color': COLORS['success']},
        'service_level': {'title': 'Service Level', 'format': '{:.1%}', 'color': COLORS['primary']},
        'spoilage_rate': {'title': 'Spoilage Rate', 'format': '{:.1%}', 'color': COLORS['danger']},
        'avg_inventory': {'title': 'Avg Inventory', 'format': '{:.1f}', 'color': COLORS['teal']},
        'avg_cost': {'title': 'Avg Cost/Period', 'format': '{:.2f}', 'color': COLORS['warning']},
        'total_reward': {'title': 'Total Reward', 'format': '{:.0f}', 'color': COLORS['purple']}
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [r.get(metric, 0) for r in results]
        config = metric_configs.get(metric, {'title': metric, 'format': '{:.2f}', 'color': COLORS['gray']})
        
        bars = ax.bar(range(n_policies), values, color=config['color'], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = config['format'].format(val)
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        ax.set_title(config['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(n_policies))
        ax.set_xticklabels(policies, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(config['title'], fontsize=10)
        
        # Improve appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Use last subplot for total reward comparison
    if len(metrics) < 6:
        ax = axes[5]
        rewards = [r.get('total_reward', 0) for r in results]
        colors = [COLORS['success'] if r == max(rewards) else COLORS['gray'] for r in rewards]
        bars = ax.bar(range(n_policies), rewards, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, rewards):
            height = bar.get_height()
            ax.annotate(f'{val:,.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, -15 if val < 0 else 5),
                       textcoords="offset points",
                       ha='center', va='top' if val < 0 else 'bottom',
                       fontsize=10, fontweight='bold')
        
        ax.set_title('Total Reward (Higher is Better)', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(n_policies))
        ax.set_xticklabels(policies, rotation=30, ha='right', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide any unused subplots
    for idx in range(len(metrics) + 1, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Policy Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_simulation_trace(
    trace_data: List[Dict],
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series from a simulation trace.
    
    Args:
        trace_data: List of dicts per period containing:
            - period, demand, sales, arrivals, spoiled, inventory_total,
              order_slow, order_fast, ip_total, cost
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    periods = [d['period'] for d in trace_data]
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Plot 1: Inventory & Demand
    ax1 = axes[0, 0]
    ax1.fill_between(periods, [d['inventory_total'] for d in trace_data], 
                     alpha=0.3, color=COLORS['primary'], label='Inventory')
    ax1.plot(periods, [d['inventory_total'] for d in trace_data], 
             color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
    ax1.plot(periods, [d['demand'] for d in trace_data], 
             color=COLORS['danger'], linewidth=2, linestyle='--', marker='s', markersize=4, label='Demand')
    ax1.set_xlabel('Period', fontsize=10)
    ax1.set_ylabel('Units', fontsize=10)
    ax1.set_title('Inventory Level vs Demand', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(min(periods), max(periods))
    
    # Plot 2: Orders by Supplier
    ax2 = axes[0, 1]
    order_slow = [d.get('order_slow', 0) for d in trace_data]
    order_fast = [d.get('order_fast', 0) for d in trace_data]
    
    width = 0.35
    x = np.array(periods)
    ax2.bar(x - width/2, order_slow, width, label='Slow (Cheap)', color=COLORS['success'], alpha=0.8)
    ax2.bar(x + width/2, order_fast, width, label='Fast (Expensive)', color=COLORS['warning'], alpha=0.8)
    ax2.set_xlabel('Period', fontsize=10)
    ax2.set_ylabel('Order Quantity', fontsize=10)
    ax2.set_title('Orders by Supplier (TBS Policy)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xticks(periods)
    
    # Plot 3: Sales vs Arrivals
    ax3 = axes[1, 0]
    ax3.plot(periods, [d['sales'] for d in trace_data], 
             color=COLORS['success'], linewidth=2, marker='o', markersize=4, label='Sales')
    ax3.plot(periods, [d['arrivals'] for d in trace_data], 
             color=COLORS['teal'], linewidth=2, marker='^', markersize=4, label='Arrivals')
    ax3.set_xlabel('Period', fontsize=10)
    ax3.set_ylabel('Units', fontsize=10)
    ax3.set_title('Sales and Arrivals', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_xlim(min(periods), max(periods))
    
    # Plot 4: Spoilage
    ax4 = axes[1, 1]
    spoiled = [d['spoiled'] for d in trace_data]
    ax4.bar(periods, spoiled, color=COLORS['danger'], alpha=0.7, edgecolor='white')
    ax4.set_xlabel('Period', fontsize=10)
    ax4.set_ylabel('Spoiled Units', fontsize=10)
    ax4.set_title('Inventory Spoilage', fontsize=12, fontweight='bold')
    ax4.set_xticks(periods)
    
    # Plot 5: Inventory Position
    ax5 = axes[2, 0]
    ip_total = [d.get('ip_total', d['inventory_total']) for d in trace_data]
    ip_slow = [d.get('ip_slow', d['inventory_total']) for d in trace_data]
    
    ax5.plot(periods, ip_total, color=COLORS['primary'], linewidth=2, marker='o', markersize=4, label='IP Total')
    ax5.plot(periods, ip_slow, color=COLORS['secondary'], linewidth=2, marker='s', markersize=4, label='IP Slow Only')
    
    # Add threshold lines if available
    if 'base_stock' in trace_data[0]:
        ax5.axhline(y=trace_data[0]['base_stock'], color=COLORS['success'], linestyle='--', 
                   linewidth=1.5, label=f"Base Stock ({trace_data[0]['base_stock']})")
    if 'reorder_point' in trace_data[0]:
        ax5.axhline(y=trace_data[0]['reorder_point'], color=COLORS['warning'], linestyle='--', 
                   linewidth=1.5, label=f"Reorder Point ({trace_data[0]['reorder_point']})")
    
    ax5.set_xlabel('Period', fontsize=10)
    ax5.set_ylabel('Inventory Position', fontsize=10)
    ax5.set_title('Inventory Position Over Time', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_xlim(min(periods), max(periods))
    
    # Plot 6: Period Costs
    ax6 = axes[2, 1]
    costs = [d['cost'] for d in trace_data]
    ax6.fill_between(periods, costs, alpha=0.3, color=COLORS['purple'])
    ax6.plot(periods, costs, color=COLORS['purple'], linewidth=2, marker='o', markersize=4)
    ax6.set_xlabel('Period', fontsize=10)
    ax6.set_ylabel('Cost', fontsize=10)
    ax6.set_title('Period Cost', fontsize=12, fontweight='bold')
    ax6.set_xlim(min(periods), max(periods))
    
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Simulation Trace Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_cost_breakdown(
    cost_data: Dict[str, float],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a pie chart and bar chart of cost breakdown.
    
    Args:
        cost_data: Dict with cost components:
            - purchase, holding, shortage, spoilage
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    labels = ['Purchase', 'Holding', 'Shortage', 'Spoilage']
    values = [
        cost_data.get('purchase', 0),
        cost_data.get('holding', 0),
        cost_data.get('shortage', 0),
        cost_data.get('spoilage', 0)
    ]
    colors = [COLORS['primary'], COLORS['teal'], COLORS['danger'], COLORS['warning']]
    
    total = sum(values)
    
    # Pie chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        explode=(0.02, 0.02, 0.02, 0.02),
        shadow=True
    )
    ax1.set_title('Cost Distribution', fontsize=12, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Bar chart with values
    ax2 = axes[1]
    bars = ax2.barh(labels, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        width = bar.get_width()
        pct = 100 * val / total if total > 0 else 0
        ax2.annotate(f'{val:,.0f} ({pct:.1f}%)',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Cost', fontsize=10)
    ax2.set_title(f'Cost Breakdown (Total: {total:,.0f})', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Cost Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_seasonal_demand(
    demand_by_cycle: List[List[float]],
    period_length: int = 12,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot seasonal demand patterns across cycles.
    
    Args:
        demand_by_cycle: List of demand lists, one per cycle
        period_length: Length of one seasonal period
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    periods = list(range(1, period_length + 1))
    
    # Plot 1: Demand by cycle
    ax1 = axes[0]
    cycle_colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]
    
    for i, cycle in enumerate(demand_by_cycle):
        color = cycle_colors[i % len(cycle_colors)]
        ax1.plot(periods, cycle[:period_length], 
                marker='o', linewidth=2, markersize=6,
                label=f'Cycle {i+1}', color=color, alpha=0.8)
    
    ax1.set_xlabel('Period in Cycle', fontsize=10)
    ax1.set_ylabel('Demand', fontsize=10)
    ax1.set_title('Demand Pattern by Cycle', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xticks(periods)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Average demand with confidence band
    ax2 = axes[1]
    all_cycles = np.array([c[:period_length] for c in demand_by_cycle if len(c) >= period_length])
    
    if len(all_cycles) > 0:
        mean_demand = np.mean(all_cycles, axis=0)
        std_demand = np.std(all_cycles, axis=0)
        
        ax2.fill_between(periods, mean_demand - std_demand, mean_demand + std_demand,
                        alpha=0.3, color=COLORS['primary'], label='±1 Std Dev')
        ax2.plot(periods, mean_demand, color=COLORS['primary'], linewidth=3,
                marker='o', markersize=8, label='Mean Demand')
        
        # Add trend line
        z = np.polyfit(periods, mean_demand, 2)
        p = np.poly1d(z)
        ax2.plot(periods, p(periods), color=COLORS['danger'], linewidth=2,
                linestyle='--', label='Trend')
    
    ax2.set_xlabel('Period in Cycle', fontsize=10)
    ax2.set_ylabel('Demand', fontsize=10)
    ax2.set_title('Average Seasonal Pattern', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xticks(periods)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Seasonal Demand Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_inventory_buckets(
    inventory_history: List[np.ndarray],
    shelf_life: int,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot stacked area chart of inventory by expiry bucket.
    
    Args:
        inventory_history: List of inventory arrays per period
        shelf_life: Number of expiry buckets
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    periods = list(range(1, len(inventory_history) + 1))
    
    # Prepare data for stacking
    data = np.array(inventory_history)
    
    # Create color gradient from red (oldest) to green (freshest)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, shelf_life))
    
    # Stacked area chart
    ax.stackplot(periods, data.T, labels=[f'Bucket {i+1} ({"Oldest" if i==0 else "Freshest" if i==shelf_life-1 else ""})' 
                                           for i in range(shelf_life)],
                colors=colors, alpha=0.8)
    
    ax.set_xlabel('Period', fontsize=10)
    ax.set_ylabel('Inventory Units', fontsize=10)
    ax.set_title('Inventory by Expiry Bucket (FIFO)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(min(periods), max(periods))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig

