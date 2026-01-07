"""
Examples and plotting utilities for Predator: Badlands simulation.
Provides functions for running example simulations and generating analysis plots.
"""
import argparse
import os
import sys
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from world import World, BatchSimulationRunner
import config

logger = logging.getLogger(__name__)


def run_example_simulation(config_name: str = "standard", seed: int = 42) -> Dict[str, Any]:
    """
    Run a single example simulation and return results.
    
    Args:
        config_name: Configuration preset to use
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing simulation results and data
    """
    print(f"Running example simulation with config '{config_name}' and seed {seed}")
    
    # Get configuration
    config_dict = config.get_config(config_name)
    config_dict['seed'] = seed
    
    # Run simulation
    world = World(config_dict)
    summary = world.run_simulation()
    
    return {
        'summary': summary,
        'time_series': world.time_series_data,
        'world_state': world.get_world_state(),
        'config': config_dict
    }


def generate_plots(csv_path: str, output_dir: str) -> None:
    """
    Generate analysis plots from batch simulation CSV results.
    
    Args:
        csv_path: Path to CSV file containing batch results
        output_dir: Directory to save plots
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} simulation results from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Set up matplotlib style
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Survival Rate Plot
    plt.figure(figsize=fig_size)
    
    survival_rate = df['dek_survived'].mean()
    boss_defeat_rate = df['boss_defeated'].mean()
    
    categories = ['Dek Survival', 'Boss Defeat']
    rates = [survival_rate, boss_defeat_rate]
    colors = ['#2E8B57', '#DC143C']
    
    bars = plt.bar(categories, rates, color=colors, alpha=0.8, edgecolor='black')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Success Rate')
    plt.title('Simulation Success Rates')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'survival_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Steps Distribution
    plt.figure(figsize=fig_size)
    
    plt.hist(df['steps'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['steps'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df["steps"].mean():.1f}')
    plt.axvline(df['steps'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {df["steps"].median():.1f}')
    
    plt.xlabel('Simulation Steps')
    plt.ylabel('Frequency')
    plt.title('Distribution of Simulation Duration')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steps_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Honor Trajectory (for survivors)
    if 'dek_final_honor' in df.columns:
        survivors_df = df[df['dek_survived'] == True]
        
        if len(survivors_df) > 0:
            plt.figure(figsize=fig_size)
            
            plt.hist(survivors_df['dek_final_honor'], bins=15, alpha=0.7, 
                    color='gold', edgecolor='black')
            plt.axvline(survivors_df['dek_final_honor'].mean(), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean: {survivors_df["dek_final_honor"].mean():.1f}')
            
            plt.xlabel('Final Honor')
            plt.ylabel('Frequency')
            plt.title('Distribution of Final Honor (Survivors Only)')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'honor_trajectory.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Boss Defeat Analysis
    plt.figure(figsize=fig_size)
    
    # Create contingency table
    outcome_counts = pd.crosstab(df['dek_survived'], df['boss_defeated'], margins=True)
    
    # Visualization as stacked bar
    categories = ['Dek Died', 'Dek Survived']
    boss_not_defeated = [
        len(df[(df['dek_survived'] == False) & (df['boss_defeated'] == False)]),
        len(df[(df['dek_survived'] == True) & (df['boss_defeated'] == False)])
    ]
    boss_defeated = [
        len(df[(df['dek_survived'] == False) & (df['boss_defeated'] == True)]),
        len(df[(df['dek_survived'] == True) & (df['boss_defeated'] == True)])
    ]
    
    width = 0.6
    x = np.arange(len(categories))
    
    p1 = plt.bar(x, boss_not_defeated, width, label='Boss Not Defeated', color='lightcoral')
    p2 = plt.bar(x, boss_defeated, width, bottom=boss_not_defeated, 
                label='Boss Defeated', color='lightgreen')
    
    plt.xlabel('Dek Outcome')
    plt.ylabel('Number of Runs')
    plt.title('Simulation Outcomes')
    plt.xticks(x, categories)
    plt.legend()
    
    # Add count labels
    for i, (cat, not_def, defeated) in enumerate(zip(categories, boss_not_defeated, boss_defeated)):
        if not_def > 0:
            plt.text(i, not_def/2, str(not_def), ha='center', va='center', fontweight='bold')
        if defeated > 0:
            plt.text(i, not_def + defeated/2, str(defeated), ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boss_defeat_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Performance Metrics Comparison
    if all(col in df.columns for col in ['trophies_collected', 'dek_damage_dealt_to_boss', 'monsters_killed']):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trophies
        axes[0, 0].hist(df['trophies_collected'], bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].set_title('Trophies Collected')
        axes[0, 0].set_xlabel('Number of Trophies')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(alpha=0.3)
        
        # Boss Damage
        axes[0, 1].hist(df['dek_damage_dealt_to_boss'], bins=15, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Boss Damage Dealt')
        axes[0, 1].set_xlabel('Total Damage')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(alpha=0.3)
        
        # Monsters Killed
        axes[1, 0].hist(df['monsters_killed'], bins=10, alpha=0.7, color='brown', edgecolor='black')
        axes[1, 0].set_title('Monsters Killed')
        axes[1, 0].set_xlabel('Number of Monsters')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(alpha=0.3)
        
        # Correlation scatter: Honor vs Boss Damage
        survivors = df[df['dek_survived'] == True]
        if len(survivors) > 0:
            axes[1, 1].scatter(survivors['dek_damage_dealt_to_boss'], survivors['dek_final_honor'], 
                              alpha=0.6, color='blue')
            axes[1, 1].set_xlabel('Boss Damage Dealt')
            axes[1, 1].set_ylabel('Final Honor')
            axes[1, 1].set_title('Honor vs Boss Damage (Survivors)')
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Summary Statistics Plot
    plt.figure(figsize=(10, 6))
    
    stats_data = {
        'Total Runs': len(df),
        'Dek Survival Rate': f"{survival_rate:.1%}",
        'Boss Defeat Rate': f"{boss_defeat_rate:.1%}",
        'Mean Steps': f"{df['steps'].mean():.1f}",
        'Mean Honor (Survivors)': f"{df[df['dek_survived']]['dek_final_honor'].mean():.1f}" if len(df[df['dek_survived']]) > 0 else "N/A",
        'Total Trophies': df['trophies_collected'].sum() if 'trophies_collected' in df.columns else "N/A"
    }
    
    # Create text-based summary
    plt.text(0.1, 0.9, 'Simulation Statistics Summary', fontsize=20, fontweight='bold', 
             transform=plt.gca().transAxes)
    
    y_pos = 0.8
    for key, value in stats_data.items():
        plt.text(0.1, y_pos, f'{key}: {value}', fontsize=14, 
                transform=plt.gca().transAxes)
        y_pos -= 0.1
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated plots in: {output_dir}")
    print("Files created:")
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            print(f"  - {filename}")


def run_visual_example(config_name: str = "basic", seed: int = 42, max_steps: int = 50) -> None:
    """
    Run a short simulation with step-by-step visualization output.
    
    Args:
        config_name: Configuration preset
        seed: Random seed
        max_steps: Maximum steps to run (for demonstration)
    """
    print(f"Running visual example with {max_steps} steps...")
    
    config_dict = config.get_config(config_name)
    config_dict['seed'] = seed
    config_dict['max_steps'] = max_steps
    
    world = World(config_dict)
    world.setup_simulation()
    
    print("Initial state:")
    print(world.visualize_grid())
    
    step_interval = max(1, max_steps // 10)  # Show ~10 snapshots
    
    while world.step():
        if world.step_count % step_interval == 0:
            print(f"\nStep {world.step_count}:")
            print(world.visualize_grid())
            
            # Show key metrics
            metrics = world.metrics_history[-1]
            print(f"Dek: Health={metrics.dek_health}, Stamina={metrics.dek_stamina}, Honor={metrics.dek_honor:.1f}")
            print(f"Boss: Health={metrics.boss_health}, Alive={metrics.boss_alive}")
            print(f"Monsters Alive: {metrics.monsters_alive}")
            print(f"Thia: Functional={metrics.thia_functional}, Being Carried={metrics.thia_being_carried}")
    
    print(f"\nFinal state after {world.step_count} steps:")
    print(world.visualize_grid())
    print(f"Simulation result: {world.simulation_result}")


def main():
    """Main function for running examples."""
    parser = argparse.ArgumentParser(description="Predator: Badlands Examples and Analysis")
    
    parser.add_argument(
        'command',
        choices=['single', 'batch', 'visual', 'plot'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        choices=list(config.PRESETS.keys()),
        default='standard',
        help='Configuration preset'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=20,
        help='Number of runs for batch mode'
    )
    
    parser.add_argument(
        '--csv',
        help='CSV file path for plot generation'
    )
    
    parser.add_argument(
        '--output',
        default='results',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.command == 'single':
        result = run_example_simulation(args.config, args.seed)
        print("\nSimulation Summary:")
        summary = result['summary']
        print(f"Steps: {summary.steps}")
        print(f"Dek Survived: {summary.dek_survived}")
        print(f"Boss Defeated: {summary.boss_defeated}")
        print(f"Final Honor: {summary.dek_final_honor:.1f}")
        print(f"Trophies: {summary.trophies_collected}")
        
    elif args.command == 'batch':
        config_dict = config.get_config(args.config)
        config_dict['seed'] = args.seed
        
        runner = BatchSimulationRunner(config_dict)
        results = runner.run_batch(args.runs, args.seed)
        
        # Export and analyze
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.output, f"example_batch_{timestamp}.csv")
        os.makedirs(args.output, exist_ok=True)
        
        runner.export_summary_csv(csv_path)
        
        # Generate plots
        plot_dir = os.path.join(args.output, f"plots_{timestamp}")
        generate_plots(csv_path, plot_dir)
        
        # Display stats
        stats = runner.get_statistics()
        print(f"\nBatch Results ({args.runs} runs):")
        print(f"Dek Survival Rate: {stats['dek_survival_rate']:.1%}")
        print(f"Boss Defeat Rate: {stats['boss_defeat_rate']:.1%}")
        print(f"Mean Steps: {stats['mean_steps']:.1f}")
        print(f"Results saved to: {csv_path}")
        print(f"Plots saved to: {plot_dir}")
        
    elif args.command == 'visual':
        run_visual_example(args.config, args.seed, 50)
        
    elif args.command == 'plot':
        if not args.csv:
            print("Error: --csv argument required for plot command")
            return
        plot_dir = os.path.join(args.output, "plots")
        generate_plots(args.csv, plot_dir)


if __name__ == "__main__":
    main()