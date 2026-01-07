import argparse,logging,os,sys
from datetime import datetime
from typing import Dict, Any
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

import config
from world import World, BatchSimulationRunner


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('simulation.log', mode='a')
        ]
    )


def run_single_simulation(args) -> None:
    """Run a single simulation with specified parameters."""
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config_dict = config.get_config(args.config)
    
    # Override with command line arguments
    if args.seed is not None:
        config_dict['seed'] = args.seed
    if args.grid_size is not None:
        config_dict['grid_size'] = args.grid_size  
    if args.max_steps is not None:
        config_dict['max_steps'] = args.max_steps
    if args.monsters is not None:
        config_dict['monsters'] = args.monsters
    if args.boss_health is not None:
        config_dict['boss_health'] = args.boss_health
        
    # Set up results directory
    results_dir = args.results_dir or 'results'
    os.makedirs(results_dir, exist_ok=True)
    config_dict['results_dir'] = results_dir
    
    logger.info(f"Starting single simulation with config: {args.config}")
    logger.info(f"Parameters: seed={config_dict['seed']}, "
                f"grid_size={config_dict['grid_size']}, "
                f"max_steps={config_dict['max_steps']}")
    
    # Run simulation
    world = World(config_dict)
    summary = world.run_simulation()
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Time series data
    timeseries_path = os.path.join(results_dir, f"single_run_{timestamp}_timeseries.csv")
    world.export_time_series(timeseries_path)
    
    # Summary results
    summary_path = os.path.join(results_dir, f"single_run_{timestamp}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Predator: Badlands Simulation Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Seed: {summary.seed}\n")
        f.write(f"Steps: {summary.steps}\n")
        f.write(f"Dek Survived: {summary.dek_survived}\n")
        f.write(f"Boss Defeated: {summary.boss_defeated}\n")
        f.write(f"Final Honor: {summary.dek_final_honor:.1f}\n")
        f.write(f"Trophies Collected: {summary.trophies_collected}\n")
        f.write(f"Boss Damage Dealt: {summary.dek_damage_dealt_to_boss}\n")
        f.write(f"Thia Repaired: {summary.thia_repaired}\n")
        f.write(f"Monsters Killed: {summary.monsters_killed}\n")
        f.write(f"Simulation Time: {summary.simulation_time:.2f}s\n\n")
        
        # Final world state
        f.write("Final Grid State:\n")
        f.write(world.visualize_grid())
        
    logger.info(f"Single simulation complete!")
    logger.info(f"Results: Dek survived={summary.dek_survived}, "
                f"Boss defeated={summary.boss_defeated}, "
                f"Honor={summary.dek_final_honor:.1f}")
    logger.info(f"Time series exported to: {timeseries_path}")
    logger.info(f"Summary exported to: {summary_path}")


def run_batch_simulation(args) -> None:
    """Run batch simulation for statistical analysis."""
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config_dict = config.get_config(args.config)
    
    # Override with command line arguments
    if args.seed is not None:
        seed_start = args.seed
    else:
        seed_start = config_dict['seed']
        
    if args.grid_size is not None:
        config_dict['grid_size'] = args.grid_size
    if args.max_steps is not None:
        config_dict['max_steps'] = args.max_steps
    if args.monsters is not None:
        config_dict['monsters'] = args.monsters
    if args.boss_health is not None:
        config_dict['boss_health'] = args.boss_health
        
    # Set up results directory
    results_dir = args.results_dir or 'results'
    os.makedirs(results_dir, exist_ok=True)
    config_dict['results_dir'] = results_dir
    
    logger.info(f"Starting batch simulation: {args.runs} runs")
    logger.info(f"Config: {args.config}, seed_start: {seed_start}")
    
    # Run batch
    runner = BatchSimulationRunner(config_dict)
    results = runner.run_batch(args.runs, seed_start)
    
    # Export summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = os.path.join(results_dir, f"predator_sim_summary_{timestamp}.csv")
    runner.export_summary_csv(summary_csv_path)
    
    # Calculate and display statistics
    stats = runner.get_statistics()
    
    logger.info(f"Batch simulation complete!")
    logger.info(f"Results summary:")
    logger.info(f"  Total runs: {stats['total_runs']}")
    logger.info(f"  Dek survival rate: {stats['dek_survival_rate']:.1%}")
    logger.info(f"  Boss defeat rate: {stats['boss_defeat_rate']:.1%}")
    logger.info(f"  Mean steps: {stats['mean_steps']:.1f}")
    logger.info(f"  Median steps: {stats['median_steps']}")
    logger.info(f"  Mean honor (survivors): {stats['mean_honor']:.1f}")
    logger.info(f"  Thia repair rate: {stats['thia_repair_rate']:.1%}")
    
    # Export detailed statistics
    stats_path = os.path.join(results_dir, f"batch_statistics_{timestamp}.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Predator: Badlands Batch Simulation Statistics\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Number of runs: {args.runs}\n")
        f.write(f"Seed range: {seed_start} to {seed_start + args.runs - 1}\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"  Dek Survival Rate: {stats['dek_survival_rate']:.1%}\n")
        f.write(f"  Boss Defeat Rate: {stats['boss_defeat_rate']:.1%}\n")
        f.write(f"  Mean Steps per Run: {stats['mean_steps']:.1f}\n")
        f.write(f"  Median Steps per Run: {stats['median_steps']}\n")
        f.write(f"  Mean Honor (Survivors): {stats['mean_honor']:.1f}\n")
        f.write(f"  Total Trophies Collected: {stats['total_trophies']}\n")
        f.write(f"  Total Boss Damage Dealt: {stats['total_boss_damage']}\n")
        f.write(f"  Thia Repair Rate: {stats['thia_repair_rate']:.1%}\n\n")
        
        # Individual run results
        f.write("Individual Run Results:\n")
        f.write("Run | Dek Survived | Boss Defeated | Steps | Honor | Trophies\n")
        f.write("-" * 65 + "\n")
        for result in results:
            f.write(f"{result.run_id:3d} | {str(result.dek_survived):12s} | "
                   f"{str(result.boss_defeated):13s} | {result.steps:5d} | "
                   f"{result.dek_final_honor:5.1f} | {result.trophies_collected:8d}\n")
    
    logger.info(f"Summary CSV exported to: {summary_csv_path}")
    logger.info(f"Statistics exported to: {stats_path}")
    
    # Generate plots if matplotlib available
    try:
        sys.path.append(os.path.join(current_dir, 'examples'))
        from run_example import generate_plots
        plot_dir = os.path.join(results_dir, f"plots_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        generate_plots(summary_csv_path, plot_dir)
        logger.info(f"Plots generated in: {plot_dir}")
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Predator: Badlands Multi-Agent Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with standard config
  python -m src.main --mode single --config standard --seed 42
  
  # Batch run for analysis (20 runs)
  python -m src.main --mode batch --config expert --runs 20 --seed 0
  
  # Custom parameters
  python -m src.main --mode single --grid-size 30 --monsters 50 --boss-health 300
  
  # Generate plots from existing CSV
  python -m src.examples.run_example plot --csv results/predator_sim_summary.csv
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['single', 'batch'], 
        default='single',
        help='Simulation mode: single run or batch analysis'
    )
    
    parser.add_argument(
        '--config',
        choices=list(config.PRESETS.keys()),
        default='standard',
        help='Configuration preset to use'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (default: from config)'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=20,
        help='Number of runs for batch mode (default: 20)'
    )
    
    parser.add_argument(
        '--grid-size',
        type=int,
        help='Grid size (creates NxN grid)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        help='Maximum simulation steps'
    )
    
    parser.add_argument(
        '--monsters',
        type=int,
        help='Number of monsters to spawn'
    )
    
    parser.add_argument(
        '--boss-health',
        type=int,
        help='Boss health points'
    )
    
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory for output files (default: results)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Predator: Badlands Simulation v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Predator: Badlands Simulation Starting")
    logger.info(f"Mode: {args.mode}, Config: {args.config}")
    
    try:
        if args.mode == 'single':
            run_single_simulation(args)
        elif args.mode == 'batch':
            run_batch_simulation(args)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Simulation completed successfully")


if __name__ == "__main__":
    main()