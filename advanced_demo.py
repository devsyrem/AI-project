#!/usr/bin/env python3
"""
Advanced Swarm Intelligence and Genetic Evolution Demonstration.
Showcases emergent behaviors, evolutionary dynamics, and complex multi-agent interactions.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# Add the src directory and subdirectories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
agents_dir = os.path.join(src_dir, 'agents')

sys.path.insert(0, src_dir)
sys.path.insert(0, agents_dir)

import config
from enhanced_world import AdvancedWorld
from swarm_intelligence import SwarmBehaviorType
from genetic_evolution import GeneType


def demonstrate_swarm_intelligence():
    """Demonstrate advanced swarm behaviors and coordination."""
    print("🧠 Advanced Swarm Intelligence Demonstration")
    print("=" * 60)
    
    # Create world with enhanced swarm capabilities
    swarm_config = config.get_config('expert')
    swarm_config.update({
        'grid_size': 20,
        'max_steps': 150,
        'monsters': 25,  # Larger swarm for better demonstration
        'max_monster_population': 40,
        'territorial_markers': True,
        'dynamic_hazards': True
    })
    
    world = AdvancedWorld(swarm_config)
    world.setup_simulation()
    
    print(f"Initial Setup:")
    print(f"  Grid Size: {world.grid.size}×{world.grid.size}")
    print(f"  Monsters: {len(world.evolutionary_monsters)}")
    print(f"  Swarm Behaviors Available: {[b.value for b in SwarmBehaviorType]}")
    
    # Track swarm metrics over time
    swarm_metrics = {
        'steps': [],
        'behavior_diversity': [],
        'coordination_scores': [],
        'pheromone_trails': [],
        'collective_memory': []
    }
    
    # Run simulation and track swarm evolution
    print(f"\nRunning swarm intelligence simulation...")
    step = 0
    while world.step() and step < 50:  # Limited for demo
        step += 1
        
        if step % 10 == 0:
            # Analyze current swarm state
            behavior_counts = {}
            for monster in world.evolutionary_monsters.values():
                if monster.alive:
                    behavior = monster.current_swarm_behavior.value
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            
            behavior_diversity = len(behavior_counts)
            coordination = world._calculate_swarm_coordination()
            
            swarm_state = world.swarm_intelligence.get_swarm_state()
            
            swarm_metrics['steps'].append(step)
            swarm_metrics['behavior_diversity'].append(behavior_diversity)
            swarm_metrics['coordination_scores'].append(coordination)
            swarm_metrics['pheromone_trails'].append(swarm_state.get('pheromone_trails', 0))
            swarm_metrics['collective_memory'].append(
                swarm_state.get('collective_memory', {}).get('tracked_enemies', 0)
            )
            
            print(f"  Step {step}: Behaviors={behavior_counts}, "
                  f"Coordination={coordination:.2f}, "
                  f"Pheromones={swarm_state.get('pheromone_trails', 0)}")
    
    # Analyze final swarm state
    final_behaviors = {}
    for monster in world.evolutionary_monsters.values():
        if monster.alive:
            behavior = monster.current_swarm_behavior.value
            final_behaviors[behavior] = final_behaviors.get(behavior, 0) + 1
    
    print(f"\nSwarm Intelligence Results:")
    print(f"  Final Behavior Distribution: {final_behaviors}")
    print(f"  Peak Coordination Score: {max(swarm_metrics['coordination_scores']):.3f}")
    print(f"  Final Pheromone Trails: {swarm_metrics['pheromone_trails'][-1] if swarm_metrics['pheromone_trails'] else 0}")
    
    # Create swarm intelligence visualization
    if len(swarm_metrics['steps']) > 1:
        plot_swarm_evolution(swarm_metrics)
    
    return world, swarm_metrics


def demonstrate_genetic_evolution():
    """Demonstrate genetic evolution and population dynamics."""
    print("\n🧬 Genetic Evolution Demonstration")
    print("=" * 60)
    
    # Create world optimized for evolutionary dynamics
    evolution_config = config.get_config('expert')
    evolution_config.update({
        'grid_size': 25,
        'max_steps': 300,
        'monsters': 30,
        'max_monster_population': 60,
        'breeding_season_frequency': 15,  # More frequent breeding
        'resource_variability': True
    })
    
    world = AdvancedWorld(evolution_config)
    world.setup_simulation()
    
    print(f"Initial Genetic Population:")
    initial_stats = world.genetic_evolution.population_dynamics.get_population_statistics()
    print(f"  Population Size: {initial_stats['size']}")
    print(f"  Average Fitness: {initial_stats['average_fitness']:.2f}")
    print(f"  Generation 0 Trait Averages:")
    for trait, value in initial_stats['trait_averages'].items():
        print(f"    {trait}: {value:.3f}")
    
    # Track evolutionary metrics
    evolution_metrics = {
        'steps': [],
        'population_size': [],
        'average_fitness': [],
        'genetic_diversity': [],
        'dominant_traits': {},
        'births': [],
        'deaths': []
    }
    
    # Initialize trait tracking
    for trait in GeneType:
        evolution_metrics['dominant_traits'][trait.value] = []
    
    print(f"\nRunning genetic evolution simulation...")
    step = 0
    while world.step() and step < 100:  # Limited for demo
        step += 1
        
        if step % 20 == 0:
            # Collect evolutionary data
            pop_stats = world.genetic_evolution.population_dynamics.get_population_statistics()
            
            evolution_metrics['steps'].append(step)
            evolution_metrics['population_size'].append(pop_stats['size'])
            evolution_metrics['average_fitness'].append(pop_stats['average_fitness'])
            evolution_metrics['genetic_diversity'].append(world._calculate_genetic_diversity())
            
            # Track dominant traits
            for trait_name, value in pop_stats['trait_averages'].items():
                if trait_name in evolution_metrics['dominant_traits']:
                    evolution_metrics['dominant_traits'][trait_name].append(value)
            
            print(f"  Step {step}: Population={pop_stats['size']}, "
                  f"Avg Fitness={pop_stats['average_fitness']:.2f}, "
                  f"Diversity={world._calculate_genetic_diversity():.3f}")
    
    # Analyze evolutionary results
    final_stats = world.genetic_evolution.population_dynamics.get_population_statistics()
    
    print(f"\nGenetic Evolution Results:")
    print(f"  Final Population: {final_stats['size']}")
    print(f"  Total Births: {world.total_births}")
    print(f"  Total Deaths: {world.total_deaths}")
    print(f"  Final Average Fitness: {final_stats['average_fitness']:.2f}")
    print(f"  Generation Spread: {final_stats['generation_distribution']}")
    
    print(f"\nEvolved Trait Averages:")
    for trait, value in final_stats['trait_averages'].items():
        initial_value = initial_stats['trait_averages'].get(trait, 0.5)
        change = value - initial_value
        direction = "↑" if change > 0.05 else "↓" if change < -0.05 else "→"
        print(f"    {trait}: {value:.3f} ({direction}{abs(change):.3f})")
    
    # Show elite individuals
    elite = world.genetic_evolution.population_dynamics.get_elite_individuals(3)
    print(f"\nTop 3 Elite Individuals:")
    for i, (individual_id, genome) in enumerate(elite):
        print(f"    #{i+1}: Fitness={genome.fitness_score:.2f}, "
              f"Age={genome.age}, Generation={genome.generation}")
    
    # Create evolution visualization
    if len(evolution_metrics['steps']) > 1:
        plot_genetic_evolution(evolution_metrics)
    
    return world, evolution_metrics


def demonstrate_emergent_behaviors():
    """Demonstrate emergent behaviors from swarm + genetic interactions."""
    print("\n🌟 Emergent Behavior Demonstration")
    print("=" * 60)
    
    # Create world with both systems optimized
    emergent_config = config.get_config('expert')
    emergent_config.update({
        'grid_size': 30,
        'max_steps': 200,
        'monsters': 40,
        'max_monster_population': 80,
        'clan_size': 5,
        'territorial_markers': True,
        'dynamic_hazards': True,
        'learning_enabled': True,
        'boss_adaptation_enabled': True
    })
    
    world = AdvancedWorld(emergent_config)
    world.setup_simulation()
    
    print(f"Complex Multi-Agent Environment:")
    print(f"  Grid Size: {world.grid.size}×{world.grid.size}")
    print(f"  Total Agents: {len(world.agents)}")
    print(f"  Evolutionary Monsters: {len(world.evolutionary_monsters)}")
    print(f"  Predators: {len([a for a in world.agents.values() if type(a).__name__ == 'Predator'])}")
    
    # Track emergent phenomena
    emergent_metrics = {
        'step': [],
        'system_complexity': [],
        'behavioral_entropy': [],
        'adaptation_rate': [],
        'territory_formation': [],
        'collective_intelligence': []
    }
    
    print(f"\nObserving emergent behaviors...")
    step = 0
    while world.step() and step < 50:  # Limited for demo
        step += 1
        
        if step % 10 == 0:
            # Measure emergent properties
            complexity = calculate_system_complexity(world)
            entropy = calculate_behavioral_entropy(world)
            adaptation = measure_adaptation_rate(world)
            territory = assess_territory_formation(world)
            intelligence = measure_collective_intelligence(world)
            
            emergent_metrics['step'].append(step)
            emergent_metrics['system_complexity'].append(complexity)
            emergent_metrics['behavioral_entropy'].append(entropy)
            emergent_metrics['adaptation_rate'].append(adaptation)
            emergent_metrics['territory_formation'].append(territory)
            emergent_metrics['collective_intelligence'].append(intelligence)
            
            print(f"  Step {step}: Complexity={complexity:.2f}, "
                  f"Entropy={entropy:.2f}, Territory={territory:.2f}")
    
    print(f"\nEmergent Behavior Analysis:")
    if emergent_metrics['step']:
        print(f"  Peak System Complexity: {max(emergent_metrics['system_complexity']):.3f}")
        print(f"  Average Behavioral Entropy: {np.mean(emergent_metrics['behavioral_entropy']):.3f}")
        print(f"  Territory Formation Score: {max(emergent_metrics['territory_formation']):.3f}")
        print(f"  Collective Intelligence Peak: {max(emergent_metrics['collective_intelligence']):.3f}")
    
    # Analyze final state
    analyze_final_ecosystem(world)
    
    return world, emergent_metrics


def calculate_system_complexity(world) -> float:
    """Calculate overall system complexity."""
    # Count distinct agent types and behaviors
    agent_types = set()
    behaviors = set()
    
    for agent in world.agents.values():
        if agent.alive:
            agent_types.add(type(agent).__name__)
            
            if hasattr(agent, 'current_swarm_behavior'):
                behaviors.add(agent.current_swarm_behavior.value)
    
    # Factor in genetic diversity
    genetic_diversity = world._calculate_genetic_diversity()
    
    # Pheromone complexity
    pheromone_complexity = len(world.swarm_intelligence.pheromone_map) / (world.grid.size ** 2)
    
    return len(agent_types) * len(behaviors) * (1 + genetic_diversity) * (1 + pheromone_complexity)


def calculate_behavioral_entropy(world) -> float:
    """Calculate entropy in behavioral patterns."""
    behavior_counts = {}
    
    for monster in world.evolutionary_monsters.values():
        if monster.alive:
            behavior = monster.current_swarm_behavior.value
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
    
    if not behavior_counts:
        return 0.0
    
    total = sum(behavior_counts.values())
    entropy = 0.0
    
    for count in behavior_counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)
    
    return entropy


def measure_adaptation_rate(world) -> float:
    """Measure how quickly the system adapts."""
    # Count recent genetic changes
    recent_generation = 0
    if world.genetic_evolution.population_dynamics.population:
        recent_generation = max(
            genome.generation for genome in world.genetic_evolution.population_dynamics.population.values()
        )
    
    # Factor in learning progress
    learning_progress = 0.0
    if world.dek and hasattr(world.dek, 'bandit_arms'):
        total_selections = sum(arm.times_selected for arm in world.dek.bandit_arms.values())
        learning_progress = min(1.0, total_selections / 100.0)
    
    return (recent_generation * 0.1) + learning_progress


def assess_territory_formation(world) -> float:
    """Assess how well territories are formed."""
    territory_score = 0.0
    
    # Count monsters with established territories
    territorial_monsters = 0
    for monster in world.evolutionary_monsters.values():
        if monster.alive and monster.territory_center:
            territorial_monsters += 1
            
            # Bonus for staying in territory
            if monster._distance_to_territory_center() <= monster.territory_radius:
                territory_score += 0.1
    
    if len(world.evolutionary_monsters) > 0:
        territory_score += territorial_monsters / len(world.evolutionary_monsters)
    
    return territory_score


def measure_collective_intelligence(world) -> float:
    """Measure collective intelligence emergence."""
    # Pheromone trail efficiency
    pheromone_efficiency = 0.0
    if world.swarm_intelligence.pheromone_map:
        total_strength = sum(
            sum(trail.strength for trail in trails)
            for trails in world.swarm_intelligence.pheromone_map.values()
        )
        pheromone_efficiency = min(1.0, total_strength / 1000.0)
    
    # Coordination efficiency
    coordination = world._calculate_swarm_coordination()
    
    # Memory utilization
    memory_size = len(world.swarm_intelligence.collective_memory.enemy_positions)
    memory_score = min(1.0, memory_size / 10.0)
    
    return (pheromone_efficiency + coordination + memory_score) / 3.0


def analyze_final_ecosystem(world) -> None:
    """Analyze the final state of the ecosystem."""
    print(f"\nFinal Ecosystem Analysis:")
    
    # Agent survival rates
    agent_counts = {}
    for agent in world.agents.values():
        agent_type = type(agent).__name__
        if agent_type not in agent_counts:
            agent_counts[agent_type] = {'alive': 0, 'total': 0}
        
        agent_counts[agent_type]['total'] += 1
        if agent.alive:
            agent_counts[agent_type]['alive'] += 1
    
    for agent_type, counts in agent_counts.items():
        survival_rate = (counts['alive'] / counts['total']) * 100 if counts['total'] > 0 else 0
        print(f"  {agent_type}: {counts['alive']}/{counts['total']} ({survival_rate:.1f}% survival)")
    
    # Genetic composition
    if world.evolutionary_monsters:
        trait_analysis = {}
        for monster in world.evolutionary_monsters.values():
            if monster.alive and monster.evolved_traits:
                for trait, value in monster.evolved_traits.items():
                    if trait not in trait_analysis:
                        trait_analysis[trait] = []
                    trait_analysis[trait].append(value)
        
        print(f"\nDominant Genetic Traits:")
        for trait, values in trait_analysis.items():
            if values:
                avg = np.mean(values)
                std = np.std(values)
                print(f"    {trait}: {avg:.3f} ±{std:.3f}")
    
    # Territorial distribution
    territories = {}
    for monster in world.evolutionary_monsters.values():
        if monster.alive and monster.territory_center:
            territory_key = f"{monster.territory_center[0]//5}-{monster.territory_center[1]//5}"
            territories[territory_key] = territories.get(territory_key, 0) + 1
    
    if territories:
        print(f"\nTerritorial Clusters: {len(territories)} distinct regions")
        for region, count in sorted(territories.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    Region {region}: {count} monsters")


def plot_swarm_evolution(metrics: Dict[str, List]) -> None:
    """Create visualization of swarm evolution."""
    if not metrics['steps']:
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Swarm Intelligence Evolution', fontsize=16)
    
    steps = metrics['steps']
    
    # Behavior diversity
    ax1.plot(steps, metrics['behavior_diversity'], 'b-o', markersize=4)
    ax1.set_title('Behavioral Diversity')
    ax1.set_ylabel('Number of Behaviors')
    ax1.grid(True, alpha=0.3)
    
    # Coordination scores
    ax2.plot(steps, metrics['coordination_scores'], 'r-o', markersize=4)
    ax2.set_title('Swarm Coordination')
    ax2.set_ylabel('Coordination Score')
    ax2.grid(True, alpha=0.3)
    
    # Pheromone trails
    ax3.plot(steps, metrics['pheromone_trails'], 'g-o', markersize=4)
    ax3.set_title('Pheromone Trail Count')
    ax3.set_ylabel('Active Trails')
    ax3.set_xlabel('Simulation Step')
    ax3.grid(True, alpha=0.3)
    
    # Collective memory
    ax4.plot(steps, metrics['collective_memory'], 'm-o', markersize=4)
    ax4.set_title('Collective Memory')
    ax4.set_ylabel('Tracked Entities')
    ax4.set_xlabel('Simulation Step')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('advanced_results', exist_ok=True)
    plt.savefig('advanced_results/swarm_evolution.png', dpi=150, bbox_inches='tight')
    print(f"\nSwarm evolution plot saved to: advanced_results/swarm_evolution.png")


def plot_genetic_evolution(metrics: Dict[str, Any]) -> None:
    """Create visualization of genetic evolution."""
    if not metrics['steps']:
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Genetic Evolution Dynamics', fontsize=16)
    
    steps = metrics['steps']
    
    # Population size
    ax1.plot(steps, metrics['population_size'], 'b-o', markersize=4)
    ax1.set_title('Population Size')
    ax1.set_ylabel('Individuals')
    ax1.grid(True, alpha=0.3)
    
    # Average fitness
    ax2.plot(steps, metrics['average_fitness'], 'r-o', markersize=4)
    ax2.set_title('Average Fitness')
    ax2.set_ylabel('Fitness Score')
    ax2.grid(True, alpha=0.3)
    
    # Genetic diversity
    ax3.plot(steps, metrics['genetic_diversity'], 'g-o', markersize=4)
    ax3.set_title('Genetic Diversity')
    ax3.set_ylabel('Diversity Score')
    ax3.set_xlabel('Simulation Step')
    ax3.grid(True, alpha=0.3)
    
    # Dominant traits evolution
    ax4.set_title('Trait Evolution')
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Trait Value')
    
    # Plot key traits
    key_traits = ['aggression_modifier', 'pack_affinity', 'intelligence_modifier', 'speed_modifier']
    colors = ['red', 'blue', 'green', 'orange']
    
    for trait, color in zip(key_traits, colors):
        if trait in metrics['dominant_traits'] and len(metrics['dominant_traits'][trait]) > 0:
            ax4.plot(steps, metrics['dominant_traits'][trait], 
                    color=color, marker='o', markersize=3, label=trait, linewidth=2)
    
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_results/genetic_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Genetic evolution plot saved to: advanced_results/genetic_evolution.png")


if __name__ == "__main__":
    print("🚀 Advanced Multi-Agent Simulation with Swarm Intelligence & Genetic Evolution")
    print("Predator: Badlands - Next Generation AI Systems")
    print("=" * 80)
    
    try:
        # Demonstrate swarm intelligence
        swarm_world, swarm_data = demonstrate_swarm_intelligence()
        
        # Demonstrate genetic evolution
        evolution_world, evolution_data = demonstrate_genetic_evolution()
        
        # Demonstrate emergent behaviors
        emergent_world, emergent_data = demonstrate_emergent_behaviors()
        
        print("\n" + "=" * 80)
        print("🎉 ADVANCED DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nKey Achievements:")
        print("  ✓ Swarm Intelligence: Flocking, pheromone trails, collective behavior")
        print("  ✓ Genetic Evolution: Population dynamics, trait inheritance, fitness selection")
        print("  ✓ Emergent Behaviors: Complex interactions, territorial formation, adaptation")
        print("  ✓ Multi-System Integration: Swarm + Genetic + Learning + Adaptation")
        
        print("\nAdvanced Features Demonstrated:")
        print("  • Reynolds flocking algorithms with enhanced coordination")
        print("  • Pheromone-based communication and pathfinding")
        print("  • Multi-armed bandit learning with genetic trait modifiers")
        print("  • Population genetics with breeding, mutation, and selection")
        print("  • Territorial behavior and resource competition")
        print("  • Collective intelligence emergence")
        print("  • Adaptive counter-strategies and pattern recognition")
        
        print("\nNext Steps for Research:")
        print("  • Neural network integration for enhanced decision-making")
        print("  • Multi-objective optimization for competing survival goals")
        print("  • Emergent communication protocols between agents")
        print("  • Hierarchical learning and meta-adaptation")
        print("  • Cross-species cooperation and competition dynamics")
        
        print(f"\nVisualization files saved in: advanced_results/")
        
    except Exception as e:
        print(f"\n❌ Advanced demonstration failed: {e}")
        import traceback
        traceback.print_exc()