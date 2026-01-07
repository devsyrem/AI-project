"""
Enhanced World simulation controller with Swarm Intelligence and Genetic Evolution.
Manages advanced multi-agent systems with emergent behaviors.
"""
from typing import Dict, List, Optional, Tuple, Any
import random
import logging
import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime

from grid import Grid
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents.agent import Agent, ActionType
from agents.predator import Predator, PredatorRole
from agents.monster import Monster, MonsterType
from agents.thia import Thia
from agents.boss import Boss
from agents.evolutionary_monster import EvolutionaryMonster
import config

# Import advanced systems
from swarm_intelligence import SwarmIntelligence
from genetic_evolution import GeneticEvolutionEngine

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSimulationMetrics:
    """Enhanced metrics including swarm and genetic data."""
    step: int
    dek_alive: bool
    dek_health: int
    dek_stamina: int
    dek_honor: float
    boss_alive: bool
    boss_health: int
    monsters_alive: int
    thia_functional: bool
    thia_being_carried: bool
    
    # Swarm intelligence metrics
    active_swarm_behaviors: Dict[str, int]
    pheromone_trail_count: int
    collective_memory_size: int
    
    # Genetic evolution metrics
    monster_population_size: int
    average_monster_fitness: float
    genetic_diversity_score: float
    current_generation: int


@dataclass
class EnhancedSimulationSummary:
    """Enhanced summary with evolutionary and swarm data."""
    run_id: int
    seed: int
    config_name: str
    steps: int
    dek_survived: bool
    boss_defeated: bool
    dek_final_honor: float
    dek_final_health: int
    trophies_collected: int
    dek_damage_dealt_to_boss: int
    thia_repaired: bool
    monsters_killed: int
    simulation_time: float
    
    # Enhanced metrics
    final_monster_population: int
    total_monster_births: int
    total_monster_deaths: int
    peak_swarm_coordination: float
    evolutionary_generations: int
    dominant_traits: Dict[str, float]


class AdvancedWorld:
    """
    Advanced World simulation with Swarm Intelligence and Genetic Evolution.
    Manages complex emergent behaviors and evolutionary dynamics.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize advanced world simulation."""
        self.config = config_dict
        self.seed = config_dict.get('seed', random.randint(1, 1000000))
        random.seed(self.seed)
        
        # Core simulation components
        self.grid = Grid(config_dict.get('grid_size', 15))
        self.agents: Dict[str, Agent] = {}
        self.step_count = 0
        self.max_steps = config_dict.get('max_steps', 500)
        self.running = False
        
        # Enhanced systems
        self.swarm_intelligence = SwarmIntelligence(self.grid.size)
        self.genetic_evolution = GeneticEvolutionEngine(
            initial_population=config_dict.get('monsters', 20),
            max_population=config_dict.get('max_monster_population', 50)
        )
        
        # Tracking and metrics
        self.metrics_history: List[EnhancedSimulationMetrics] = []
        self.time_series_data: List[Dict[str, Any]] = []
        
        # Agent references for quick access
        self.dek: Optional[Predator] = None
        self.thia: Optional[Thia] = None
        self.boss: Optional[Boss] = None
        self.evolutionary_monsters: Dict[str, EvolutionaryMonster] = {}
        
        # Population tracking
        self.total_births = 0
        self.total_deaths = 0
        self.generation_peaks: List[float] = []
        
        logger.info(f"Initialized advanced world with seed {self.seed}")
    
    def setup_simulation(self) -> None:
        """Set up the simulation environment with advanced features."""
        logger.info("Setting up advanced simulation...")
        
        # Generate terrain
        self._generate_terrain()
        
        # Create agents
        self._create_agents()
        
        # Initialize swarm territories
        self._initialize_swarm_territories()
        
        # Start simulation
        self.running = True
        self.step_count = 0
        
        logger.info(f"Advanced simulation ready with {len(self.agents)} agents")
    
    def _generate_terrain(self) -> None:
        """Generate procedural terrain with enhanced features."""
        trap_density = self.config.get('trap_density', 0.05)
        dynamic_hazards = self.config.get('dynamic_hazards', True)
        
        # Place initial traps
        trap_count = int(self.grid.size * self.grid.size * trap_density)
        
        for _ in range(trap_count):
            attempts = 0
            while attempts < 50:
                x = random.randint(0, self.grid.size - 1)
                y = random.randint(0, self.grid.size - 1)
                
                if self.grid.get((x, y)) is None:
                    # Vary trap types for complexity
                    trigger_once = random.random() < 0.7  # 70% single-use traps
                    damage = random.randint(10, 25)  # Random trap damage
                    self.grid.add_trap((x, y), damage, trigger_once)
                    break
                
                attempts += 1
        
        # Initialize territorial markers for swarm intelligence
        if self.config.get('territorial_markers', True):
            self._place_territorial_markers()
    
    def _place_territorial_markers(self) -> None:
        """Place territorial markers that influence swarm behavior."""
        marker_count = max(3, self.grid.size // 5)
        
        for _ in range(marker_count):
            x = random.randint(0, self.grid.size - 1)
            y = random.randint(0, self.grid.size - 1)
            
            # Deposit territorial pheromones
            self.swarm_intelligence.deposit_pheromone((x, y), 'territory', 200.0)
    
    def _create_agents(self) -> None:
        """Create agents with enhanced genetic and swarm capabilities."""
        # Create Dek (player character)
        dek_pos = self.grid.random_empty()
        self.dek = Predator(
            "dek", 
            role=PredatorRole.DEK,
            pos=dek_pos,
            health=self.config.get('dek_health', 100),
            stamina=self.config.get('dek_stamina', 80),

        )
        self.grid.place(self.dek, dek_pos)
        self.agents[self.dek.id] = self.dek
        
        # Create clan members
        clan_size = self.config.get('clan_size', 3)
        for i in range(clan_size):
            clan_pos = self.grid.random_empty()
            clan_member = Predator(
                f"clan_{i}",
                role=random.choice([PredatorRole.FATHER, PredatorRole.BROTHER, PredatorRole.CLAN]),
                pos=clan_pos,
                health=self.config.get('predator_health', 80),
                stamina=self.config.get('predator_stamina', 70)
            )
            self.grid.place(clan_member, clan_pos)
            self.agents[clan_member.id] = clan_member
        
        # Create Thia synthetic
        thia_pos = self.grid.random_empty()
        self.thia = Thia(
            "thia",
            pos=thia_pos,
            health=self.config.get('thia_health', 60),
vision_range=8
        )
        self.grid.place(self.thia, thia_pos)
        self.agents[self.thia.id] = self.thia
        
        # Create Boss
        boss_pos = self.grid.random_empty()
        self.boss = Boss(
            "boss",
            pos=boss_pos,
            health=self.config.get('boss_health', 200),
territory_size=3
        )
        self.grid.place(self.boss, boss_pos)
        self.agents[self.boss.id] = self.boss
        
        # Create evolutionary monsters
        self._create_evolutionary_monsters()
        
        # Set up agent relationships
        self._setup_agent_relationships()
    
    def _create_evolutionary_monsters(self) -> None:
        """Create monsters with genetic evolution capabilities."""
        monster_count = self.config.get('monsters', 20)
        
        for i in range(monster_count):
            # Get genetic individual
            genetic_id = self.genetic_evolution.population_dynamics.add_individual()
            if genetic_id == -1:  # Population at capacity
                break
            
            genome = self.genetic_evolution.get_individual_genome(genetic_id)
            evolved_traits = self.genetic_evolution.create_evolved_monster_traits(genetic_id)
            
            # Determine monster type based on traits
            monster_type = self._determine_monster_type(evolved_traits)
            
            # Create evolutionary monster
            monster_pos = self.grid.random_empty()
            monster = EvolutionaryMonster(
                f"monster_{i}",
                monster_type=monster_type,
                pos=monster_pos,
                health=self.config.get('monster_health', 30),
                genetic_id=genetic_id,
                evolved_traits=evolved_traits
            )
            
            self.grid.place(monster, monster_pos)
            self.agents[monster.id] = monster
            self.evolutionary_monsters[monster.id] = monster
    
    def _determine_monster_type(self, traits: Dict[str, float]) -> MonsterType:
        """Determine monster type based on genetic traits."""
        pack_affinity = traits.get('pack_affinity', 0.5)
        aggression = traits.get('aggression_modifier', 0.5)
        intelligence = traits.get('intelligence_modifier', 0.5)
        
        # Evolved type for highly advanced monsters
        if intelligence > 0.8 and (pack_affinity > 0.8 or aggression > 0.8):
            return MonsterType.EVOLVED
        
        # Pack type for highly social monsters
        elif pack_affinity > 0.7:
            return MonsterType.PACK
        
        # Large type for aggressive, territorial monsters  
        elif aggression > 0.7 and traits.get('territory_drive', 0.5) > 0.6:
            return MonsterType.LARGE
        
        # Medium type for balanced monsters
        elif 0.3 < aggression < 0.7 and 0.3 < pack_affinity < 0.7:
            return MonsterType.MEDIUM
        
        # Small type for others (high speed, low aggression)
        else:
            return MonsterType.SMALL
    
    def _initialize_swarm_territories(self) -> None:
        """Initialize territorial structures for swarm behavior."""
        # Assign territories to evolutionary monsters
        territory_centers = []
        territory_count = max(2, len(self.evolutionary_monsters) // 8)
        
        for _ in range(territory_count):
            center = (random.randint(0, self.grid.size - 1), 
                     random.randint(0, self.grid.size - 1))
            territory_centers.append(center)
            
            # Deposit strong territorial pheromones
            self.swarm_intelligence.deposit_pheromone(center, 'territory', 300.0)
        
        # Assign monsters to territories
        monsters_list = list(self.evolutionary_monsters.values())
        for i, monster in enumerate(monsters_list):
            if territory_centers:
                territory_idx = i % len(territory_centers)
                monster.territory_center = territory_centers[territory_idx]
    
    def step(self) -> bool:
        """Execute one simulation step with advanced systems."""
        if not self.running:
            return False
        
        # Update advanced systems
        self._update_swarm_intelligence()
        self._update_genetic_evolution()
        
        # Execute agent behaviors
        agent_actions = {}
        for agent_id, agent in list(self.agents.items()):
            if agent.alive:
                try:
                    agent.step(self)
                    agent_actions[agent_id] = agent.last_action if hasattr(agent, 'last_action') else ActionType.WAIT
                except Exception as e:
                    logger.error(f"Error in agent {agent_id} step: {e}")
                    agent_actions[agent_id] = ActionType.WAIT
            else:
                # Handle agent death
                self._handle_agent_death(agent_id, agent)
        
        # Update environment
        self._update_environment()
        
        # Collect enhanced metrics
        metrics = self._collect_enhanced_metrics()
        self.metrics_history.append(metrics)
        
        # Increment step counter
        self.step_count += 1
        
        # Check end conditions
        end_condition = self._check_end_conditions()
        if end_condition or self.step_count >= self.max_steps:
            self.running = False
            logger.info(f"Simulation ended at step {self.step_count}: {end_condition or 'max steps reached'}")
            return False
        
        return True
    
    def _update_swarm_intelligence(self) -> None:
        """Update swarm intelligence systems."""
        # Update pheromone trails
        self.swarm_intelligence.update_pheromones()
        
        # Update collective memory
        self.swarm_intelligence.collective_memory.last_updated = self.step_count
        
        # Analyze swarm coordination
        if len(self.evolutionary_monsters) > 1:
            coordination_score = self._calculate_swarm_coordination()
            if coordination_score > 0:
                self.generation_peaks.append(coordination_score)
    
    def _update_genetic_evolution(self) -> None:
        """Update genetic evolution systems."""
        # Prepare environment data
        environment_data = {
            'resource_level': self._calculate_resource_level(),
            'predator_count': len([a for a in self.agents.values() 
                                 if isinstance(a, Predator) and a.alive]),
            'territory_pressure': self._calculate_territory_pressure()
        }
        
        # Update fitness scores for living monsters
        for monster in self.evolutionary_monsters.values():
            if monster.alive and monster.genetic_id:
                fitness_metrics = monster.get_fitness_metrics()
                self.genetic_evolution.update_fitness(
                    monster.genetic_id,
                    fitness_metrics['survival_time'],
                    fitness_metrics['damage_dealt'],
                    fitness_metrics['offspring_count'],
                    fitness_metrics['territory_time']
                )
        
        # Update genetic evolution
        evolution_data = self.genetic_evolution.update(self.step_count, environment_data)
        
        # Handle new births
        if evolution_data['population_updates']['births'] > 0:
            self._spawn_new_monsters(evolution_data['population_updates']['births'])
            self.total_births += evolution_data['population_updates']['births']
        
        # Track deaths
        if evolution_data['population_updates']['deaths'] > 0:
            self.total_deaths += evolution_data['population_updates']['deaths']
    
    def _spawn_new_monsters(self, birth_count: int) -> None:
        """Spawn new monsters from genetic evolution."""
        for i in range(birth_count):
            # Find available genetic ID
            population = self.genetic_evolution.population_dynamics.population
            new_genetic_ids = [id for id in population.keys() 
                             if not any(m.genetic_id == id for m in self.evolutionary_monsters.values())]
            
            if new_genetic_ids:
                genetic_id = new_genetic_ids[0]
                evolved_traits = self.genetic_evolution.create_evolved_monster_traits(genetic_id)
                monster_type = self._determine_monster_type(evolved_traits)
                
                # Find spawn position
                spawn_pos = self.grid.random_empty()
                if spawn_pos:
                    new_monster_id = f"evolved_monster_{self.step_count}_{i}"
                    new_monster = EvolutionaryMonster(
                        new_monster_id,
                        monster_type=monster_type,
                        pos=spawn_pos,
                        genetic_id=genetic_id,
                        evolved_traits=evolved_traits
                    )
                    
                    self.grid.place(new_monster, spawn_pos)
                    self.agents[new_monster_id] = new_monster
                    self.evolutionary_monsters[new_monster_id] = new_monster
    
    def _handle_agent_death(self, agent_id: str, agent: Agent) -> None:
        """Handle agent death with genetic implications."""
        if agent_id in self.evolutionary_monsters:
            monster = self.evolutionary_monsters[agent_id]
            
            # Remove from genetic population if appropriate
            if monster.genetic_id:
                self.genetic_evolution.population_dynamics.remove_individual(monster.genetic_id)
            
            # Remove from tracking
            del self.evolutionary_monsters[agent_id]
        
        # Remove from grid and agents
        if agent.pos:
            self.grid.remove_agent(agent.pos)
        
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def _calculate_swarm_coordination(self) -> float:
        """Calculate overall swarm coordination score."""
        if len(self.evolutionary_monsters) < 2:
            return 0.0
        
        coordination_score = 0.0
        total_pairs = 0
        
        monsters_list = list(self.evolutionary_monsters.values())
        
        for i, monster1 in enumerate(monsters_list):
            for monster2 in monsters_list[i+1:]:
                if monster1.alive and monster2.alive and monster1.pos and monster2.pos:
                    distance = self.grid.distance(monster1.pos, monster2.pos)
                    
                    # Closer monsters with similar behaviors score higher
                    if distance <= 5:  # Within coordination range
                        behavior_similarity = 1.0 if (monster1.current_swarm_behavior == 
                                                     monster2.current_swarm_behavior) else 0.3
                        proximity_score = (5 - distance) / 5
                        coordination_score += behavior_similarity * proximity_score
                    
                    total_pairs += 1
        
        return coordination_score / max(1, total_pairs)
    
    def _calculate_resource_level(self) -> float:
        """Calculate available resources in environment."""
        # Base resource level
        resource_level = 0.5
        
        # Fewer predators = more resources
        predator_count = len([a for a in self.agents.values() 
                             if isinstance(a, Predator) and a.alive])
        resource_level += max(0, (5 - predator_count) * 0.1)
        
        # Boss presence reduces resources
        if self.boss and self.boss.alive:
            resource_level -= 0.2
        
        return max(0.0, min(1.0, resource_level))
    
    def _calculate_territory_pressure(self) -> float:
        """Calculate territorial competition pressure."""
        active_monsters = len([m for m in self.evolutionary_monsters.values() if m.alive])
        
        # Normalize by grid capacity
        grid_capacity = (self.grid.size * self.grid.size) // 10
        pressure = active_monsters / max(1, grid_capacity)
        
        return min(1.0, pressure)
    
    def _collect_enhanced_metrics(self) -> EnhancedSimulationMetrics:
        """Collect comprehensive metrics including advanced systems."""
        # Basic metrics
        dek_alive = self.dek.alive if self.dek else False
        dek_health = self.dek.health if self.dek else 0
        dek_stamina = self.dek.stamina if self.dek else 0
        dek_honor = self.dek.honor if self.dek else 0.0
        
        boss_alive = self.boss.alive if self.boss else False
        boss_health = self.boss.health if self.boss else 0
        
        monsters_alive = len([m for m in self.evolutionary_monsters.values() if m.alive])
        
        thia_functional = self.thia.functional if self.thia else False
        thia_being_carried = (self.thia and hasattr(self.thia, 'being_carried') and 
                             self.thia.being_carried) if self.thia else False
        
        # Swarm intelligence metrics
        swarm_behaviors = {}
        for monster in self.evolutionary_monsters.values():
            if monster.alive:
                behavior = monster.current_swarm_behavior.value
                swarm_behaviors[behavior] = swarm_behaviors.get(behavior, 0) + 1
        
        swarm_state = self.swarm_intelligence.get_swarm_state()
        
        # Genetic evolution metrics
        pop_stats = self.genetic_evolution.population_dynamics.get_population_statistics()
        
        return EnhancedSimulationMetrics(
            step=self.step_count,
            dek_alive=dek_alive,
            dek_health=dek_health,
            dek_stamina=dek_stamina,
            dek_honor=dek_honor,
            boss_alive=boss_alive,
            boss_health=boss_health,
            monsters_alive=monsters_alive,
            thia_functional=thia_functional,
            thia_being_carried=thia_being_carried,
            active_swarm_behaviors=swarm_behaviors,
            pheromone_trail_count=swarm_state.get('pheromone_trails', 0),
            collective_memory_size=swarm_state.get('collective_memory', {}).get('tracked_enemies', 0),
            monster_population_size=pop_stats['size'],
            average_monster_fitness=pop_stats['average_fitness'],
            genetic_diversity_score=self._calculate_genetic_diversity(),
            current_generation=max([g for g in pop_stats['generation_distribution'].keys()], default=0)
        )
    
    def _calculate_genetic_diversity(self) -> float:
        """Calculate genetic diversity in monster population."""
        if not self.evolutionary_monsters:
            return 0.0
        
        # Calculate trait variance across population
        trait_values = {}
        
        for monster in self.evolutionary_monsters.values():
            if monster.alive and monster.evolved_traits:
                for trait_name, value in monster.evolved_traits.items():
                    if trait_name not in trait_values:
                        trait_values[trait_name] = []
                    trait_values[trait_name].append(value)
        
        if not trait_values:
            return 0.0
        
        # Average variance across all traits
        total_variance = 0.0
        trait_count = 0
        
        for trait_name, values in trait_values.items():
            if len(values) > 1:
                import numpy as np
                variance = np.var(values)
                total_variance += variance
                trait_count += 1
        
        return total_variance / max(1, trait_count)
    
    def run_simulation(self) -> EnhancedSimulationSummary:
        """Run complete simulation and return enhanced summary."""
        start_time = datetime.now()
        
        # Setup simulation
        self.setup_simulation()
        
        # Run simulation loop
        while self.step():
            pass
        
        # Calculate simulation time
        end_time = datetime.now()
        simulation_time = (end_time - start_time).total_seconds()
        
        # Generate enhanced summary
        return self._generate_enhanced_summary(simulation_time)
    
    def _generate_enhanced_summary(self, simulation_time: float) -> EnhancedSimulationSummary:
        """Generate comprehensive simulation summary."""
        # Basic results
        dek_survived = self.dek.alive if self.dek else False
        boss_defeated = not self.boss.alive if self.boss else True
        
        dek_final_honor = self.dek.honor if self.dek else 0.0
        dek_final_health = self.dek.health if self.dek else 0
        trophies_collected = len(self.dek.trophies) if self.dek else 0
        
        dek_damage_to_boss = 0
        if self.dek and hasattr(self.dek, 'damage_dealt'):
            dek_damage_to_boss = sum(
                damage for target_id, damage in self.dek.damage_dealt.items()
                if target_id == (self.boss.id if self.boss else "")
            )
        
        thia_repaired = self.thia.functional if self.thia else False
        monsters_killed = self._count_dead_monsters()
        
        # Enhanced metrics
        final_population = len([m for m in self.evolutionary_monsters.values() if m.alive])
        peak_coordination = max(self.generation_peaks) if self.generation_peaks else 0.0
        
        # Dominant traits analysis
        dominant_traits = self._analyze_dominant_traits()
        
        # Generation count
        pop_stats = self.genetic_evolution.population_dynamics.get_population_statistics()
        max_generation = max([g for g in pop_stats['generation_distribution'].keys()], default=0)
        
        return EnhancedSimulationSummary(
            run_id=0,  # Set by batch runner
            seed=self.seed,
            config_name="advanced",
            steps=self.step_count,
            dek_survived=dek_survived,
            boss_defeated=boss_defeated,
            dek_final_honor=dek_final_honor,
            dek_final_health=dek_final_health,
            trophies_collected=trophies_collected,
            dek_damage_dealt_to_boss=dek_damage_to_boss,
            thia_repaired=thia_repaired,
            monsters_killed=monsters_killed,
            simulation_time=simulation_time,
            final_monster_population=final_population,
            total_monster_births=self.total_births,
            total_monster_deaths=self.total_deaths,
            peak_swarm_coordination=peak_coordination,
            evolutionary_generations=max_generation,
            dominant_traits=dominant_traits
        )
    
    def _analyze_dominant_traits(self) -> Dict[str, float]:
        """Analyze dominant genetic traits in current population."""
        trait_averages = {}
        
        if not self.evolutionary_monsters:
            return trait_averages
        
        trait_sums = {}
        trait_counts = {}
        
        for monster in self.evolutionary_monsters.values():
            if monster.alive and monster.evolved_traits:
                for trait_name, value in monster.evolved_traits.items():
                    if trait_name not in trait_sums:
                        trait_sums[trait_name] = 0.0
                        trait_counts[trait_name] = 0
                    
                    trait_sums[trait_name] += value
                    trait_counts[trait_name] += 1
        
        for trait_name in trait_sums:
            if trait_counts[trait_name] > 0:
                trait_averages[trait_name] = trait_sums[trait_name] / trait_counts[trait_name]
        
        return trait_averages
    
    def get_advanced_state(self) -> Dict[str, Any]:
        """Get comprehensive world state including advanced systems."""
        base_state = {
            'step': self.step_count,
            'running': self.running,
            'agents': {agent_id: agent.get_agent_state() 
                      for agent_id, agent in self.agents.items()},
            'grid_state': self.grid.get_state()
        }
        
        # Add swarm intelligence state
        base_state['swarm_intelligence'] = self.swarm_intelligence.get_swarm_state()
        
        # Add genetic evolution state
        base_state['genetic_evolution'] = self.genetic_evolution.population_dynamics.get_population_statistics()
        
        return base_state
    
    def _setup_agent_relationships(self) -> None:
        """Set up initial relationships and coordination between agents."""
        # Find Dek for special setup
        dek = None
        predators = [a for a in self.agents.values() if isinstance(a, Predator)]
        
        for predator in predators:
            if predator.role == PredatorRole.DEK:
                dek = predator
                break
                
        # Set up Thia coordination with Dek if both exist
        if dek and self.thia and self.thia.pos and dek.pos:
            distance = self.grid.distance(self.thia.pos, dek.pos)
            if distance <= 5:  # Close enough for initial coordination
                if hasattr(self.thia, 'coordinate_with_predator'):
                    self.thia.coordinate_with_predator(dek)
                
        # Set up clan relationships
        for i, pred1 in enumerate(predators):
            for pred2 in predators[i+1:]:
                # Initialize neutral relationships
                if hasattr(pred1, 'clan_relationships'):
                    pred1.clan_relationships[pred2.id] = 0.0
                if hasattr(pred2, 'clan_relationships'):
                    pred2.clan_relationships[pred1.id] = 0.0
    
    def _check_end_conditions(self) -> Optional[str]:
        """Check if simulation should end and return reason."""
        # Dek death ends simulation
        if not self.dek or not self.dek.alive:
            return "dek_defeated"
        
        # Boss death is victory condition
        if self.boss and not self.boss.alive:
            return "boss_defeated"
        
        # All monsters dead
        active_monsters = len([m for m in self.evolutionary_monsters.values() if m.alive])
        if active_monsters == 0:
            return "monsters_eliminated"
        
        return None
    
    def _update_environment(self) -> None:
        """Update environmental conditions and dynamic hazards."""
        # Spawn dynamic hazards periodically
        hazard_interval = self.config.get('hazard_spawn_interval', 50)
        if self.config.get('dynamic_hazards', False) and self.step_count % hazard_interval == 0:
            # Spawn new trap
            for _ in range(3):  # Try a few times
                pos = self.grid.random_empty()
                if pos:
                    trigger_once = random.random() < 0.6  # 60% single-use dynamic traps
                    damage = random.randint(15, 30)  # Dynamic traps are stronger
                    if hasattr(self.grid, 'add_trap'):
                        self.grid.add_trap(pos, damage, trigger_once)
                    break
    
    def _count_dead_monsters(self) -> int:
        """Count how many monsters have died during simulation."""
        return self.total_deaths


# Maintain backward compatibility
World = AdvancedWorld


# Maintain backward compatibility
World = AdvancedWorld