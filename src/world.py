"""
World simulation controller for Predator: Badlands.
Manages the simulation environment, agent interactions, and data logging.
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
import config

logger = logging.getLogger(__name__)


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
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


@dataclass
class SimulationSummary:
    """Summary statistics for a completed simulation run."""
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


class World:
    """
    Main simulation world controller.
    Manages grid, agents, procedural generation, and data collection.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize the simulation world.
        
        Args:
            config_dict: Configuration parameters
        """
        self.config = config_dict
        self.step_count = 0
        self.max_steps = config_dict.get('max_steps', config.DEFAULT_MAX_STEPS)
        
        # Set up random seed
        self.seed = config_dict.get('seed', config.DEFAULT_SEED)
        random.seed(self.seed)
        
        # Initialize grid
        grid_size = config_dict.get('grid_size', config.DEFAULT_GRID_SIZE)
        self.grid = Grid(size=grid_size, wrap=config_dict.get('wrap_grid', True), seed=self.seed)
        
        # Agent storage
        self.agents: Dict[str, Agent] = {}
        self.predators: List[Predator] = []
        self.monsters: List[Monster] = []
        self.thia: Optional[Thia] = None
        self.boss: Optional[Boss] = None
        
        # Data collection
        self.metrics_history: List[SimulationMetrics] = []
        self.time_series_data: List[Dict[str, Any]] = []
        
        # Procedural generation
        self.hazard_spawn_timer = 0
        self.hazard_spawn_interval = config_dict.get('hazard_spawn_interval', 
                                                   config.HAZARD_SPAWN_INTERVAL)
        self.dynamic_hazards = config_dict.get('dynamic_hazards', True)
        
        # Simulation state
        self.running = False
        self.simulation_result: Optional[str] = None
        
        logger.info(f"World initialized with seed {self.seed}, grid {grid_size}x{grid_size}")
        
    def setup_simulation(self) -> None:
        """Set up the simulation with initial agents and environment."""
        self._generate_terrain()
        self._spawn_agents()
        self._setup_agent_relationships()
        
        # Initialize all agents with grid reference
        for agent in self.agents.values():
            agent.set_grid(self.grid)
            
        self.running = True
        logger.info(f"Simulation setup complete. Agents: {len(self.agents)}")
        
    def _generate_terrain(self) -> None:
        """Generate initial terrain features and hazards."""
        trap_count = int(self.grid.size * self.grid.size * 
                        self.config.get('trap_density', config.TRAP_DENSITY))
        
        for _ in range(trap_count):
            pos = self.grid.random_empty()
            if pos:
                damage = random.randint(config.TRAP_DAMAGE // 2, config.TRAP_DAMAGE * 2)
                trigger_once = random.choice([True, False])
                self.grid.add_trap(pos, damage, trigger_once)
                
        logger.info(f"Generated {trap_count} terrain hazards")
        
    def _spawn_agents(self) -> None:
        """Spawn all agents in the simulation."""
        # Spawn Thia first
        thia_pos = self.grid.random_empty()
        if thia_pos:
            self.thia = Thia("Thia", pos=thia_pos)
            self.grid.place(self.thia, thia_pos)
            self.agents["Thia"] = self.thia
            
        # Spawn predators
        predator_configs = [
            ("Dek", PredatorRole.DEK),
            ("Father", PredatorRole.FATHER),
            ("Brother", PredatorRole.BROTHER)
        ]
        
        for pred_id, role in predator_configs:
            pos = self.grid.random_empty()
            if pos:
                predator = Predator(pred_id, role, pos=pos)
                self.grid.place(predator, pos)
                self.agents[pred_id] = predator
                self.predators.append(predator)
                
        # Spawn monsters
        monster_count = self.config.get('monsters', config.DEFAULT_MONSTERS)
        for i in range(monster_count):
            pos = self.grid.random_empty()
            if pos:
                # Vary monster types
                monster_type = random.choice(list(MonsterType))
                monster = Monster(f"Monster_{i}", monster_type, pos=pos)
                self.grid.place(monster, pos)
                self.agents[monster.id] = monster
                self.monsters.append(monster)
                
        # Create monster packs
        self._create_monster_packs()
        
        # Spawn boss
        boss_pos = self._find_boss_territory()
        if boss_pos:
            boss_health = self.config.get('boss_health', config.BOSS_HEALTH)
            self.boss = Boss("Boss", pos=boss_pos, health=boss_health)
            self.grid.place(self.boss, boss_pos)
            self.agents["Boss"] = self.boss
            
        logger.info(f"Spawned {len(self.predators)} predators, {len(self.monsters)} monsters, "
                   f"1 Thia, 1 Boss")
                   
    def _create_monster_packs(self) -> None:
        """Organize some monsters into coordinated packs."""
        pack_monsters = [m for m in self.monsters if m.monster_type == MonsterType.PACK]
        
        # Group pack monsters into packs of 3-5
        pack_size = 3
        for i in range(0, len(pack_monsters), pack_size):
            pack_group = pack_monsters[i:i + pack_size]
            if len(pack_group) >= 2:  # Need at least 2 for a pack
                # Set up pack relationships
                for monster in pack_group:
                    other_members = [m for m in pack_group if m != monster]
                    monster.add_to_pack(other_members)
                    
        logger.info(f"Created {len(pack_monsters) // pack_size} monster packs")
        
    def _find_boss_territory(self) -> Optional[Tuple[int, int]]:
        """Find suitable location for boss territory (away from spawn areas)."""
        # Try to place boss away from other agents
        attempts = 20
        for _ in range(attempts):
            pos = self.grid.random_empty()
            if pos:
                # Check distance from other agents
                min_distance = float('inf')
                for agent in self.agents.values():
                    if agent.pos:
                        distance = self.grid.distance(pos, agent.pos)
                        min_distance = min(min_distance, distance)
                        
                # Good spot if reasonably far from others
                if min_distance > 5:
                    return pos
                    
        # Fallback to any empty position
        return self.grid.random_empty()
        
    def _setup_agent_relationships(self) -> None:
        """Set up initial relationships and coordination between agents."""
        # Find Dek for special setup
        dek = None
        for predator in self.predators:
            if predator.role == PredatorRole.DEK:
                dek = predator
                break
                
        # Set up Thia coordination with Dek if both exist
        if dek and self.thia and self.thia.pos and dek.pos:
            distance = self.grid.distance(self.thia.pos, dek.pos)
            if distance <= 5:  # Close enough for initial coordination
                self.thia.coordinate_with_predator(dek)
                
        # Set up clan relationships
        for i, pred1 in enumerate(self.predators):
            for pred2 in self.predators[i+1:]:
                # Initialize neutral relationships
                pred1.clan_relationships[pred2.id] = 0.0
                pred2.clan_relationships[pred1.id] = 0.0
                
    def step(self) -> bool:
        """
        Execute one simulation step.
        
        Returns:
            True if simulation should continue, False if ended
        """
        if not self.running:
            return False
            
        self.step_count += 1
        
        # Procedural hazard generation
        if self.dynamic_hazards:
            self._update_hazards()
            
        # Update all agents
        agents_to_update = list(self.agents.values())
        random.shuffle(agents_to_update)  # Randomize update order
        
        for agent in agents_to_update:
            if agent.is_alive():
                agent.step(self)
                
        # Collect metrics
        self._collect_metrics()
        
        # Check end conditions
        end_result = self._check_end_conditions()
        if end_result:
            self.simulation_result = end_result
            self.running = False
            logger.info(f"Simulation ended at step {self.step_count}: {end_result}")
            return False
            
        # Check max steps
        if self.step_count >= self.max_steps:
            self.simulation_result = "max_steps_reached"
            self.running = False
            logger.info(f"Simulation ended: maximum steps ({self.max_steps}) reached")
            return False
            
        return True
        
    def _update_hazards(self) -> None:
        """Update dynamic hazards and spawn new ones."""
        self.hazard_spawn_timer += 1
        
        if self.hazard_spawn_timer >= self.hazard_spawn_interval:
            self.hazard_spawn_timer = 0
            
            # Spawn new hazards based on difficulty curve
            difficulty_multiplier = 1.0 + (self.step_count / 100) * 0.1
            spawn_count = max(1, int(2 * difficulty_multiplier))
            
            for _ in range(spawn_count):
                pos = self.grid.random_empty()
                if pos:
                    # Escalating hazard damage
                    base_damage = config.TRAP_DAMAGE
                    damage = int(base_damage * difficulty_multiplier)
                    self.grid.add_trap(pos, damage, trigger_once=True)
                    
            logger.debug(f"Spawned {spawn_count} dynamic hazards (difficulty: {difficulty_multiplier:.1f})")
            
    def _collect_metrics(self) -> None:
        """Collect current simulation metrics."""
        # Find key agents
        dek = None
        for predator in self.predators:
            if predator.role == PredatorRole.DEK:
                dek = predator
                break
                
        monsters_alive = sum(1 for monster in self.monsters if monster.is_alive())
        
        metrics = SimulationMetrics(
            step=self.step_count,
            dek_alive=dek.is_alive() if dek else False,
            dek_health=dek.health if dek and dek.is_alive() else 0,
            dek_stamina=dek.stamina if dek and dek.is_alive() else 0,
            dek_honor=dek.honor if dek and dek.is_alive() else 0.0,
            boss_alive=self.boss.is_alive() if self.boss else False,
            boss_health=self.boss.health if self.boss and self.boss.is_alive() else 0,
            monsters_alive=monsters_alive,
            thia_functional=self.thia.functional if self.thia else False,
            thia_being_carried=self.thia.being_carried if self.thia else False
        )
        
        self.metrics_history.append(metrics)
        
        # Also store as time series data for CSV export
        time_series_entry = asdict(metrics)
        self.time_series_data.append(time_series_entry)
        
    def _check_end_conditions(self) -> Optional[str]:
        """
        Check if simulation should end.
        
        Returns:
            End condition string or None to continue
        """
        # Find Dek
        dek = None
        for predator in self.predators:
            if predator.role == PredatorRole.DEK:
                dek = predator
                break
                
        # Dek death ends simulation
        if not dek or not dek.is_alive():
            return "dek_defeated"
            
        # Boss defeat ends simulation
        if self.boss and not self.boss.is_alive():
            return "boss_defeated"
            
        # All monsters dead (optional victory condition)
        monsters_alive = sum(1 for monster in self.monsters if monster.is_alive())
        if monsters_alive == 0 and self.boss and not self.boss.is_alive():
            return "total_victory"
            
        return None
        
    def run_simulation(self) -> SimulationSummary:
        """
        Run complete simulation and return summary.
        
        Returns:
            Simulation summary with final statistics
        """
        start_time = datetime.now()
        
        self.setup_simulation()
        
        while self.step():
            # Optional: Add progress logging for long simulations
            if self.step_count % 100 == 0:
                logger.debug(f"Simulation step {self.step_count}/{self.max_steps}")
                
        end_time = datetime.now()
        simulation_time = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = self._generate_summary(simulation_time)
        
        logger.info(f"Simulation complete: {summary.steps} steps, "
                   f"Dek survived: {summary.dek_survived}, "
                   f"Boss defeated: {summary.boss_defeated}")
        
        return summary
        
    def _generate_summary(self, simulation_time: float) -> SimulationSummary:
        """Generate simulation summary statistics."""
        # Find key agents
        dek = None
        for predator in self.predators:
            if predator.role == PredatorRole.DEK:
                dek = predator
                break
                
        return SimulationSummary(
            run_id=0,  # Will be set by batch runner
            seed=self.seed,
            config_name=self.config.get('name', 'default'),
            steps=self.step_count,
            dek_survived=dek.is_alive() if dek else False,
            boss_defeated=not self.boss.is_alive() if self.boss else True,
            dek_final_honor=dek.honor if dek else 0.0,
            dek_final_health=dek.health if dek else 0,
            trophies_collected=len(dek.trophies) if dek else 0,
            dek_damage_dealt_to_boss=dek.boss_damage_dealt if dek else 0,
            thia_repaired=self.thia.functional if self.thia else False,
            monsters_killed=sum(1 for m in self.monsters if not m.is_alive()),
            simulation_time=simulation_time
        )
        
    def export_time_series(self, filepath: str) -> None:
        """
        Export time series data to CSV.
        
        Args:
            filepath: Path to CSV file
        """
        if not self.time_series_data:
            logger.warning("No time series data to export")
            return
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = self.time_series_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.time_series_data)
            
        logger.info(f"Exported time series data to {filepath}")
        
    def get_world_state(self) -> Dict[str, Any]:
        """
        Get current world state for analysis.
        
        Returns:
            Dictionary containing world state information
        """
        agent_states = {}
        for agent_id, agent in self.agents.items():
            agent_states[agent_id] = agent.get_state_dict()
            
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "running": self.running,
            "result": self.simulation_result,
            "grid_state": self.grid.serialize_state(),
            "agents": agent_states,
            "config": self.config
        }
        
    def visualize_grid(self) -> str:
        """
        Create text visualization of current grid state.
        
        Returns:
            String representation of grid
        """
        if not self.grid:
            return "No grid available"
            
        visualization = f"Step {self.step_count} - Grid {self.grid.size}x{self.grid.size}\n"
        visualization += "Legend: P=Predator, T=Thia, M=Monster, B=Boss, X=Trap, .=Empty\n\n"
        
        for y in range(self.grid.size):
            row = ""
            for x in range(self.grid.size):
                pos = (x, y)
                
                if pos in self.grid.entities:
                    agent = self.grid.entities[pos]
                    if isinstance(agent, Predator):
                        row += "P"
                    elif isinstance(agent, Thia):
                        row += "T"
                    elif isinstance(agent, Monster):
                        row += "M"
                    elif isinstance(agent, Boss):
                        row += "B"
                    else:
                        row += "?"
                elif pos in self.grid.traps:
                    row += "X"
                else:
                    row += "."
                    
            visualization += row + "\n"
            
        return visualization


class BatchSimulationRunner:
    """
    Runs multiple simulations for statistical analysis.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize batch runner.
        
        Args:
            base_config: Base configuration for simulations
        """
        self.base_config = base_config
        self.results: List[SimulationSummary] = []
        
    def run_batch(self, num_runs: int, seed_start: int = 0) -> List[SimulationSummary]:
        """
        Run batch of simulations.
        
        Args:
            num_runs: Number of simulations to run
            seed_start: Starting seed value
            
        Returns:
            List of simulation summaries
        """
        logger.info(f"Starting batch run: {num_runs} simulations")
        
        for run_id in range(num_runs):
            config_copy = self.base_config.copy()
            config_copy['seed'] = seed_start + run_id
            
            logger.info(f"Running simulation {run_id + 1}/{num_runs} (seed: {config_copy['seed']})")
            
            world = World(config_copy)
            summary = world.run_simulation()
            summary.run_id = run_id
            
            self.results.append(summary)
            
            # Export individual run time series
            if 'results_dir' in self.base_config:
                results_dir = self.base_config['results_dir']
                time_series_path = os.path.join(results_dir, f"run_{run_id:03d}_timeseries.csv")
                world.export_time_series(time_series_path)
                
        logger.info(f"Batch run complete: {len(self.results)} simulations")
        return self.results
        
    def export_summary_csv(self, filepath: str) -> None:
        """
        Export batch results summary to CSV.
        
        Args:
            filepath: Path to CSV file
        """
        if not self.results:
            logger.warning("No results to export")
            return
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = asdict(self.results[0]).keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow(asdict(result))
                
        logger.info(f"Exported summary CSV to {filepath}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all runs.
        
        Returns:
            Dictionary of statistics
        """
        if not self.results:
            return {}
            
        total_runs = len(self.results)
        dek_survivals = sum(1 for r in self.results if r.dek_survived)
        boss_defeats = sum(1 for r in self.results if r.boss_defeated)
        
        steps = [r.steps for r in self.results]
        honors = [r.dek_final_honor for r in self.results if r.dek_survived]
        
        return {
            "total_runs": total_runs,
            "dek_survival_rate": dek_survivals / total_runs,
            "boss_defeat_rate": boss_defeats / total_runs,
            "mean_steps": sum(steps) / len(steps),
            "median_steps": sorted(steps)[len(steps) // 2],
            "mean_honor": sum(honors) / len(honors) if honors else 0.0,
            "total_trophies": sum(r.trophies_collected for r in self.results),
            "total_boss_damage": sum(r.dek_damage_dealt_to_boss for r in self.results),
            "thia_repair_rate": sum(1 for r in self.results if r.thia_repaired) / total_runs
        }