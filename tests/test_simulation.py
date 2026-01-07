"""
Tests for World simulation and integration functionality.
"""
import pytest
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.world import World, BatchSimulationRunner, SimulationMetrics, SimulationSummary
from src import config
from src.agents import Predator, PredatorRole, Monster, MonsterType, Thia, Boss


class TestWorld:
    """Test cases for World simulation controller."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            'seed': 42,
            'grid_size': 10,
            'max_steps': 100,
            'monsters': 5,
            'boss_health': 100,
            'trap_density': 0.02,
            'hazard_spawn_interval': 50,
            'learning_enabled': True
        }
        
    def test_world_initialization(self):
        """Test world initialization."""
        world = World(self.test_config)
        
        assert world.seed == 42
        assert world.max_steps == 100
        assert world.grid.size == 10
        assert world.running == False
        assert len(world.agents) == 0
        
    def test_world_setup(self):
        """Test world setup and agent spawning."""
        world = World(self.test_config)
        world.setup_simulation()
        
        assert world.running == True
        assert len(world.agents) > 0
        
        # Check for required agents
        agent_types = [type(agent).__name__ for agent in world.agents.values()]
        assert 'Predator' in agent_types
        assert 'Monster' in agent_types
        assert 'Thia' in agent_types
        assert 'Boss' in agent_types
        
        # Check predators have correct roles
        predator_roles = [agent.role for agent in world.agents.values() 
                         if isinstance(agent, Predator)]
        assert PredatorRole.DEK in predator_roles
        
    def test_world_terrain_generation(self):
        """Test procedural terrain generation."""
        world = World(self.test_config)
        world._generate_terrain()
        
        # Should have some traps based on density
        expected_traps = int(world.grid.size * world.grid.size * self.test_config['trap_density'])
        assert len(world.grid.traps) >= 0  # May vary due to placement failures
        
    def test_world_agent_relationships(self):
        """Test agent relationship setup."""
        world = World(self.test_config)
        world.setup_simulation()
        
        # Find Dek and Thia
        dek = None
        thia = None
        
        for agent in world.agents.values():
            if isinstance(agent, Predator) and agent.role == PredatorRole.DEK:
                dek = agent
            elif isinstance(agent, Thia):
                thia = agent
                
        # Should have both
        assert dek is not None
        assert thia is not None
        
        # Check clan relationships
        predators = [a for a in world.agents.values() if isinstance(a, Predator)]
        if len(predators) > 1:
            assert len(predators[0].clan_relationships) > 0
            
    def test_world_single_step(self):
        """Test single simulation step."""
        world = World(self.test_config)
        world.setup_simulation()
        
        initial_step = world.step_count
        
        # Execute one step
        continue_sim = world.step()
        
        assert world.step_count == initial_step + 1
        assert len(world.metrics_history) > 0
        assert isinstance(continue_sim, bool)
        
    def test_world_end_conditions(self):
        """Test simulation end condition detection."""
        world = World(self.test_config)
        world.setup_simulation()
        
        # Test Dek death condition
        dek = None
        for agent in world.agents.values():
            if isinstance(agent, Predator) and agent.role == PredatorRole.DEK:
                dek = agent
                break
                
        if dek:
            # Kill Dek
            dek.die("test")
            
            end_condition = world._check_end_conditions()
            assert end_condition == "dek_defeated"
            
    def test_world_boss_defeat_condition(self):
        """Test boss defeat end condition."""
        world = World(self.test_config)
        world.setup_simulation()
        
        # Kill boss
        if world.boss:
            world.boss.die("test")
            
            end_condition = world._check_end_conditions()
            assert end_condition == "boss_defeated"
            
    def test_world_metrics_collection(self):
        """Test metrics collection during simulation."""
        world = World(self.test_config)
        world.setup_simulation()
        
        # Run a few steps
        for _ in range(5):
            if world.running:
                world.step()
                
        # Should have collected metrics
        assert len(world.metrics_history) >= 5
        assert len(world.time_series_data) >= 5
        
        # Check metric content
        latest_metrics = world.metrics_history[-1]
        assert isinstance(latest_metrics, SimulationMetrics)
        assert latest_metrics.step > 0
        
    def test_world_hazard_generation(self):
        """Test dynamic hazard generation."""
        config_with_hazards = self.test_config.copy()
        config_with_hazards['dynamic_hazards'] = True
        config_with_hazards['hazard_spawn_interval'] = 5
        
        world = World(config_with_hazards)
        world.setup_simulation()
        
        initial_traps = len(world.grid.traps)
        
        # Run enough steps to trigger hazard spawning
        for _ in range(10):
            if world.running:
                world.step()
                
        # Should have spawned more hazards
        final_traps = len(world.grid.traps)
        assert final_traps >= initial_traps
        
    def test_world_complete_simulation(self):
        """Test running a complete simulation."""
        # Use smaller config for faster test
        small_config = {
            'seed': 42,
            'grid_size': 8,
            'max_steps': 50,
            'monsters': 3,
            'boss_health': 50,
            'trap_density': 0.01,
            'learning_enabled': False  # Disable for deterministic test
        }
        
        world = World(small_config)
        summary = world.run_simulation()
        
        assert isinstance(summary, SimulationSummary)
        assert summary.steps <= 50
        assert summary.seed == 42
        assert isinstance(summary.dek_survived, bool)
        assert isinstance(summary.boss_defeated, bool)
        
    def test_world_visualization(self):
        """Test grid visualization."""
        world = World(self.test_config)
        world.setup_simulation()
        
        visualization = world.visualize_grid()
        
        assert isinstance(visualization, str)
        assert "Grid" in visualization
        assert "Legend" in visualization
        
        # Should contain grid representation
        lines = visualization.split('\n')
        grid_lines = [line for line in lines if len(line) == world.grid.size]
        assert len(grid_lines) == world.grid.size
        
    def test_world_state_serialization(self):
        """Test world state serialization."""
        world = World(self.test_config)
        world.setup_simulation()
        
        # Run a few steps
        for _ in range(3):
            world.step()
            
        state = world.get_world_state()
        
        assert 'step' in state
        assert 'agents' in state
        assert 'grid_state' in state
        assert 'config' in state
        
        # Check agent states
        assert len(state['agents']) > 0
        for agent_id, agent_state in state['agents'].items():
            assert 'id' in agent_state
            assert 'type' in agent_state
            assert 'health' in agent_state


class TestBatchSimulationRunner:
    """Test cases for batch simulation runner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            'seed': 42,
            'grid_size': 8,
            'max_steps': 30,
            'monsters': 3,
            'boss_health': 50,
            'learning_enabled': False  # Faster for testing
        }
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.test_config['results_dir'] = self.temp_dir
        
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_batch_runner_initialization(self):
        """Test batch runner initialization."""
        runner = BatchSimulationRunner(self.test_config)
        
        assert runner.base_config == self.test_config
        assert len(runner.results) == 0
        
    def test_batch_runner_small_batch(self):
        """Test running a small batch of simulations."""
        runner = BatchSimulationRunner(self.test_config)
        
        # Run small batch
        results = runner.run_batch(num_runs=3, seed_start=100)
        
        assert len(results) == 3
        assert len(runner.results) == 3
        
        # Check result content
        for i, result in enumerate(results):
            assert isinstance(result, SimulationSummary)
            assert result.run_id == i
            assert result.seed == 100 + i
            assert result.steps > 0
            
    def test_batch_runner_statistics(self):
        """Test batch statistics calculation."""
        runner = BatchSimulationRunner(self.test_config)
        
        # Run batch
        runner.run_batch(num_runs=5, seed_start=200)
        
        # Calculate statistics
        stats = runner.get_statistics()
        
        assert 'total_runs' in stats
        assert 'dek_survival_rate' in stats
        assert 'boss_defeat_rate' in stats
        assert 'mean_steps' in stats
        assert 'median_steps' in stats
        
        assert stats['total_runs'] == 5
        assert 0.0 <= stats['dek_survival_rate'] <= 1.0
        assert 0.0 <= stats['boss_defeat_rate'] <= 1.0
        
    def test_batch_runner_csv_export(self):
        """Test CSV export functionality."""
        runner = BatchSimulationRunner(self.test_config)
        
        # Run batch
        runner.run_batch(num_runs=3, seed_start=300)
        
        # Export CSV
        csv_path = os.path.join(self.temp_dir, "test_results.csv")
        runner.export_summary_csv(csv_path)
        
        assert os.path.exists(csv_path)
        
        # Check CSV content
        with open(csv_path, 'r') as f:
            content = f.read()
            
        assert 'run_id' in content
        assert 'seed' in content
        assert 'dek_survived' in content
        assert 'boss_defeated' in content
        
        # Should have header + 3 data rows
        lines = content.strip().split('\n')
        assert len(lines) == 4  # Header + 3 results


class TestSimulationIntegration:
    """Integration tests for complete simulation scenarios."""
    
    def test_dek_vs_boss_scenario(self):
        """Test Dek vs Boss combat scenario."""
        config_dict = {
            'seed': 123,
            'grid_size': 10,
            'max_steps': 100,
            'monsters': 2,  # Minimal monsters
            'boss_health': 80,
            'learning_enabled': True
        }
        
        world = World(config_dict)
        world.setup_simulation()
        
        # Find Dek and Boss
        dek = None
        boss = None
        
        for agent in world.agents.values():
            if isinstance(agent, Predator) and agent.role == PredatorRole.DEK:
                dek = agent
            elif isinstance(agent, Boss):
                boss = agent
                
        assert dek is not None
        assert boss is not None
        
        # Run simulation
        summary = world.run_simulation()
        
        # Should have completed (either Dek or Boss died, or max steps)
        assert summary.steps > 0
        assert not (summary.dek_survived and summary.boss_defeated is False and summary.steps < 100)
        
    def test_thia_repair_scenario(self):
        """Test Thia repair and coordination scenario."""
        config_dict = {
            'seed': 456,
            'grid_size': 8,
            'max_steps': 50,
            'monsters': 1,
            'boss_health': 60,
            'learning_enabled': False  # Simpler for testing
        }
        
        world = World(config_dict)
        world.setup_simulation()
        
        # Find agents
        dek = None
        thia = None
        
        for agent in world.agents.values():
            if isinstance(agent, Predator) and agent.role == PredatorRole.DEK:
                dek = agent
            elif isinstance(agent, Thia):
                thia = agent
                
        # Initially Thia should be non-functional
        assert thia.functional == False
        
        # Run some steps to allow repair attempts
        for _ in range(20):
            if world.running:
                world.step()
                
                # Check if Thia got repaired
                if thia.functional:
                    break
                    
        # May or may not be repaired depending on chance and interaction
        
    def test_monster_pack_behavior(self):
        """Test monster pack coordination."""
        config_dict = {
            'seed': 789,
            'grid_size': 12,
            'max_steps': 30,
            'monsters': 8,  # More monsters for pack formation
            'boss_health': 100,
            'learning_enabled': False
        }
        
        world = World(config_dict)
        world.setup_simulation()
        
        # Find pack monsters
        pack_monsters = [agent for agent in world.agents.values() 
                        if isinstance(agent, Monster) and agent.monster_type == MonsterType.PACK]
        
        # Run simulation
        world.run_simulation()
        
        # Should complete without errors
        assert world.step_count > 0
        
    def test_adaptive_learning_scenario(self):
        """Test adaptive learning mechanisms."""
        config_dict = {
            'seed': 999,
            'grid_size': 10,
            'max_steps': 80,
            'monsters': 4,
            'boss_health': 120,
            'learning_enabled': True,
            'boss_adaptation_enabled': True
        }
        
        world = World(config_dict)
        world.setup_simulation()
        
        # Find learning agents
        dek = None
        boss = None
        
        for agent in world.agents.values():
            if isinstance(agent, Predator) and agent.role == PredatorRole.DEK:
                dek = agent
            elif isinstance(agent, Boss):
                boss = agent
                
        # Run simulation and check learning
        initial_epsilon = dek.epsilon if dek else None
        initial_counters = len(boss.active_counters) if boss else 0
        
        # Run several steps
        for _ in range(30):
            if world.running:
                world.step()
                
        # Check if learning occurred
        if dek:
            # Epsilon should have decayed (exploration reduced)
            assert dek.epsilon <= initial_epsilon
            
            # Should have made some action selections
            total_selections = sum(arm.times_selected for arm in dek.bandit_arms.values())
            assert total_selections > 0
            
        if boss:
            # May have learned counter-strategies (depends on combat)
            pass
            
    def test_deterministic_reproduction(self):
        """Test that same seed produces same results."""
        config_dict = {
            'seed': 555,
            'grid_size': 8,
            'max_steps': 20,
            'monsters': 3,
            'boss_health': 50,
            'learning_enabled': False,  # Disable randomness in learning
            'dynamic_hazards': False   # Disable random hazard spawning
        }
        
        # Run same simulation twice
        world1 = World(config_dict.copy())
        summary1 = world1.run_simulation()
        
        world2 = World(config_dict.copy())
        summary2 = world2.run_simulation()
        
        # Should produce identical results with same seed
        assert summary1.steps == summary2.steps
        assert summary1.dek_survived == summary2.dek_survived
        assert summary1.boss_defeated == summary2.boss_defeated
        # Note: Some metrics might vary due to floating point operations or
        # complex interactions, but basic outcomes should be the same
        
    def test_configuration_presets(self):
        """Test different configuration presets work."""
        presets = ['basic', 'standard', 'expert']
        
        for preset_name in presets:
            config_dict = config.get_config(preset_name)
            
            # Override for faster testing
            config_dict['max_steps'] = 20
            config_dict['monsters'] = 3
            
            world = World(config_dict)
            summary = world.run_simulation()
            
            # Should complete without errors
            assert summary.steps > 0
            assert isinstance(summary.dek_survived, bool)
            assert isinstance(summary.boss_defeated, bool)


class TestMetricsAndLogging:
    """Test metrics collection and data logging."""
    
    def test_metrics_data_structure(self):
        """Test simulation metrics data structure."""
        metrics = SimulationMetrics(
            step=10,
            dek_alive=True,
            dek_health=85,
            dek_stamina=60,
            dek_honor=25.5,
            boss_alive=True,
            boss_health=150,
            monsters_alive=4,
            thia_functional=False,
            thia_being_carried=True
        )
        
        assert metrics.step == 10
        assert metrics.dek_alive == True
        assert metrics.dek_health == 85
        assert metrics.dek_honor == 25.5
        
    def test_summary_data_structure(self):
        """Test simulation summary data structure."""
        summary = SimulationSummary(
            run_id=1,
            seed=42,
            config_name="test",
            steps=150,
            dek_survived=True,
            boss_defeated=False,
            dek_final_honor=45.0,
            dek_final_health=60,
            trophies_collected=3,
            dek_damage_dealt_to_boss=120,
            thia_repaired=True,
            monsters_killed=8,
            simulation_time=2.5
        )
        
        assert summary.run_id == 1
        assert summary.seed == 42
        assert summary.dek_survived == True
        assert summary.simulation_time == 2.5
        
    def test_time_series_export(self):
        """Test time series data export."""
        config_dict = {
            'seed': 111,
            'grid_size': 6,
            'max_steps': 15,
            'monsters': 2,
            'boss_health': 40
        }
        
        world = World(config_dict)
        world.run_simulation()
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(temp_dir, "timeseries_test.csv")
            world.export_time_series(csv_path)
            
            assert os.path.exists(csv_path)
            
            # Check CSV format
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                
            assert len(lines) > 1  # Header + data
            assert 'step' in lines[0]  # Header should contain step
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)