"""
Tests for Grid class functionality.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.grid import Grid, Trap


class TestGrid:
    """Test cases for Grid class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = Grid(size=5, wrap=True, seed=42)
        
    def test_grid_initialization(self):
        """Test grid is initialized correctly."""
        assert self.grid.size == 5
        assert self.grid.wrap == True
        assert len(self.grid.entities) == 0
        assert len(self.grid.traps) == 0
        
    def test_position_normalization_with_wrapping(self):
        """Test position normalization with wrapping enabled."""
        # Test normal positions
        assert self.grid._normalize_position((2, 3)) == (2, 3)
        
        # Test wrapping
        assert self.grid._normalize_position((5, 3)) == (0, 3)
        assert self.grid._normalize_position((2, 5)) == (2, 0)
        assert self.grid._normalize_position((-1, 3)) == (4, 3)
        assert self.grid._normalize_position((2, -1)) == (2, 4)
        
    def test_position_normalization_without_wrapping(self):
        """Test position normalization without wrapping."""
        no_wrap_grid = Grid(size=5, wrap=False)
        
        # Test clamping to bounds
        assert no_wrap_grid._normalize_position((6, 3)) == (4, 3)
        assert no_wrap_grid._normalize_position((2, 6)) == (2, 4)
        assert no_wrap_grid._normalize_position((-1, 3)) == (0, 3)
        assert no_wrap_grid._normalize_position((2, -1)) == (2, 0)
        
    def test_entity_placement_and_removal(self):
        """Test placing and removing entities."""
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.pos = None
                
        agent = MockAgent("test_agent")
        
        # Test placement
        success = self.grid.place(agent, (2, 3))
        assert success == True
        assert agent.pos == (2, 3)
        assert self.grid.get((2, 3)) == agent
        
        # Test placement at occupied position
        agent2 = MockAgent("test_agent2")
        success2 = self.grid.place(agent2, (2, 3))
        assert success2 == False
        
        # Test removal
        removed = self.grid.remove((2, 3))
        assert removed == agent
        assert self.grid.get((2, 3)) == None
        
    def test_entity_movement(self):
        """Test moving entities between positions."""
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.pos = None
                
        agent = MockAgent("test_agent")
        
        # Place agent
        self.grid.place(agent, (1, 1))
        
        # Move agent
        success = self.grid.move((1, 1), (2, 2))
        assert success == True
        assert self.grid.get((1, 1)) == None
        assert self.grid.get((2, 2)) == agent
        assert agent.pos == (2, 2)
        
    def test_empty_positions(self):
        """Test getting empty positions."""
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.pos = None
                
        # Initially all positions should be empty
        empty = self.grid.get_empty_positions()
        assert len(empty) == 25  # 5x5 grid
        
        # Place some agents
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        
        self.grid.place(agent1, (0, 0))
        self.grid.place(agent2, (1, 1))
        
        empty = self.grid.get_empty_positions()
        assert len(empty) == 23
        assert (0, 0) not in empty
        assert (1, 1) not in empty
        
    def test_random_empty_position(self):
        """Test getting random empty positions."""
        # Should return a valid position
        pos = self.grid.random_empty()
        assert pos is not None
        assert 0 <= pos[0] < 5
        assert 0 <= pos[1] < 5
        
    def test_neighbors(self):
        """Test neighbor calculation."""
        # Test neighbors with radius 1
        neighbors = self.grid.neighbors((2, 2), radius=1)
        assert len(neighbors) == 8  # 8 surrounding positions
        
        # Check all neighbors are within expected range
        for nx, ny in neighbors:
            assert abs(nx - 2) <= 1
            assert abs(ny - 2) <= 1
            assert (nx, ny) != (2, 2)  # Should not include center
            
    def test_adjacent_positions(self):
        """Test 4-connected adjacent positions."""
        adjacent = self.grid.adjacent_positions((2, 2))
        expected = [(2, 3), (2, 1), (3, 2), (1, 2)]
        
        assert len(adjacent) == 4
        for pos in expected:
            assert pos in adjacent
            
    def test_traps(self):
        """Test trap functionality."""
        # Add trap
        self.grid.add_trap((2, 2), damage=20, trigger_once=True)
        assert (2, 2) in self.grid.traps
        
        # Trigger trap
        damage = self.grid.trigger_trap((2, 2))
        assert damage == 20
        
        # Trap should be removed after triggering (trigger_once=True)
        damage2 = self.grid.trigger_trap((2, 2))
        assert damage2 == 0
        assert (2, 2) not in self.grid.traps
        
    def test_reusable_traps(self):
        """Test traps that can trigger multiple times."""
        # Add reusable trap
        self.grid.add_trap((3, 3), damage=15, trigger_once=False)
        
        # Should trigger multiple times
        damage1 = self.grid.trigger_trap((3, 3))
        damage2 = self.grid.trigger_trap((3, 3))
        
        assert damage1 == 15
        assert damage2 == 15
        assert (3, 3) in self.grid.traps  # Should still exist
        
    def test_distance_calculation(self):
        """Test distance calculation with and without wrapping."""
        # Test normal distance
        distance = self.grid.distance((0, 0), (3, 4))
        assert distance == 5.0  # 3-4-5 triangle
        
        # Test with wrapping - should find shorter path
        distance_wrap = self.grid.distance((0, 0), (4, 0))
        assert distance_wrap == 1.0  # Should wrap around
        
    def test_distance_without_wrapping(self):
        """Test distance calculation without wrapping."""
        no_wrap_grid = Grid(size=5, wrap=False)
        
        distance = no_wrap_grid.distance((0, 0), (4, 0))
        assert distance == 4.0  # Direct distance without wrapping
        
    def test_entities_in_radius(self):
        """Test getting entities within radius."""
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.pos = None
                
        # Place some agents
        agents = [MockAgent(f"agent{i}") for i in range(3)]
        positions = [(2, 2), (2, 3), (4, 4)]
        
        for agent, pos in zip(agents, positions):
            self.grid.place(agent, pos)
            
        # Get entities in radius
        entities = self.grid.get_entities_in_radius((2, 2), radius=2)
        
        # Should include agents at (2,2) and (2,3), but not (4,4)
        assert len(entities) >= 2
        
        found_agents = [entity for pos, entity in entities]
        assert agents[0] in found_agents  # (2,2)
        assert agents[1] in found_agents  # (2,3)
        
    def test_line_of_sight(self):
        """Test line of sight calculation."""
        # Adjacent positions should have line of sight
        assert self.grid.line_of_sight((2, 2), (2, 3)) == True
        assert self.grid.line_of_sight((2, 2), (3, 2)) == True
        
        # Distant positions (current simple implementation)
        # This is a placeholder test - real LOS would be more complex
        result = self.grid.line_of_sight((0, 0), (4, 4))
        assert isinstance(result, bool)
        
    def test_grid_serialization(self):
        """Test grid state serialization."""
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.pos = None
                
        # Add some entities and traps
        agent = MockAgent("test")
        self.grid.place(agent, (1, 1))
        self.grid.add_trap((2, 2), damage=10)
        
        state = self.grid.serialize_state()
        
        assert state['size'] == 5
        assert state['wrap'] == True
        assert 'MockAgent' in state['entity_count']
        assert state['entity_count']['MockAgent'] == 1
        assert state['trap_count'] == 1
        assert state['empty_positions'] == 23  # 25 - 1 entity - 1 trap
        
    def test_grid_string_representation(self):
        """Test grid string representation."""
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.pos = None
                
        # Add entity and trap
        agent = MockAgent("test")
        self.grid.place(agent, (0, 0))
        self.grid.add_trap((1, 1), damage=10)
        
        grid_str = str(self.grid)
        
        # Should be a multi-line string
        lines = grid_str.strip().split('\n')
        assert len(lines) == 5  # 5x5 grid
        
        # First line should show the agent (M for MockAgent)
        assert lines[0][0] == 'M'
        
        # Second line should show the trap
        assert lines[1][1] == 'T'


class TestTrap:
    """Test cases for Trap class."""
    
    def test_trap_single_use(self):
        """Test single-use trap behavior."""
        trap = Trap(damage=25, trigger_once=True)
        
        # First trigger should work
        damage = trap.trigger()
        assert damage == 25
        assert trap.triggered == True
        
        # Second trigger should do nothing
        damage2 = trap.trigger()
        assert damage2 == 0
        
    def test_trap_reusable(self):
        """Test reusable trap behavior."""
        trap = Trap(damage=15, trigger_once=False)
        
        # Multiple triggers should work
        damage1 = trap.trigger()
        damage2 = trap.trigger()
        damage3 = trap.trigger()
        
        assert damage1 == 15
        assert damage2 == 15
        assert damage3 == 15
        assert trap.triggered == True  # Marked as triggered but still works