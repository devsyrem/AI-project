"""
Grid module for Predator: Badlands simulation.
Implements toroidal 2D environment with terrain, obstacles, and entity management.
"""
from typing import Optional, List, Tuple, Set, Dict, Any
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class Trap:
    """Represents a trap or hazard on the grid."""
    damage: int
    trigger_once: bool = True
    triggered: bool = False
    
    def trigger(self) -> int:
        """
        Trigger the trap and return damage dealt.
        
        Returns:
            Damage amount, 0 if already triggered and trigger_once=True
        """
        if self.trigger_once and self.triggered:
            return 0
        self.triggered = True
        return self.damage


class Grid:
    """
    Toroidal 2D grid environment for the simulation.
    Supports entity placement, movement with wrapping, and terrain features.
    """
    
    def __init__(self, size: int = 20, wrap: bool = True, seed: Optional[int] = None):
        """
        Initialize the grid.
        
        Args:
            size: Grid dimension (creates size x size grid)
            wrap: Whether to use toroidal (wrap-around) topology
            seed: Random seed for reproducible terrain generation
        """
        self.size = size
        self.wrap = wrap
        self.rng = random.Random(seed)
        
        # Grid storage: dict mapping (x, y) -> agent object
        self.entities: Dict[Tuple[int, int], Any] = {}
        
        # Traps and hazards: dict mapping (x, y) -> Trap object  
        self.traps: Dict[Tuple[int, int], Trap] = {}
        
        # Track empty positions for efficient random placement
        self._empty_positions: Optional[Set[Tuple[int, int]]] = None
        
    def _normalize_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Normalize position coordinates with wrapping if enabled.
        
        Args:
            pos: (x, y) coordinate tuple
            
        Returns:
            Normalized coordinate tuple
        """
        x, y = pos
        if self.wrap:
            x = x % self.size
            y = y % self.size
        else:
            # Clamp to grid bounds
            x = max(0, min(x, self.size - 1))
            y = max(0, min(y, self.size - 1))
        return (x, y)
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds or wrappable)."""
        if self.wrap:
            return True  # Any position is valid with wrapping
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size
    
    def place(self, agent: Any, pos: Tuple[int, int]) -> bool:
        """
        Place an agent at the specified position.
        
        Args:
            agent: Agent object to place
            pos: (x, y) coordinates
            
        Returns:
            True if placement successful, False if position occupied
        """
        normalized_pos = self._normalize_position(pos)
        
        if normalized_pos in self.entities:
            return False  # Position occupied
            
        self.entities[normalized_pos] = agent
        
        # Update agent position if it has pos attribute
        if hasattr(agent, 'pos'):
            agent.pos = normalized_pos
            
        # Invalidate empty positions cache
        self._empty_positions = None
        
        return True
    
    def remove(self, pos: Tuple[int, int]) -> Optional[Any]:
        """
        Remove agent from specified position.
        
        Args:
            pos: (x, y) coordinates
            
        Returns:
            Removed agent or None if position was empty
        """
        normalized_pos = self._normalize_position(pos)
        agent = self.entities.pop(normalized_pos, None)
        
        if agent:
            # Invalidate empty positions cache
            self._empty_positions = None
            
        return agent
    
    def get(self, pos: Tuple[int, int]) -> Optional[Any]:
        """
        Get agent at specified position.
        
        Args:
            pos: (x, y) coordinates
            
        Returns:
            Agent at position or None if empty
        """
        normalized_pos = self._normalize_position(pos)
        return self.entities.get(normalized_pos)
    
    def move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Move agent from one position to another.
        
        Args:
            from_pos: Current position
            to_pos: Destination position
            
        Returns:
            True if move successful, False otherwise
        """
        agent = self.remove(from_pos)
        if agent is None:
            return False
            
        if not self.place(agent, to_pos):
            # Failed to place at destination, put back
            self.place(agent, from_pos)
            return False
            
        return True
    
    def get_empty_positions(self) -> Set[Tuple[int, int]]:
        """
        Get set of all empty positions on the grid.
        
        Returns:
            Set of (x, y) coordinates that are empty
        """
        if self._empty_positions is None:
            all_positions = {(x, y) for x in range(self.size) for y in range(self.size)}
            self._empty_positions = all_positions - set(self.entities.keys())
            
        return self._empty_positions.copy()
    
    def random_empty(self) -> Optional[Tuple[int, int]]:
        """
        Get a random empty position.
        
        Returns:
            Random empty position or None if grid is full
        """
        empty = self.get_empty_positions()
        if not empty:
            return None
        return self.rng.choice(list(empty))
    
    def neighbors(self, pos: Tuple[int, int], radius: int = 1) -> List[Tuple[int, int]]:
        """
        Get neighboring positions within specified radius.
        
        Args:
            pos: Center position
            radius: Maximum distance from center
            
        Returns:
            List of neighboring positions
        """
        x, y = self._normalize_position(pos)
        neighbors = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip center position
                    
                neighbor_pos = self._normalize_position((x + dx, y + dy))
                neighbors.append(neighbor_pos)
                
        return neighbors
    
    def adjacent_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 4-connected adjacent positions (up, down, left, right).
        
        Args:
            pos: Center position
            
        Returns:
            List of adjacent positions
        """
        x, y = self._normalize_position(pos)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        return [self._normalize_position((x + dx, y + dy)) for dx, dy in directions]
    
    def add_trap(self, pos: Tuple[int, int], damage: int, trigger_once: bool = True) -> None:
        """
        Add a trap at the specified position.
        
        Args:
            pos: Position to place trap
            damage: Damage dealt by trap
            trigger_once: Whether trap can only trigger once
        """
        normalized_pos = self._normalize_position(pos)
        self.traps[normalized_pos] = Trap(damage=damage, trigger_once=trigger_once)
    
    def trigger_trap(self, pos: Tuple[int, int]) -> int:
        """
        Trigger trap at position if present.
        
        Args:
            pos: Position to check for traps
            
        Returns:
            Damage dealt by trap, 0 if no trap or already triggered
        """
        normalized_pos = self._normalize_position(pos)
        trap = self.traps.get(normalized_pos)
        
        if trap:
            damage = trap.trigger()
            # Remove single-use traps after triggering
            if trap.trigger_once and trap.triggered:
                del self.traps[normalized_pos]
            return damage
            
        return 0
    
    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate distance between two positions (considering wrapping).
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        x1, y1 = self._normalize_position(pos1)
        x2, y2 = self._normalize_position(pos2)
        
        if self.wrap:
            # Consider wrapping for shortest distance
            dx = min(abs(x2 - x1), self.size - abs(x2 - x1))
            dy = min(abs(y2 - y1), self.size - abs(y2 - y1))
        else:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
        return np.sqrt(dx * dx + dy * dy)
    
    def line_of_sight(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Check if there's a clear line of sight between two positions.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            True if clear line of sight exists
        """
        # Simple implementation - could be enhanced with raycasting
        # For now, just check if path is clear (no agents blocking)
        
        x1, y1 = self._normalize_position(from_pos)
        x2, y2 = self._normalize_position(to_pos)
        
        # Use Bresenham's line algorithm (simplified)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx == 0 and dy == 0:
            return True
            
        # For simplicity, just check direct adjacency for now
        return self.distance(from_pos, to_pos) <= 1.5
    
    def serialize_state(self) -> Dict[str, Any]:
        """
        Serialize current grid state for logging/analysis.
        
        Returns:
            Dictionary containing grid state information
        """
        entity_count = {}
        for pos, agent in self.entities.items():
            agent_type = type(agent).__name__
            entity_count[agent_type] = entity_count.get(agent_type, 0) + 1
            
        return {
            "size": self.size,
            "wrap": self.wrap,
            "entity_count": entity_count,
            "trap_count": len(self.traps),
            "empty_positions": len(self.get_empty_positions())
        }
    
    def get_entities_in_radius(self, center: Tuple[int, int], radius: int) -> List[Tuple[Tuple[int, int], Any]]:
        """
        Get all entities within specified radius of center position.
        
        Args:
            center: Center position to search from
            radius: Search radius
            
        Returns:
            List of (position, agent) tuples within radius
        """
        entities_in_radius = []
        
        for pos, agent in self.entities.items():
            if self.distance(center, pos) <= radius:
                entities_in_radius.append((pos, agent))
                
        return entities_in_radius
    
    def __str__(self) -> str:
        """String representation of the grid for debugging."""
        grid_str = ""
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                pos = (x, y)
                if pos in self.entities:
                    agent = self.entities[pos]
                    # Use first letter of class name
                    row += type(agent).__name__[0]
                elif pos in self.traps:
                    row += "T"  # Trap
                else:
                    row += "."  # Empty
            grid_str += row + "\n"
        return grid_str