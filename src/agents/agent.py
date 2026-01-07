"""
Base Agent class for Predator: Badlands simulation.
Defines common functionality for all entities in the simulation.
"""
from typing import Optional, Tuple, TYPE_CHECKING, List, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from grid import Grid

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Enumeration of possible agent actions."""
    MOVE = "move"
    REST = "rest" 
    ATTACK = "attack"
    SEARCH = "search"
    CARRY = "carry"
    REPAIR = "repair"
    FLEE = "flee"
    HUNT = "hunt"
    SEEK_BOSS = "seek_boss"
    SEEK_THIA = "seek_thia"


@dataclass
class AgentStats:
    """Statistics tracking for an agent."""
    steps_taken: int = 0
    damage_dealt: int = 0
    damage_received: int = 0
    stamina_spent: int = 0
    kills: int = 0
    deaths: int = 0


class Agent(ABC):
    """
    Base class for all agents in the simulation.
    Handles movement, health, stamina, and basic interactions.
    """
    
    def __init__(
        self,
        agent_id: str,
        pos: Optional[Tuple[int, int]] = None,
        health: int = 100,
        max_health: int = 100,
        stamina: int = 100,
        max_stamina: int = 100,
        move_cost: int = 1,
        vision_range: int = 3
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for the agent
            pos: Initial position (x, y)
            health: Current health points
            max_health: Maximum health points
            stamina: Current stamina points
            max_stamina: Maximum stamina points
            move_cost: Stamina cost per movement
            vision_range: How far the agent can see
        """
        self.id = agent_id
        self.pos = pos
        self.health = health
        self.max_health = max_health
        self.stamina = stamina
        self.max_stamina = max_stamina
        self.move_cost = move_cost
        self.vision_range = vision_range
        
        self.alive = True
        self.grid: Optional['Grid'] = None
        
        # Statistics and history tracking
        self.stats = AgentStats()
        self.action_history: List[ActionType] = []
        
        # State tracking
        self.last_action: Optional[ActionType] = None
        self.target_pos: Optional[Tuple[int, int]] = None
        
    def set_grid(self, grid: 'Grid') -> None:
        """Set reference to the grid this agent exists in."""
        self.grid = grid
        
    def is_alive(self) -> bool:
        """Check if agent is still alive."""
        return self.alive and self.health > 0
        
    def take_damage(self, amount: int, source: str = "unknown") -> int:
        """
        Apply incoming damage, update stats, and handle death.
        
        Args:
            amount: Damage amount to apply
            source: Identifier of the damage source for logging
            
        Returns:
            Actual damage applied (capped by current health)
        """
        if not self.is_alive():
            return 0
        
        actual_damage = min(amount, self.health)
        self.health -= actual_damage
        self.stats.damage_received += actual_damage
        
        if self.health <= 0:
            self.die(f"killed by {source}")
        
        return actual_damage
    def heal(self, amount: int) -> int:
        """
        Heal damage up to max health.
        
        Args:
            amount: Healing amount
            
        Returns:
            Actual healing done
        """
        if not self.is_alive():
            return 0
            
        old_health = self.health
        self.health = min(self.max_health, self.health + amount)
        actual_healing = self.health - old_health
        
        logger.debug(f"{self.id} healed {actual_healing} (health: {self.health})")
        return actual_healing
        
    def spend_stamina(self, amount: int) -> bool:
        """
        Spend stamina if available.
        
        Args:
            amount: Stamina to spend
            
        Returns:
            True if stamina was available and spent, False otherwise
        """
        if self.stamina >= amount:
            self.stamina -= amount
            self.stats.stamina_spent += amount
            return True
        return False
        
    def restore_stamina(self, amount: int) -> int:
        """
        Restore stamina up to maximum.
        
        Args:
            amount: Stamina to restore
            
        Returns:
            Actual stamina restored
        """
        old_stamina = self.stamina
        self.stamina = min(self.max_stamina, self.stamina + amount)
        actual_restoration = self.stamina - old_stamina
        
        logger.debug(f"{self.id} restored {actual_restoration} stamina (stamina: {self.stamina})")
        return actual_restoration
        
    def can_move(self) -> bool:
        """Check if agent has enough stamina to move."""
        return self.stamina >= self.move_cost and self.is_alive()
        
    def move_to(self, new_pos: Tuple[int, int]) -> bool:
        """
        Move to new position if possible.
        
        Args:
            new_pos: Target position (x, y)
            
        Returns:
            True if move was successful
        """
        if not self.can_move() or not self.grid:
            return False
            
        # Check if destination is valid and empty
        if not self.grid.is_valid_position(new_pos):
            return False
            
        if self.grid.get(new_pos) is not None:
            return False  # Position occupied
            
        # Calculate actual movement cost (may be modified by subclasses)
        move_cost = self.get_move_cost()
        
        if not self.spend_stamina(move_cost):
            return False
            
        # Update grid
        if self.pos:
            self.grid.remove(self.pos)
        self.grid.place(self, new_pos)
        
        # Check for traps at new position
        trap_damage = self.grid.trigger_trap(new_pos)
        if trap_damage > 0:
            self.take_damage(trap_damage, "trap")
            
        self.stats.steps_taken += 1
        self.last_action = ActionType.MOVE
        self.action_history.append(ActionType.MOVE)
        
        logger.debug(f"{self.id} moved to {new_pos}")
        return True
        
    def move_direction(self, direction: str) -> bool:
        """
        Move in a cardinal direction.
        
        Args:
            direction: One of 'up', 'down', 'left', 'right'
            
        Returns:
            True if move was successful
        """
        if not self.pos:
            return False
            
        x, y = self.pos
        
        direction_map = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'north': (0, -1),
            'south': (0, 1),
            'west': (-1, 0),
            'east': (1, 0)
        }
        
        if direction.lower() not in direction_map:
            return False
            
        dx, dy = direction_map[direction.lower()]
        new_pos = (x + dx, y + dy)
        
        return self.move_to(new_pos)
        
    def get_move_cost(self) -> int:
        """
        Get the stamina cost for movement.
        Can be overridden by subclasses for special movement rules.
        
        Returns:
            Stamina cost for one movement
        """
        return self.move_cost
        
    def rest(self) -> None:
        """
        Rest to restore stamina and health.
        """
        if not self.is_alive():
            return
            
        # Restore stamina
        stamina_gain = min(10, self.max_stamina - self.stamina)
        self.restore_stamina(stamina_gain)
        
        # Restore some health
        health_gain = min(5, self.max_health - self.health)
        self.heal(health_gain)
        
        self.last_action = ActionType.REST
        self.action_history.append(ActionType.REST)
        
        logger.debug(f"{self.id} rested (stamina: {self.stamina}, health: {self.health})")
        
    def die(self, reason: str = "unknown") -> None:
        """
        Handle agent death.
        
        Args:
            reason: Cause of death for logging
        """
        if not self.alive:
            return
            
        self.alive = False
        self.health = 0
        self.stats.deaths += 1
        
        # Remove from grid
        if self.pos and self.grid:
            self.grid.remove(self.pos)
            self.pos = None
            
        logger.info(f"{self.id} died: {reason}")
        
    def get_visible_entities(self) -> List[Tuple[Tuple[int, int], 'Agent']]:
        """
        Get all entities visible to this agent.
        
        Returns:
            List of (position, agent) tuples within vision range
        """
        if not self.pos or not self.grid:
            return []
            
        return self.grid.get_entities_in_radius(self.pos, self.vision_range)
        
    def distance_to(self, other: 'Agent') -> float:
        """
        Calculate distance to another agent.
        
        Args:
            other: Target agent
            
        Returns:
            Distance to other agent, or float('inf') if positions unknown
        """
        if not self.pos or not other.pos or not self.grid:
            return float('inf')
            
        return self.grid.distance(self.pos, other.pos)
        
    def can_attack(self, target: 'Agent') -> bool:
        """
        Check if this agent can attack the target.
        
        Args:
            target: Target agent
            
        Returns:
            True if attack is possible
        """
        if not self.is_alive() or not target.is_alive():
            return False
            
        # Check if target is adjacent (attack range of 1)
        distance = self.distance_to(target)
        return distance <= 1.5  # Allow diagonal attacks
        
    def attack(self, target: 'Agent', damage: int) -> int:
        """
        Attack another agent.
        
        Args:
            target: Agent to attack
            damage: Damage to deal
            
        Returns:
            Actual damage dealt
        """
        if not self.can_attack(target):
            return 0
            
        actual_damage = target.take_damage(damage, self.id)
        self.stats.damage_dealt += actual_damage
        
        if not target.is_alive():
            self.stats.kills += 1
            
        self.last_action = ActionType.ATTACK
        self.action_history.append(ActionType.ATTACK)
        
        logger.debug(f"{self.id} attacked {target.id} for {actual_damage} damage")
        return actual_damage
        
    def get_state_dict(self) -> dict:
        """
        Get current state as dictionary for serialization.
        
        Returns:
            Dictionary containing agent state
        """
        return {
            "id": self.id,
            "type": type(self).__name__,
            "pos": self.pos,
            "health": self.health,
            "max_health": self.max_health,
            "stamina": self.stamina,
            "max_stamina": self.max_stamina,
            "alive": self.alive,
            "stats": {
                "steps_taken": self.stats.steps_taken,
                "damage_dealt": self.stats.damage_dealt,
                "damage_received": self.stats.damage_received,
                "stamina_spent": self.stats.stamina_spent,
                "kills": self.stats.kills,
                "deaths": self.stats.deaths
            }
        }
        
    @abstractmethod
    def step(self, world: Any) -> None:
        """
        Perform one simulation step.
        Must be implemented by subclasses.
        
        Args:
            world: Reference to the simulation world
        """
        pass
        
    def __str__(self) -> str:
        """String representation for debugging."""
        status = "alive" if self.is_alive() else "dead"
        return f"{self.id}({type(self).__name__}) at {self.pos} - {status} H:{self.health} S:{self.stamina}"
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{type(self).__name__}(id='{self.id}', pos={self.pos}, "
                f"health={self.health}/{self.max_health}, stamina={self.stamina}/{self.max_stamina}, "
                f"alive={self.alive})")