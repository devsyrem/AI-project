"""
Monster agent implementation for Predator: Badlands simulation.
Implements small wildlife creatures that wander and attack predators.
"""
from typing import Optional, Tuple, List, Any
import random
import logging
from enum import Enum

from agent import Agent, ActionType
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


class MonsterType(Enum):
    """Different types of monsters with varying capabilities."""
    SMALL = "small"  # Weak, fast
    MEDIUM = "medium"  # Balanced
    LARGE = "large"  # Strong, slow
    PACK = "pack"  # Coordinated behavior


class Monster(Agent):
    """
    Monster agent representing small wildlife that can threaten predators.
    Simple AI with wandering, hunting, and pack behavior.
    """
    
    def __init__(
        self,
        agent_id: str,
        monster_type: MonsterType = MonsterType.SMALL,
        pos: Optional[Tuple[int, int]] = None,
        health: int = 30,
        stamina: int = 50,
        attack_power: int = 15,
        vision_range: int = 3
    ):
        """
        Initialize monster agent.
        
        Args:
            agent_id: Unique identifier
            monster_type: Type of monster affecting stats
            pos: Initial position
            health: Health points
            stamina: Stamina points  
            attack_power: Attack damage
            vision_range: Vision range for detecting prey
        """
        # Adjust stats based on monster type
        type_modifiers = {
            MonsterType.SMALL: {"health": 0.7, "attack": 0.8, "stamina": 1.2, "vision": 1.0},
            MonsterType.MEDIUM: {"health": 1.0, "attack": 1.0, "stamina": 1.0, "vision": 1.0},
            MonsterType.LARGE: {"health": 1.5, "attack": 1.3, "stamina": 0.8, "vision": 0.8},
            MonsterType.PACK: {"health": 0.9, "attack": 0.9, "stamina": 1.1, "vision": 1.3}
        }
        
        modifiers = type_modifiers[monster_type]
        
        super().__init__(
            agent_id=agent_id,
            pos=pos,
            health=int(health * modifiers["health"]),
            max_health=int(health * modifiers["health"]),
            stamina=int(stamina * modifiers["stamina"]),
            max_stamina=int(stamina * modifiers["stamina"]),
            move_cost=config.MOVE_STAMINA_COST,
            vision_range=int(vision_range * modifiers["vision"])
        )
        
        self.monster_type = monster_type
        self.attack_power = int(attack_power * modifiers["attack"])
        
        # Behavior parameters
        self.aggression = random.uniform(0.3, 0.8)
        self.territorial_radius = random.randint(3, 8)
        self.pack_coordination_range = 4
        
        # State tracking
        self.home_territory: Optional[Tuple[int, int]] = None
        self.current_target: Optional[Any] = None
        self.hunt_cooldown = 0
        self.flee_threshold = 0.3  # Flee when health below this ratio
        
        # Pack behavior
        self.pack_members: List[Any] = []
        self.pack_leader: Optional[Any] = None
        
        # Movement patterns
        self.movement_pattern = random.choice(["wander", "territorial", "aggressive"])
        self.movement_counter = 0
        
    def set_home_territory(self, pos: Tuple[int, int]) -> None:
        """Set the monster's home territory center."""
        self.home_territory = pos
        
    def add_to_pack(self, other_monsters: List['Monster']) -> None:
        """Add other monsters to this monster's pack."""
        self.pack_members = [m for m in other_monsters if m != self and m.is_alive()]
        
        # Determine pack leader (highest health + attack)
        all_pack = [self] + self.pack_members
        leader = max(all_pack, key=lambda m: m.health + m.attack_power)
        
        for member in all_pack:
            member.pack_leader = leader
            member.pack_members = [m for m in all_pack if m != member]
            
    def is_in_territory(self) -> bool:
        """Check if monster is within its territorial range."""
        if not self.home_territory or not self.pos or not self.grid:
            return True  # No territory defined
            
        distance = self.grid.distance(self.pos, self.home_territory)
        return distance <= self.territorial_radius
        
    def get_pack_coordination_bonus(self) -> float:
        """Calculate combat bonus from nearby pack members."""
        if not self.pack_members or not self.pos:
            return 1.0
            
        nearby_pack = 0
        for member in self.pack_members:
            if member.is_alive() and member.pos:
                distance = self.distance_to(member)
                if distance <= self.pack_coordination_range:
                    nearby_pack += 1
                    
        # Each nearby pack member gives 10% bonus, max 50%
        return min(1.5, 1.0 + (nearby_pack * 0.1))
        
    def find_target(self) -> Optional[Any]:
        """Find the best target to attack or pursue."""
        visible_entities = self.get_visible_entities()
        
        # Prioritize predators and Thia
        potential_targets = []
        for pos, agent in visible_entities:
            if (agent.__class__.__name__ in ['Predator', 'Thia'] and 
                agent.is_alive()):
                
                # Calculate target priority
                distance = self.distance_to(agent)
                priority = 10.0 / (distance + 1)  # Closer = higher priority
                
                # Prefer wounded targets
                if hasattr(agent, 'health') and hasattr(agent, 'max_health'):
                    health_ratio = agent.health / agent.max_health
                    priority *= (2.0 - health_ratio)  # Lower health = higher priority
                    
                # Avoid very strong targets if alone
                if (agent.__class__.__name__ == 'Predator' and 
                    len(self.pack_members) == 0 and
                    hasattr(agent, 'attack_power')):
                    if agent.attack_power > self.attack_power * 1.5:
                        priority *= 0.3  # Reduce priority for strong targets
                        
                potential_targets.append((agent, priority))
                
        if potential_targets:
            # Choose target with highest priority, with some randomness
            potential_targets.sort(key=lambda x: x[1], reverse=True)
            
            # Weighted random selection favoring higher priority targets
            if random.random() < 0.7:  # 70% chance to choose best target
                return potential_targets[0][0]
            else:  # 30% chance for more random selection
                return random.choice(potential_targets)[0]
                
        return None
        
    def should_flee(self) -> bool:
        """Determine if monster should flee instead of fighting."""
        if self.health / self.max_health < self.flee_threshold:
            return True
            
        # Check for overwhelming enemies
        visible_entities = self.get_visible_entities()
        threats = [agent for pos, agent in visible_entities 
                  if (agent.__class__.__name__ == 'Predator' and 
                      agent.is_alive() and
                      self.distance_to(agent) <= 3)]
        
        if len(threats) > 2 and len(self.pack_members) == 0:
            return True  # Outnumbered and alone
            
        return False
        
    def flee_behavior(self) -> bool:
        """Execute flee behavior - move away from threats."""
        if not self.pos:
            return False
            
        visible_entities = self.get_visible_entities()
        threats = [(pos, agent) for pos, agent in visible_entities 
                  if (agent.__class__.__name__ in ['Predator', 'Boss'] and 
                      agent.is_alive())]
        
        if not threats:
            # No immediate threats, return to territory or rest
            if self.home_territory and not self.is_in_territory():
                return self._move_toward(self.home_territory)
            else:
                self.rest()
                return True
                
        # Calculate average threat position
        avg_threat_x = sum(pos[0] for pos, _ in threats) / len(threats)
        avg_threat_y = sum(pos[1] for pos, _ in threats) / len(threats)
        
        # Move away from average threat position
        dx = self.pos[0] - avg_threat_x
        dy = self.pos[1] - avg_threat_y
        
        # Choose direction to increase distance
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
            
        return self.move_direction(direction)
        
    def pack_behavior(self) -> bool:
        """Execute coordinated pack behavior."""
        if not self.pack_members or not self.pack_leader:
            return False
            
        # If this is the pack leader, coordinate the pack
        if self.pack_leader == self:
            return self._lead_pack()
        else:
            return self._follow_pack_leader()
            
    def _lead_pack(self) -> bool:
        """Behavior for pack leader."""
        # Find best target for the pack
        target = self.find_target()
        
        if target:
            # Signal pack to attack (implicitly through targeting)
            self.current_target = target
            
            # Lead by example - attack or move toward target
            if self.can_attack(target):
                bonus = self.get_pack_coordination_bonus()
                damage = int(self.attack_power * bonus)
                actual_damage = self.attack(target, damage)
                logger.debug(f"Pack leader {self.id} attacked {target.id} for {actual_damage} damage (pack bonus: {bonus:.1f})")
                return actual_damage > 0
            else:
                return self._move_toward(target.pos)
        else:
            # No target, patrol or return to territory
            return self._patrol_behavior()
            
    def _follow_pack_leader(self) -> bool:
        """Behavior for pack members following leader."""
        if not self.pack_leader.is_alive():
            # Leader dead, become independent or find new leader
            self.pack_leader = None
            return self._independent_behavior()
            
        # Follow leader's target if close enough
        if (self.pack_leader.current_target and 
            self.distance_to(self.pack_leader) <= self.pack_coordination_range):
            
            target = self.pack_leader.current_target
            
            if self.can_attack(target):
                bonus = self.get_pack_coordination_bonus()
                damage = int(self.attack_power * bonus)
                actual_damage = self.attack(target, damage)
                return actual_damage > 0
            else:
                if target and target.pos:
                    return self._move_toward(target.pos)
                else:
                    return False
        else:
            # Stay close to pack leader
            if self.distance_to(self.pack_leader) > self.pack_coordination_range:
                return self._move_toward(self.pack_leader.pos)
            else:
                return self._patrol_behavior()
                
    def _independent_behavior(self) -> bool:
        """Behavior for independent monsters (not in pack)."""
        # Check if should flee
        if self.should_flee():
            return self.flee_behavior()
            
        # Look for targets
        target = self.find_target()
        
        if target:
            self.current_target = target
            
            # Attack if possible
            if self.can_attack(target):
                actual_damage = self.attack(target, self.attack_power)
                return actual_damage > 0
            else:
                # Move toward target if aggressive enough
                if random.random() < self.aggression:
                    return self._move_toward(target.pos)
                    
        # Default behavior based on movement pattern
        return self._default_movement()
        
    def _patrol_behavior(self) -> bool:
        """Patrol around territory or wander."""
        if not self.is_in_territory() and self.home_territory:
            # Return to territory
            return self._move_toward(self.home_territory)
        else:
            # Patrol within territory
            return self._random_move()
            
    def _default_movement(self) -> bool:
        """Default movement based on monster's movement pattern."""
        self.movement_counter += 1
        
        if self.movement_pattern == "wander":
            # Random wandering
            return self._random_move()
        elif self.movement_pattern == "territorial":
            # Stay in territory
            return self._patrol_behavior()
        elif self.movement_pattern == "aggressive":
            # Actively hunt even without current target
            return self._aggressive_hunt()
        else:
            return self._random_move()
            
    def _aggressive_hunt(self) -> bool:
        """Actively hunt for targets."""
        # Move toward areas where predators might be
        # For simplicity, move toward center of map or random direction
        if self.grid and self.pos:
            center = (self.grid.size // 2, self.grid.size // 2)
            return self._move_toward(center)
        return self._random_move()
        
    def _move_toward(self, target_pos: Tuple[int, int]) -> bool:
        """Move toward target position."""
        if not self.pos or not self.grid or not target_pos:
            return False
            
        x, y = self.pos
        tx, ty = target_pos
        
        # Simple pathfinding
        dx = tx - x
        dy = ty - y
        
        # Handle wrapping if enabled
        if self.grid.wrap:
            if abs(dx) > self.grid.size // 2:
                dx = -dx / abs(dx) * (self.grid.size - abs(dx))
            if abs(dy) > self.grid.size // 2:
                dy = -dy / abs(dy) * (self.grid.size - abs(dy))
        
        # Choose direction
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
            
        return self.move_direction(direction)
        
    def _random_move(self) -> bool:
        """Move in a random direction."""
        directions = ['up', 'down', 'left', 'right']
        
        # Sometimes rest instead of moving
        if random.random() < 0.2:
            self.rest()
            return True
            
        return self.move_direction(random.choice(directions))
        
    def step(self, world) -> None:
        """
        Perform one simulation step.
        
        Args:
            world: Simulation world
        """
        if not self.is_alive():
            return
            
        # Set home territory if not set
        if self.home_territory is None and self.pos:
            self.set_home_territory(self.pos)
            
        # Passive stamina regeneration
        if self.stamina < self.max_stamina:
            self.restore_stamina(config.STAMINA_REGEN_RATE)
            
        # Reduce hunt cooldown
        if self.hunt_cooldown > 0:
            self.hunt_cooldown -= 1
            
        # Choose behavior based on pack status
        if self.monster_type == MonsterType.PACK and self.pack_members:
            self.pack_behavior()
        else:
            self._independent_behavior()
            
    def get_state_dict(self) -> dict:
        """Extended state dictionary including monster-specific data."""
        state = super().get_state_dict()
        state.update({
            "monster_type": self.monster_type.value,
            "attack_power": self.attack_power,
            "aggression": self.aggression,
            "territorial_radius": self.territorial_radius,
            "home_territory": self.home_territory,
            "current_target_id": self.current_target.id if self.current_target else None,
            "pack_size": len(self.pack_members),
            "is_pack_leader": self.pack_leader == self if self.pack_leader else False,
            "movement_pattern": self.movement_pattern,
            "in_territory": self.is_in_territory()
        })
        return state