"""
Boss adversary implementation for Predator: Badlands simulation.
Implements high-HP boss with territorial behavior and adaptive counter-strategies.
"""
from typing import Optional, Tuple, List, Dict, Any
import random
import numpy as np
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from agent import Agent, ActionType
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


class BossMode(Enum):
    """Different behavioral modes for the boss."""
    PATROL = "patrol"
    HUNT = "hunt"
    DEFEND = "defend"
    RAGE = "rage"
    ADAPTIVE = "adaptive"


@dataclass
class AttackPattern:
    """Tracks patterns in predator attacks for adaptation."""
    attacker_id: str
    attack_direction: Tuple[int, int]
    timestamp: int
    damage_dealt: int


class Boss(Agent):
    """
    Boss adversary with high health, territorial behavior, and adaptive AI.
    Learns to counter repeated attack patterns from predators.
    """
    
    def __init__(
        self,
        agent_id: str = "Boss",
        pos: Optional[Tuple[int, int]] = None,
        health: int = 200,
        stamina: int = 150,
        attack_power: int = 40,
        vision_range: int = 8,
        territory_size: int = 3
    ):
        """
        Initialize boss adversary.
        
        Args:
            agent_id: Unique identifier
            pos: Initial position
            health: Health points (much higher than other agents)
            stamina: Stamina points
            attack_power: Attack damage
            vision_range: Extended vision range
            territory_size: Radius of boss territory
        """
        super().__init__(
            agent_id=agent_id,
            pos=pos,
            health=health,
            max_health=health,
            stamina=stamina,
            max_stamina=stamina,
            move_cost=config.MOVE_STAMINA_COST * 2,  # Boss moves slowly
            vision_range=vision_range
        )
        
        self.attack_power = attack_power
        self.territory_size = territory_size
        
        # Territory and positioning
        self.territory_center: Optional[Tuple[int, int]] = None
        self.patrol_points: List[Tuple[int, int]] = []
        self.current_patrol_index = 0
        
        # Behavioral modes
        self.current_mode = BossMode.PATROL
        self.mode_timer = 0
        self.rage_threshold = 0.4  # Activate rage mode when health below this
        
        # Adaptive AI - track attack patterns
        self.attack_history: deque = deque(maxlen=50)  # Last 50 attacks
        self.pattern_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.adaptation_threshold = config.BOSS_COUNTER_THRESHOLD
        self.adaptation_decay = config.BOSS_ADAPTATION_DECAY
        
        # Counter-strategies
        self.active_counters: Dict[str, float] = {}  # attacker_id -> counter_strength
        self.defense_multipliers: Dict[str, float] = {}  # direction -> defense_bonus
        
        # Special abilities
        self.area_attack_cooldown = 0
        self.area_attack_range = 2
        self.defensive_stance = False
        self.regeneration_rate = 2  # HP per turn when not in combat
        
        # Combat state
        self.last_attackers: List[str] = []
        self.combat_timer = 0
        self.total_damage_received = 0
        
    def set_territory(self, center: Tuple[int, int]) -> None:
        """
        Set up boss territory with center and patrol points.
        
        Args:
            center: Center position of territory
        """
        self.territory_center = center
        
        # Create patrol points around territory perimeter
        if self.grid:
            self.patrol_points = []
            for angle in range(0, 360, 45):  # 8 patrol points
                rad = np.radians(angle)
                x = center[0] + int(self.territory_size * np.cos(rad))
                y = center[1] + int(self.territory_size * np.sin(rad))
                patrol_pos = self.grid._normalize_position((x, y))
                self.patrol_points.append(patrol_pos)
                
    def is_in_territory(self) -> bool:
        """Check if boss is within its territory."""
        if not self.territory_center or not self.pos or not self.grid:
            return True
            
        distance = self.grid.distance(self.pos, self.territory_center)
        return distance <= self.territory_size
        
    def analyze_attack_patterns(self) -> Dict[str, Any]:
        """
        Analyze recent attack patterns to identify strategies to counter.
        
        Returns:
            Dictionary of pattern analysis results
        """
        if len(self.attack_history) < 5:
            return {}
            
        analysis = {
            'frequent_attackers': {},
            'common_directions': {},
            'timing_patterns': {},
            'damage_patterns': {}
        }
        
        # Analyze attacker frequency
        attacker_counts = defaultdict(int)
        direction_counts = defaultdict(int)
        
        for pattern in self.attack_history:
            attacker_counts[pattern.attacker_id] += 1
            
            # Convert position to direction string for pattern recognition
            dx, dy = pattern.attack_direction
            if abs(dx) > abs(dy):
                direction = 'horizontal'
            else:
                direction = 'vertical'
            direction_counts[f"{pattern.attacker_id}_{direction}"] += 1
            
        # Identify patterns that meet adaptation threshold
        for attacker, count in attacker_counts.items():
            if count >= self.adaptation_threshold:
                analysis['frequent_attackers'][attacker] = count
                
        for pattern, count in direction_counts.items():
            if count >= self.adaptation_threshold:
                analysis['common_directions'][pattern] = count
                
        return analysis
        
    def adapt_to_patterns(self, analysis: Dict[str, Any]) -> None:
        """
        Adapt boss behavior based on pattern analysis.
        
        Args:
            analysis: Results from pattern analysis
        """
        # Increase counter-strength against frequent attackers
        for attacker, frequency in analysis.get('frequent_attackers', {}).items():
            current_counter = self.active_counters.get(attacker, 0.0)
            # Increase counter strength based on frequency
            new_counter = min(1.0, current_counter + (frequency * 0.1))
            self.active_counters[attacker] = new_counter
            
            logger.info(f"Boss adapted counter to {attacker}: {new_counter:.2f}")
            
        # Adjust defensive positioning against common attack directions
        for pattern, frequency in analysis.get('common_directions', {}).items():
            attacker, direction = pattern.split('_')
            defense_key = f"{attacker}_{direction}"
            current_defense = self.defense_multipliers.get(defense_key, 1.0)
            new_defense = min(2.0, current_defense + (frequency * 0.05))
            self.defense_multipliers[defense_key] = new_defense
            
    def apply_pattern_decay(self) -> None:
        """Apply decay to learned patterns to prevent over-adaptation."""
        # Decay active counters
        for attacker in list(self.active_counters.keys()):
            self.active_counters[attacker] *= self.adaptation_decay
            if self.active_counters[attacker] < 0.1:
                del self.active_counters[attacker]
                
        # Decay defense multipliers
        for pattern in list(self.defense_multipliers.keys()):
            self.defense_multipliers[pattern] *= self.adaptation_decay
            if self.defense_multipliers[pattern] < 1.1:
                del self.defense_multipliers[pattern]
                
    def record_attack(self, attacker: Any, damage: int) -> None:
        """
        Record an attack for pattern analysis.
        
        Args:
            attacker: Agent that attacked
            damage: Damage dealt
        """
        if not attacker.pos or not self.pos:
            return
            
        # Calculate attack direction
        dx = attacker.pos[0] - self.pos[0]
        dy = attacker.pos[1] - self.pos[1]
        
        pattern = AttackPattern(
            attacker_id=attacker.id,
            attack_direction=(dx, dy),
            timestamp=getattr(self, 'world_step', 0),
            damage_dealt=damage
        )
        
        self.attack_history.append(pattern)
        self.last_attackers.append(attacker.id)
        self.combat_timer = 10  # Stay in combat mode
        self.total_damage_received += damage
        
        # Keep only recent attackers
        if len(self.last_attackers) > 5:
            self.last_attackers = self.last_attackers[-5:]
            
        logger.debug(f"Boss recorded attack from {attacker.id}: {damage} damage")
        
    def calculate_damage_reduction(self, attacker: Any) -> float:
        """
        Calculate damage reduction based on active counters.
        
        Args:
            attacker: Attacking agent
            
        Returns:
            Damage reduction multiplier (0.0 to 1.0)
        """
        base_reduction = 0.0
        
        # Counter-strategy reduction
        counter_strength = self.active_counters.get(attacker.id, 0.0)
        base_reduction += counter_strength * 0.5  # Up to 50% reduction
        
        # Defensive stance reduction
        if self.defensive_stance:
            base_reduction += 0.2  # 20% reduction in defensive stance
            
        # Direction-based defense
        if attacker.pos and self.pos:
            dx = attacker.pos[0] - self.pos[0]
            dy = attacker.pos[1] - self.pos[1]
            direction = 'horizontal' if abs(dx) > abs(dy) else 'vertical'
            defense_key = f"{attacker.id}_{direction}"
            
            defense_multiplier = self.defense_multipliers.get(defense_key, 1.0)
            if defense_multiplier > 1.0:
                base_reduction += (defense_multiplier - 1.0) * 0.3
                
        return min(0.8, base_reduction)  # Max 80% damage reduction
        
    def take_damage(self, amount: int, source: str = "unknown") -> int:
        """Override damage taking to include adaptive defenses."""
        # Find the attacking agent for counter-strategy
        attacker = None
        if hasattr(self, 'grid') and self.grid:
            for pos, agent in self.grid.entities.items():
                if getattr(agent, 'id', None) == source:
                    attacker = agent
                    break
                    
        # Calculate damage reduction
        reduction = 0.0
        if attacker:
            reduction = self.calculate_damage_reduction(attacker)
            self.record_attack(attacker, amount)
            
        # Apply reduction
        reduced_amount = int(amount * (1.0 - reduction))
        actual_damage = super().take_damage(reduced_amount, source)
        
        if reduction > 0:
            logger.info(f"Boss reduced {amount} damage to {reduced_amount} "
                       f"(reduction: {reduction:.1%}) from {source}")
            
        # Mode changes based on damage
        if self.health / self.max_health < self.rage_threshold:
            self.current_mode = BossMode.RAGE
            
        return actual_damage
        
    def area_attack(self) -> List[Tuple[Any, int]]:
        """
        Perform area attack hitting all adjacent enemies.
        
        Returns:
            List of (target, damage) tuples
        """
        if not self.pos or self.area_attack_cooldown > 0:
            return []
            
        targets_hit = []
        
        # Get all entities in area attack range
        if self.grid:
            entities_in_range = self.grid.get_entities_in_radius(self.pos, self.area_attack_range)
            
            for pos, entity in entities_in_range:
                if (entity != self and 
                    entity.__class__.__name__ in ['Predator', 'Thia'] and 
                    entity.is_alive()):
                    
                    # Calculate damage with area attack multiplier
                    distance = self.grid.distance(self.pos, pos)
                    damage_multiplier = max(0.5, 1.0 - (distance / self.area_attack_range))
                    damage = int(self.attack_power * 1.5 * damage_multiplier)
                    
                    actual_damage = entity.take_damage(damage, self.id)
                    targets_hit.append((entity, actual_damage))
                    
        self.area_attack_cooldown = 5  # Cooldown period
        self.last_action = ActionType.ATTACK
        
        logger.info(f"Boss area attack hit {len(targets_hit)} targets")
        return targets_hit
        
    def select_mode(self) -> BossMode:
        """
        Select behavioral mode based on current situation.
        
        Returns:
            Selected boss mode
        """
        # Health-based mode selection
        health_ratio = self.health / self.max_health
        
        if health_ratio < self.rage_threshold:
            return BossMode.RAGE
            
        # Check for nearby enemies
        if self.grid and self.pos:
            entities_nearby = self.grid.get_entities_in_radius(self.pos, self.vision_range)
            enemies = [entity for pos, entity in entities_nearby 
                      if (entity.__class__.__name__ in ['Predator', 'Thia'] and 
                          entity.is_alive())]
            
            if len(enemies) >= 2:
                return BossMode.DEFEND
            elif enemies:
                return BossMode.HUNT
                
        # Adaptive mode if learned enough patterns
        if len(self.active_counters) > 0 or len(self.defense_multipliers) > 0:
            if random.random() < 0.3:  # 30% chance to use adaptive mode
                return BossMode.ADAPTIVE
                
        # Default patrol
        return BossMode.PATROL
        
    def execute_patrol_mode(self) -> bool:
        """Execute patrol behavior."""
        if not self.patrol_points or not self.pos:
            return self._random_move()
            
        # Move toward current patrol point
        target_patrol = self.patrol_points[self.current_patrol_index]
        
        if self.grid and self.grid.distance(self.pos, target_patrol) < 1.5:
            # Reached patrol point, move to next
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
            
        return self._move_toward(target_patrol)
        
    def execute_hunt_mode(self) -> bool:
        """Execute hunting behavior."""
        # Find closest enemy
        if not self.grid or not self.pos:
            return False
            
        entities_nearby = self.grid.get_entities_in_radius(self.pos, self.vision_range)
        enemies = [(pos, entity) for pos, entity in entities_nearby 
                  if (entity.__class__.__name__ in ['Predator', 'Thia'] and 
                      entity.is_alive())]
        
        if not enemies:
            return self.execute_patrol_mode()
            
        # Target closest enemy, but prefer previous attackers
        target_entity = None
        
        # Prioritize recent attackers
        for pos, entity in enemies:
            if entity.id in self.last_attackers:
                target_entity = entity
                break
                
        # If no recent attacker, choose closest
        if not target_entity:
            closest = min(enemies, key=lambda x: self.grid.distance(self.pos, x[0]))
            target_entity = closest[1]
            
        # Attack if in range, otherwise move closer
        if self.can_attack(target_entity):
            actual_damage = self.attack(target_entity, self.attack_power)
            return actual_damage > 0
        else:
            return self._move_toward(target_entity.pos)
            
    def execute_defend_mode(self) -> bool:
        """Execute defensive behavior."""
        self.defensive_stance = True
        
        # Use area attack if multiple enemies nearby and cooldown ready
        if (self.area_attack_cooldown == 0 and 
            self.grid and self.pos):
            
            entities_in_range = self.grid.get_entities_in_radius(self.pos, self.area_attack_range)
            enemies_in_range = sum(1 for pos, entity in entities_in_range 
                                 if (entity.__class__.__name__ in ['Predator', 'Thia'] and 
                                     entity.is_alive()))
            
            if enemies_in_range >= 2:
                self.area_attack()
                return True
                
        # Otherwise, hunt normally but stay defensive
        return self.execute_hunt_mode()
        
    def execute_rage_mode(self) -> bool:
        """Execute rage mode with increased aggression."""
        self.defensive_stance = False
        
        # Increased attack power and speed in rage mode
        original_attack = self.attack_power
        self.attack_power = int(original_attack * 1.3)
        
        # More aggressive hunting
        result = self.execute_hunt_mode()
        
        # Area attack more frequently
        if self.area_attack_cooldown == 0:
            self.area_attack()
            
        self.attack_power = original_attack  # Restore normal attack
        return result
        
    def execute_adaptive_mode(self) -> bool:
        """Execute adaptive behavior using learned patterns."""
        # Analyze recent patterns and adapt
        analysis = self.analyze_attack_patterns()
        if analysis:
            self.adapt_to_patterns(analysis)
            
        # Use counters strategically
        # If we know an attacker's pattern, try to position against it
        if self.active_counters and self.grid and self.pos:
            # Find the most countered attacker nearby
            entities_nearby = self.grid.get_entities_in_radius(self.pos, self.vision_range)
            countered_enemies = []
            
            for pos, entity in entities_nearby:
                if (entity.__class__.__name__ == 'Predator' and 
                    entity.is_alive() and 
                    entity.id in self.active_counters):
                    counter_strength = self.active_counters[entity.id]
                    countered_enemies.append((entity, counter_strength))
                    
            if countered_enemies:
                # Target the most countered enemy
                best_target = max(countered_enemies, key=lambda x: x[1])[0]
                
                if self.can_attack(best_target):
                    # Apply counter-strategy bonus damage
                    bonus_damage = int(self.attack_power * 0.5)
                    total_damage = self.attack_power + bonus_damage
                    actual_damage = self.attack(best_target, total_damage)
                    logger.info(f"Boss counter-attack on {best_target.id}: {actual_damage} damage")
                    return actual_damage > 0
                else:
                    return self._move_toward(best_target.pos)
                    
        # Fall back to hunt mode
        return self.execute_hunt_mode()
        
    def _move_toward(self, target_pos: Tuple[int, int]) -> bool:
        """Move toward target position with boss pathfinding."""
        if not self.pos or not self.grid:
            return False
            
        x, y = self.pos
        tx, ty = target_pos
        
        # Boss uses more sophisticated pathfinding
        dx = tx - x
        dy = ty - y
        
        # Handle wrapping if enabled
        if self.grid.wrap:
            if abs(dx) > self.grid.size // 2:
                dx = -dx / abs(dx) * (self.grid.size - abs(dx))
            if abs(dy) > self.grid.size // 2:
                dy = -dy / abs(dy) * (self.grid.size - abs(dy))
        
        # Choose direction, prefer diagonal movement occasionally
        if random.random() < 0.3 and abs(dx) > 0 and abs(dy) > 0:
            # Diagonal movement (choose both x and y movement)
            x_dir = 'right' if dx > 0 else 'left'
            y_dir = 'down' if dy > 0 else 'up'
            
            # Try both directions, prefer one that gets us closer
            if abs(dx) > abs(dy):
                return self.move_direction(x_dir) or self.move_direction(y_dir)
            else:
                return self.move_direction(y_dir) or self.move_direction(x_dir)
        else:
            # Standard movement
            if abs(dx) > abs(dy):
                direction = 'right' if dx > 0 else 'left'
            else:
                direction = 'down' if dy > 0 else 'up'
                
            return self.move_direction(direction)
            
    def _random_move(self) -> bool:
        """Random movement for patrol."""
        directions = ['up', 'down', 'left', 'right']
        
        # Sometimes rest instead of moving
        if random.random() < 0.1:
            self.rest()
            return True
            
        return self.move_direction(random.choice(directions))
        
    def step(self, world) -> None:
        """
        Perform one simulation step with adaptive behavior.
        
        Args:
            world: Simulation world
        """
        if not self.is_alive():
            return
            
        # Store world step for timestamps
        self.world_step = world.step_count
        
        # Set territory if not set
        if self.territory_center is None and self.pos:
            self.set_territory(self.pos)
            
        # Passive regeneration when not in combat
        if self.combat_timer == 0 and self.health < self.max_health:
            self.heal(self.regeneration_rate)
            
        # Passive stamina regeneration
        if self.stamina < self.max_stamina:
            self.restore_stamina(config.STAMINA_REGEN_RATE * 2)  # Boss regenerates faster
            
        # Reduce cooldowns
        if self.area_attack_cooldown > 0:
            self.area_attack_cooldown -= 1
            
        if self.combat_timer > 0:
            self.combat_timer -= 1
            
        # Apply pattern decay
        if world.step_count % 10 == 0:  # Every 10 steps
            self.apply_pattern_decay()
            
        # Mode selection and execution
        new_mode = self.select_mode()
        if new_mode != self.current_mode:
            logger.debug(f"Boss mode change: {self.current_mode.value} -> {new_mode.value}")
            self.current_mode = new_mode
            self.mode_timer = 0
            
        # Execute current mode
        if self.current_mode == BossMode.PATROL:
            self.execute_patrol_mode()
        elif self.current_mode == BossMode.HUNT:
            self.execute_hunt_mode()
        elif self.current_mode == BossMode.DEFEND:
            self.execute_defend_mode()
        elif self.current_mode == BossMode.RAGE:
            self.execute_rage_mode()
        elif self.current_mode == BossMode.ADAPTIVE:
            self.execute_adaptive_mode()
            
        # Reset defensive stance each turn (must be actively maintained)
        self.defensive_stance = False
        
        self.mode_timer += 1
        
    def get_state_dict(self) -> dict:
        """Extended state dictionary including boss-specific data."""
        state = super().get_state_dict()
        state.update({
            "attack_power": self.attack_power,
            "territory_center": self.territory_center,
            "territory_size": self.territory_size,
            "current_mode": self.current_mode.value,
            "defensive_stance": self.defensive_stance,
            "combat_timer": self.combat_timer,
            "total_damage_received": self.total_damage_received,
            "active_counters": dict(self.active_counters),
            "defense_multipliers": dict(self.defense_multipliers),
            "attack_patterns_learned": len(self.attack_history),
            "area_attack_cooldown": self.area_attack_cooldown,
            "rage_active": self.current_mode == BossMode.RAGE,
            "adaptation_strength": len(self.active_counters) + len(self.defense_multipliers)
        })
        return state