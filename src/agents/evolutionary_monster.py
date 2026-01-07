"""
Enhanced Monster agent with Swarm Intelligence and Genetic Evolution.
Integrates flocking behavior, pheromone communication, and evolved traits.
"""
from typing import Optional, Tuple, List, Any, Dict
import random
import logging
import math
from enum import Enum

from agent import Agent, ActionType
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

# Import swarm and genetic systems
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from swarm_intelligence import SwarmIntelligence, SwarmBehaviorType
from genetic_evolution import GeneticEvolutionEngine, GeneType

logger = logging.getLogger(__name__)


class MonsterType(Enum):
    """Enhanced monster types with evolutionary potential."""
    SMALL = "small"      # Fast, low health, high reproduction
    MEDIUM = "medium"    # Balanced stats
    LARGE = "large"      # High health, slow, territorial
    PACK = "pack"        # Enhanced swarm coordination
    EVOLVED = "evolved"  # Genetically enhanced variants


class EvolutionaryMonster(Agent):
    """
    Advanced Monster agent with swarm intelligence and genetic evolution.
    Features flocking behavior, pheromone communication, and evolutionary traits.
    """
    
    def __init__(
        self,
        agent_id: str,
        monster_type: MonsterType = MonsterType.SMALL,
        pos: Optional[Tuple[int, int]] = None,
        health: int = 30,
        stamina: int = 50,
        attack_power: int = 15,
        vision_range: int = 3,
        genetic_id: Optional[int] = None,
        evolved_traits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize evolutionary monster agent.
        
        Args:
            agent_id: Unique identifier
            monster_type: Type of monster
            pos: Initial position
            health: Base health points
            stamina: Base stamina points  
            attack_power: Base attack damage
            vision_range: Vision range
            genetic_id: ID in genetic evolution system
            evolved_traits: Genetic trait modifiers
        """
        # Apply genetic modifications to base stats
        if evolved_traits:
            health = int(health * (1 + evolved_traits.get('stamina_modifier', 0) * 0.5))
            stamina = int(stamina * (1 + evolved_traits.get('stamina_modifier', 0)))
            attack_power = int(attack_power * (1 + evolved_traits.get('aggression_modifier', 0) * 0.3))
            vision_range = max(1, int(vision_range * (1 + evolved_traits.get('intelligence_modifier', 0) * 0.2)))
        
        super().__init__(agent_id, monster_type, pos, health, stamina, attack_power, vision_range)
        
        self.genetic_id = genetic_id
        self.evolved_traits = evolved_traits or {}
        
        # Swarm intelligence state
        self.swarm_members: List[Tuple[int, int]] = []  # Positions of nearby swarm members
        self.current_swarm_behavior = SwarmBehaviorType.FLOCKING
        self.pheromone_last_deposit = 0
        self.pack_leader: Optional['EvolutionaryMonster'] = None
        self.pack_members: List['EvolutionaryMonster'] = []
        
        # Enhanced monster attributes
        self.territory_center: Optional[Tuple[int, int]] = None
        self.territory_radius = 5
        self.hunting_target: Optional[Agent] = None
        self.last_target_seen: Optional[Tuple[int, int]] = None
        self.fear_level = 0.0  # Current fear state (0.0 to 1.0)
        
        # Evolutionary fitness tracking
        self.survival_time = 0
        self.damage_dealt_total = 0
        self.offspring_count = 0
        self.territory_time = 0
        
        # Movement and behavior state
        self.movement_history: List[Tuple[int, int]] = []
        self.stuck_counter = 0
        self.exploration_bias = random.uniform(-1, 1)
        
        logger.debug(f"Created evolutionary monster {agent_id} of type {monster_type.value}")
    
    def step(self, world) -> None:
        """Execute one step of monster behavior with swarm intelligence."""
        if not self.alive or not self.pos or not self.grid:
            return
        
        # Update fitness tracking
        self.survival_time += 1
        if self.territory_center and self._distance_to_territory_center() <= self.territory_radius:
            self.territory_time += 1
        
        # Update swarm member positions
        self._update_swarm_awareness(world)
        
        # Determine current swarm behavior
        threats = self._detect_threats(world)
        resources = self._detect_resources(world)
        
        if hasattr(world, 'swarm_intelligence'):
            self.current_swarm_behavior = world.swarm_intelligence.determine_swarm_behavior(
                self.swarm_members, threats, resources
            )
        else:
            self.current_swarm_behavior = self._basic_behavior_selection(threats, resources)
        
        # Deposit pheromones based on behavior and genetic traits
        self._manage_pheromones(world)
        
        # Execute behavior based on swarm intelligence and genetic traits
        action_taken = self._execute_swarm_behavior(world, threats, resources)
        
        # Update fear level based on threats
        self._update_fear_level(threats)
        
        # Regenerate stamina
        if self.stamina < self.max_stamina:
            regen_rate = 2 + int(self.evolved_traits.get('stamina_modifier', 0) * 3)
            self.stamina = min(self.max_stamina, self.stamina + regen_rate)
        
        # Track movement for anti-stuck mechanism
        self._track_movement()
    
    def _update_swarm_awareness(self, world) -> None:
        """Update awareness of nearby swarm members."""
        self.swarm_members = []
        
        if not hasattr(world, 'agents'):
            return
        
        enhanced_vision = self.vision_range + int(self.evolved_traits.get('pack_affinity', 0) * 2)
        
        for agent in world.agents.values():
            if (isinstance(agent, EvolutionaryMonster) and 
                agent != self and agent.alive and agent.pos):
                
                distance = self.grid.distance(self.pos, agent.pos)
                if distance <= enhanced_vision:
                    self.swarm_members.append(agent.pos)
    
    def _detect_threats(self, world) -> List[Tuple[int, int]]:
        """Detect threatening agents (predators, boss)."""
        threats = []
        
        if not hasattr(world, 'agents'):
            return threats
        
        for agent in world.agents.values():
            if agent != self and agent.alive and agent.pos:
                # Enhanced threat detection based on intelligence
                detection_range = self.vision_range + int(self.evolved_traits.get('intelligence_modifier', 0) * 2)
                distance = self.grid.distance(self.pos, agent.pos)
                
                if distance <= detection_range:
                    # Check if agent is threatening
                    agent_type = type(agent).__name__
                    if agent_type in ['Predator', 'Boss']:
                        threats.append(agent.pos)
                        
                        # Update collective memory if available
                        if hasattr(world, 'swarm_intelligence'):
                            world.swarm_intelligence.collective_memory.update_enemy_position(
                                id(agent), agent.pos
                            )
        
        return threats
    
    def _detect_resources(self, world) -> List[Tuple[int, int]]:
        """Detect valuable resources in the environment."""
        resources = []
        
        # Look for food sources, weak agents, or territorial advantages
        if hasattr(world, 'agents'):
            for agent in world.agents.values():
                if agent != self and agent.alive and agent.pos:
                    distance = self.grid.distance(self.pos, agent.pos)
                    
                    if distance <= self.vision_range:
                        # Consider weakened agents as potential resources
                        if hasattr(agent, 'health') and agent.health < agent.max_health * 0.3:
                            resources.append(agent.pos)
        
        return resources
    
    def _manage_pheromones(self, world) -> None:
        """Manage pheromone deposition based on behavior and traits."""
        if not hasattr(world, 'swarm_intelligence'):
            return
        
        # Pheromone deposition frequency based on intelligence
        intelligence = self.evolved_traits.get('intelligence_modifier', 0.5)
        deposition_frequency = max(3, int(10 - intelligence * 7))
        
        if (world.step_count - self.pheromone_last_deposit) >= deposition_frequency:
            swarm_intel = world.swarm_intelligence
            
            if self.current_swarm_behavior == SwarmBehaviorType.HUNTING and self.hunting_target:
                swarm_intel.deposit_pheromone(self.pos, 'hunting')
            elif self.current_swarm_behavior == SwarmBehaviorType.DEFENSIVE:
                swarm_intel.deposit_pheromone(self.pos, 'danger')
            elif self.territory_center:
                swarm_intel.deposit_pheromone(self.pos, 'territory')
            else:
                swarm_intel.deposit_pheromone(self.pos, 'path')
            
            self.pheromone_last_deposit = world.step_count
    
    def _execute_swarm_behavior(self, world, threats: List[Tuple[int, int]], 
                               resources: List[Tuple[int, int]]) -> bool:
        """Execute behavior based on current swarm state and genetic traits."""
        
        # Get swarm force if swarm intelligence is available
        swarm_force = (0.0, 0.0)
        if hasattr(world, 'swarm_intelligence') and self.swarm_members:
            swarm_force = world.swarm_intelligence.calculate_swarm_force(
                self.pos, self.swarm_members, self.current_swarm_behavior, 
                threats + resources
            )
        
        # Genetic trait influences
        aggression = self.evolved_traits.get('aggression_modifier', 0.5)
        fear_threshold = self.evolved_traits.get('fear_threshold', 0.5)
        pack_affinity = self.evolved_traits.get('pack_affinity', 0.5)
        territory_drive = self.evolved_traits.get('territory_drive', 0.5)
        
        # Behavior execution based on swarm behavior type
        if self.current_swarm_behavior == SwarmBehaviorType.HUNTING:
            return self._execute_hunting_behavior(world, threats, resources, swarm_force, aggression)
        
        elif self.current_swarm_behavior == SwarmBehaviorType.DEFENSIVE:
            return self._execute_defensive_behavior(world, threats, swarm_force, fear_threshold)
        
        elif self.current_swarm_behavior == SwarmBehaviorType.TERRITORIAL:
            return self._execute_territorial_behavior(world, swarm_force, territory_drive)
        
        elif self.current_swarm_behavior == SwarmBehaviorType.FLOCKING:
            return self._execute_flocking_behavior(world, swarm_force, pack_affinity)
        
        else:
            # Default behavior - explore
            return self._execute_exploration_behavior(world, swarm_force)
    
    def _execute_hunting_behavior(self, world, threats, resources, swarm_force, aggression) -> bool:
        """Execute coordinated hunting behavior."""
        # Find or maintain hunting target
        if not self.hunting_target or not self.hunting_target.alive:
            self.hunting_target = self._select_hunting_target(world, resources, aggression)
        
        if self.hunting_target and self.hunting_target.pos:
            # Can we attack?
            if self.can_attack(self.hunting_target):
                # Enhanced pack coordination
                pack_bonus = len([m for m in self.swarm_members if 
                                self.grid.distance(self.pos, m) <= 2]) * 0.1
                
                damage = int(self.attack_power * (1 + aggression * 0.3 + pack_bonus))
                actual_damage = self.attack(self.hunting_target, damage)
                self.damage_dealt_total += actual_damage
                
                # Deposit hunting success pheromone
                if hasattr(world, 'swarm_intelligence') and actual_damage > 0:
                    world.swarm_intelligence.deposit_pheromone(self.pos, 'hunting', 150.0)
                
                return True
            else:
                # Move toward target with swarm coordination
                return self._move_toward_target_with_swarm(self.hunting_target.pos, swarm_force)
        
        return False
    
    def _execute_defensive_behavior(self, world, threats, swarm_force, fear_threshold) -> bool:
        """Execute defensive behavior when threatened."""
        if not threats:
            return False
        
        # High fear leads to fleeing
        if self.fear_level > fear_threshold:
            # Calculate flee direction away from threats
            flee_x, flee_y = 0.0, 0.0
            for threat_pos in threats:
                tx, ty = threat_pos
                ax, ay = self.pos
                
                dx = self.grid.normalize_coordinate(ax - tx, self.grid.size)
                dy = self.grid.normalize_coordinate(ay - ty, self.grid.size)
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    flee_x += dx / distance
                    flee_y += dy / distance
            
            # Combine with swarm force
            combined_x = flee_x * 0.7 + swarm_force[0] * 0.3
            combined_y = flee_y * 0.7 + swarm_force[1] * 0.3
            
            return self._move_in_direction((combined_x, combined_y))
        
        else:
            # Form defensive formation with swarm
            return self._move_in_direction(swarm_force)
    
    def _execute_territorial_behavior(self, world, swarm_force, territory_drive) -> bool:
        """Execute territorial behavior."""
        # Establish territory if not set
        if not self.territory_center:
            self.territory_center = self.pos
        
        distance_to_center = self._distance_to_territory_center()
        
        # Strong territory drive keeps monster near center
        if distance_to_center > self.territory_radius * territory_drive:
            # Move toward territory center
            cx, cy = self.territory_center
            ax, ay = self.pos
            
            dx = self.grid.normalize_coordinate(cx - ax, self.grid.size)
            dy = self.grid.normalize_coordinate(cy - ay, self.grid.size)
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                territory_force = (dx / distance, dy / distance)
                # Combine with swarm force
                combined_x = territory_force[0] * 0.6 + swarm_force[0] * 0.4
                combined_y = territory_force[1] * 0.6 + swarm_force[1] * 0.4
                
                return self._move_in_direction((combined_x, combined_y))
        
        # Patrol territory with swarm
        return self._move_in_direction(swarm_force) if swarm_force != (0.0, 0.0) else self._random_move()
    
    def _execute_flocking_behavior(self, world, swarm_force, pack_affinity) -> bool:
        """Execute basic flocking behavior."""
        # Enhanced flocking based on pack affinity
        if pack_affinity > 0.7 and len(self.swarm_members) > 2:
            # Strong pack behavior - tight formation
            enhanced_force = (swarm_force[0] * (1 + pack_affinity), 
                            swarm_force[1] * (1 + pack_affinity))
            return self._move_in_direction(enhanced_force)
        else:
            # Normal flocking with some exploration
            exploration_x = self.exploration_bias * 0.3
            exploration_y = random.uniform(-0.3, 0.3)
            
            combined_x = swarm_force[0] * 0.7 + exploration_x
            combined_y = swarm_force[1] * 0.7 + exploration_y
            
            return self._move_in_direction((combined_x, combined_y))
    
    def _execute_exploration_behavior(self, world, swarm_force) -> bool:
        """Execute exploration behavior."""
        # Follow pheromone trails if available
        if hasattr(world, 'swarm_intelligence'):
            pheromone_gradient = world.swarm_intelligence.calculate_pheromone_gradient(
                self.pos, 'path'
            )
            
            if pheromone_gradient != (0.0, 0.0):
                combined_x = pheromone_gradient[0] * 0.6 + swarm_force[0] * 0.4
                combined_y = pheromone_gradient[1] * 0.6 + swarm_force[1] * 0.4
                return self._move_in_direction((combined_x, combined_y))
        
        # Random exploration with swarm influence
        return self._move_in_direction(swarm_force) if swarm_force != (0.0, 0.0) else self._random_move()
    
    def _select_hunting_target(self, world, resources, aggression) -> Optional[Agent]:
        """Select optimal hunting target based on genetic traits."""
        if not hasattr(world, 'agents'):
            return None
        
        targets = []
        
        for agent in world.agents.values():
            if agent != self and agent.alive and agent.pos:
                distance = self.grid.distance(self.pos, agent.pos)
                
                if distance <= self.vision_range:
                    agent_type = type(agent).__name__
                    
                    # Aggression affects target selection
                    if aggression > 0.7:
                        # High aggression - attack anyone
                        if agent_type in ['Predator', 'Thia', 'Boss']:
                            targets.append((agent, distance))
                    elif aggression > 0.4:
                        # Medium aggression - avoid boss, target others
                        if agent_type in ['Predator', 'Thia']:
                            targets.append((agent, distance))
                    else:
                        # Low aggression - only attack if very advantageous
                        if agent_type == 'Thia' or (agent_type == 'Predator' and 
                                                   hasattr(agent, 'health') and 
                                                   agent.health < agent.max_health * 0.5):
                            targets.append((agent, distance))
        
        if targets:
            # Select nearest target
            targets.sort(key=lambda x: x[1])
            return targets[0][0]
        
        return None
    
    def _move_toward_target_with_swarm(self, target_pos: Tuple[int, int], 
                                      swarm_force: Tuple[float, float]) -> bool:
        """Move toward target while maintaining swarm coordination."""
        if not target_pos:
            return False
        
        ax, ay = self.pos
        tx, ty = target_pos
        
        # Calculate direction to target
        dx = self.grid.normalize_coordinate(tx - ax, self.grid.size)
        dy = self.grid.normalize_coordinate(ty - ay, self.grid.size)
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            target_force = (dx / distance, dy / distance)
            
            # Combine target seeking with swarm coordination
            combined_x = target_force[0] * 0.7 + swarm_force[0] * 0.3
            combined_y = target_force[1] * 0.7 + swarm_force[1] * 0.3
            
            return self._move_in_direction((combined_x, combined_y))
        
        return False
    
    def _move_in_direction(self, direction: Tuple[float, float]) -> bool:
        """Move in the specified direction with genetic speed modifiers."""
        if direction == (0.0, 0.0):
            return False
        
        dx, dy = direction
        
        # Apply speed modifier from genetics
        speed_modifier = self.evolved_traits.get('speed_modifier', 0.5)
        movement_efficiency = 1.0 + (speed_modifier - 0.5) * 0.4
        
        # Calculate movement with efficiency
        if abs(dx) > abs(dy):
            move_x = 1 if dx > 0 else -1
            move_y = 1 if dy > 0 and random.random() < abs(dy) * movement_efficiency else 0
        else:
            move_y = 1 if dy > 0 else -1
            move_x = 1 if dx > 0 and random.random() < abs(dx) * movement_efficiency else 0
        
        return self.try_move(move_x, move_y)
    
    def _basic_behavior_selection(self, threats, resources) -> SwarmBehaviorType:
        """Select behavior when swarm intelligence is not available."""
        if threats:
            return SwarmBehaviorType.DEFENSIVE
        elif resources:
            return SwarmBehaviorType.HUNTING
        elif self.territory_center:
            return SwarmBehaviorType.TERRITORIAL
        else:
            return SwarmBehaviorType.FLOCKING
    
    def _update_fear_level(self, threats: List[Tuple[int, int]]) -> None:
        """Update fear level based on nearby threats."""
        fear_threshold = self.evolved_traits.get('fear_threshold', 0.5)
        
        if threats:
            # Calculate threat pressure
            threat_pressure = min(1.0, len(threats) / 3.0)
            
            # Closer threats increase fear more
            min_distance = float('inf')
            for threat_pos in threats:
                distance = self.grid.distance(self.pos, threat_pos)
                min_distance = min(min_distance, distance)
            
            if min_distance < float('inf'):
                proximity_factor = max(0.1, 1.0 - (min_distance / self.vision_range))
                threat_pressure *= proximity_factor
            
            # Fear increases based on genetic fear threshold
            fear_increase = threat_pressure * (1 - fear_threshold) * 0.1
            self.fear_level = min(1.0, self.fear_level + fear_increase)
        else:
            # Fear decreases when no threats
            self.fear_level = max(0.0, self.fear_level - 0.05)
    
    def _distance_to_territory_center(self) -> float:
        """Calculate distance to territory center."""
        if not self.territory_center or not self.grid or not self.pos:
            return 0.0
        return self.grid.distance(self.pos, self.territory_center)
    
    def _track_movement(self) -> None:
        """Track movement to prevent getting stuck."""
        if self.pos:
            self.movement_history.append(self.pos)
            
            # Keep only recent history
            if len(self.movement_history) > 5:
                self.movement_history.pop(0)
            
            # Check if stuck
            if len(self.movement_history) >= 3:
                if all(pos == self.movement_history[0] for pos in self.movement_history[-3:]):
                    self.stuck_counter += 1
                    if self.stuck_counter > 2:
                        # Force random exploration
                        self.exploration_bias = random.uniform(-1, 1)
                        self.stuck_counter = 0
                else:
                    self.stuck_counter = 0
    
    def get_fitness_metrics(self) -> Dict[str, float]:
        """Get fitness metrics for genetic evolution."""
        return {
            'survival_time': self.survival_time,
            'damage_dealt': self.damage_dealt_total,
            'offspring_count': self.offspring_count,
            'territory_time': self.territory_time / max(1, self.survival_time)  # Normalize
        }
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get comprehensive agent state including swarm and genetic information."""
        base_state = super().get_agent_state()
        
        base_state.update({
            'monster_type': self.monster_type.value,
            'genetic_id': self.genetic_id,
            'evolved_traits': self.evolved_traits,
            'swarm_behavior': self.current_swarm_behavior.value,
            'swarm_members_count': len(self.swarm_members),
            'territory_center': self.territory_center,
            'fear_level': self.fear_level,
            'fitness_metrics': self.get_fitness_metrics()
        })
        
        return base_state


# Maintain backward compatibility
Monster = EvolutionaryMonster