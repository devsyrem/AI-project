"""
Swarm Intelligence system for Predator: Badlands.
Implements flocking algorithms, pheromone trails, and emergent collective behavior.
"""
import math
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SwarmBehaviorType(Enum):
    """Types of swarm behaviors."""
    FLOCKING = "flocking"
    HUNTING = "hunting"  
    FORAGING = "foraging"
    DEFENSIVE = "defensive"
    TERRITORIAL = "territorial"
    MIGRATION = "migration"


@dataclass
class PheromoneTrail:
    """Represents a pheromone trail in the environment."""
    pos: Tuple[int, int]
    strength: float
    trail_type: str  # 'food', 'danger', 'territory', 'path'
    age: int
    decay_rate: float = 0.95
    
    def update(self) -> bool:
        """Update pheromone trail, return False if expired."""
        self.age += 1
        self.strength *= self.decay_rate
        return self.strength > 0.01


@dataclass
class SwarmMemory:
    """Collective memory shared across swarm members."""
    food_locations: List[Tuple[int, int]]
    danger_zones: List[Tuple[int, int]]
    safe_zones: List[Tuple[int, int]]
    enemy_positions: Dict[int, Tuple[int, int]]  # agent_id -> position
    territorial_boundaries: List[Tuple[int, int]]
    last_updated: int = 0
    
    def add_food_location(self, pos: Tuple[int, int]) -> None:
        """Add discovered food location."""
        if pos not in self.food_locations:
            self.food_locations.append(pos)
    
    def add_danger_zone(self, pos: Tuple[int, int]) -> None:
        """Add dangerous area."""
        if pos not in self.danger_zones:
            self.danger_zones.append(pos)
    
    def update_enemy_position(self, agent_id: int, pos: Tuple[int, int]) -> None:
        """Update enemy tracking."""
        self.enemy_positions[agent_id] = pos


class SwarmIntelligence:
    """
    Advanced swarm intelligence system implementing Reynolds flocking rules
    plus pheromone communication and emergent behaviors.
    """
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.pheromone_map: Dict[Tuple[int, int], List[PheromoneTrail]] = {}
        self.collective_memory = SwarmMemory([], [], [], {}, [])
        
        # Flocking parameters
        self.separation_radius = 2.0
        self.alignment_radius = 4.0
        self.cohesion_radius = 6.0
        self.separation_weight = 2.0
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        
        # Pheromone parameters
        self.pheromone_strength = 100.0
        self.pheromone_detection_radius = 3
        
        # Swarm behavior weights
        self.behavior_weights = {
            SwarmBehaviorType.FLOCKING: 1.0,
            SwarmBehaviorType.HUNTING: 1.5,
            SwarmBehaviorType.FORAGING: 0.8,
            SwarmBehaviorType.DEFENSIVE: 2.0,
            SwarmBehaviorType.TERRITORIAL: 1.2
        }
    
    def calculate_flocking_force(self, agent_pos: Tuple[int, int], 
                               neighbors: List[Tuple[Tuple[int, int], float]]) -> Tuple[float, float]:
        """
        Calculate flocking force using Reynolds rules.
        
        Args:
            agent_pos: Current agent position
            neighbors: List of (position, velocity_direction) for nearby agents
            
        Returns:
            (force_x, force_y) normalized flocking force
        """
        if not neighbors:
            return (0.0, 0.0)
        
        ax, ay = agent_pos
        
        # Separation: steer to avoid crowding local flockmates
        sep_x, sep_y = 0.0, 0.0
        alignment_x, alignment_y = 0.0, 0.0
        cohesion_x, cohesion_y = 0.0, 0.0
        
        sep_count = 0
        
        for (nx, ny), velocity_dir in neighbors:
            # Calculate wrapped distance
            dx = self._wrapped_distance(nx - ax, self.grid_size)
            dy = self._wrapped_distance(ny - ay, self.grid_size)
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.separation_radius and distance > 0:
                # Separation: move away from too-close neighbors
                sep_x -= dx / distance
                sep_y -= dy / distance
                sep_count += 1
            
            if distance < self.alignment_radius:
                # Alignment: steer towards average heading of neighbors
                alignment_x += math.cos(velocity_dir)
                alignment_y += math.sin(velocity_dir)
            
            if distance < self.cohesion_radius:
                # Cohesion: steer towards average position of neighbors
                cohesion_x += nx
                cohesion_y += ny
        
        # Normalize separation
        if sep_count > 0:
            sep_x /= sep_count
            sep_y /= sep_count
        
        # Normalize alignment
        neighbor_count = len(neighbors)
        if neighbor_count > 0:
            alignment_x /= neighbor_count
            alignment_y /= neighbor_count
            
            # Cohesion: steer towards center of mass
            center_x = cohesion_x / neighbor_count
            center_y = cohesion_y / neighbor_count
            cohesion_x = self._wrapped_distance(center_x - ax, self.grid_size)
            cohesion_y = self._wrapped_distance(center_y - ay, self.grid_size)
        
        # Combine forces with weights
        total_x = (sep_x * self.separation_weight + 
                  alignment_x * self.alignment_weight + 
                  cohesion_x * self.cohesion_weight)
        total_y = (sep_y * self.separation_weight + 
                  alignment_y * self.alignment_weight + 
                  cohesion_y * self.cohesion_weight)
        
        # Normalize result
        magnitude = math.sqrt(total_x*total_x + total_y*total_y)
        if magnitude > 0:
            return (total_x / magnitude, total_y / magnitude)
        
        return (0.0, 0.0)
    
    def deposit_pheromone(self, pos: Tuple[int, int], trail_type: str, 
                         strength: Optional[float] = None) -> None:
        """Deposit pheromone trail at position."""
        if strength is None:
            strength = self.pheromone_strength
            
        if pos not in self.pheromone_map:
            self.pheromone_map[pos] = []
        
        # Check if pheromone of this type already exists
        for trail in self.pheromone_map[pos]:
            if trail.trail_type == trail_type:
                # Reinforce existing trail
                trail.strength = min(trail.strength + strength * 0.5, 
                                   self.pheromone_strength * 1.5)
                trail.age = 0  # Reset age
                return
        
        # Create new pheromone trail
        new_trail = PheromoneTrail(pos, strength, trail_type, 0)
        self.pheromone_map[pos].append(new_trail)
    
    def detect_pheromones(self, pos: Tuple[int, int], 
                         trail_types: Optional[List[str]] = None) -> List[PheromoneTrail]:
        """Detect pheromones in detection radius."""
        detected = []
        ax, ay = pos
        
        for radius in range(1, self.pheromone_detection_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        check_x = (ax + dx) % self.grid_size
                        check_y = (ay + dy) % self.grid_size
                        check_pos = (check_x, check_y)
                        
                        if check_pos in self.pheromone_map:
                            for trail in self.pheromone_map[check_pos]:
                                if trail_types is None or trail.trail_type in trail_types:
                                    detected.append(trail)
        
        return detected
    
    def calculate_pheromone_gradient(self, pos: Tuple[int, int], 
                                   trail_type: str) -> Tuple[float, float]:
        """Calculate gradient following pheromone trails."""
        ax, ay = pos
        gradient_x, gradient_y = 0.0, 0.0
        
        for (px, py), trails in self.pheromone_map.items():
            for trail in trails:
                if trail.trail_type == trail_type:
                    # Calculate wrapped distance
                    dx = self._wrapped_distance(px - ax, self.grid_size)
                    dy = self._wrapped_distance(py - ay, self.grid_size)
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0 and distance <= self.pheromone_detection_radius:
                        # Gradient strength decreases with distance
                        strength = trail.strength / (distance * distance)
                        gradient_x += (dx / distance) * strength
                        gradient_y += (dy / distance) * strength
        
        # Normalize gradient
        magnitude = math.sqrt(gradient_x*gradient_x + gradient_y*gradient_y)
        if magnitude > 0:
            return (gradient_x / magnitude, gradient_y / magnitude)
        
        return (0.0, 0.0)
    
    def update_pheromones(self) -> None:
        """Update all pheromone trails (decay and cleanup)."""
        expired_positions = []
        
        for pos, trails in self.pheromone_map.items():
            active_trails = []
            for trail in trails:
                if trail.update():  # Returns False if expired
                    active_trails.append(trail)
            
            if active_trails:
                self.pheromone_map[pos] = active_trails
            else:
                expired_positions.append(pos)
        
        # Clean up expired positions
        for pos in expired_positions:
            del self.pheromone_map[pos]
    
    def determine_swarm_behavior(self, swarm_members: List[Tuple[int, int]], 
                                threats: List[Tuple[int, int]], 
                                resources: List[Tuple[int, int]]) -> SwarmBehaviorType:
        """
        Determine optimal swarm behavior based on current situation.
        
        Args:
            swarm_members: Positions of swarm members
            threats: Positions of threats (predators, etc.)
            resources: Positions of resources (food, etc.)
            
        Returns:
            Recommended swarm behavior type
        """
        if not swarm_members:
            return SwarmBehaviorType.FLOCKING
        
        swarm_size = len(swarm_members)
        threat_count = len(threats)
        resource_count = len(resources)
        
        # Calculate swarm density
        if swarm_size > 1:
            distances = []
            for i, pos1 in enumerate(swarm_members):
                for pos2 in swarm_members[i+1:]:
                    dist = self._calculate_wrapped_distance(pos1, pos2)
                    distances.append(dist)
            avg_density = sum(distances) / len(distances) if distances else 0
        else:
            avg_density = 0
        
        # Decision logic based on situation
        if threat_count > swarm_size * 0.5:
            # Many threats - go defensive or flee
            return SwarmBehaviorType.DEFENSIVE
        elif resource_count > 0 and swarm_size > 3:
            # Resources available and sufficient numbers - hunt/forage
            return SwarmBehaviorType.HUNTING if threat_count > 0 else SwarmBehaviorType.FORAGING
        elif avg_density > self.cohesion_radius * 2:
            # Too spread out - increase cohesion
            return SwarmBehaviorType.FLOCKING
        elif swarm_size > 10 and threat_count == 0:
            # Large peaceful swarm - establish territory
            return SwarmBehaviorType.TERRITORIAL
        else:
            # Default flocking behavior
            return SwarmBehaviorType.FLOCKING
    
    def calculate_swarm_force(self, agent_pos: Tuple[int, int], 
                            swarm_members: List[Tuple[int, int]],
                            behavior: SwarmBehaviorType,
                            targets: Optional[List[Tuple[int, int]]] = None) -> Tuple[float, float]:
        """
        Calculate combined swarm force for an agent.
        
        Args:
            agent_pos: Current agent position
            swarm_members: Positions of other swarm members
            behavior: Current swarm behavior
            targets: Optional target positions (threats, resources, etc.)
            
        Returns:
            (force_x, force_y) normalized swarm force
        """
        # Get neighbors for flocking calculations
        neighbors = []
        for member_pos in swarm_members:
            if member_pos != agent_pos:
                dist = self._calculate_wrapped_distance(agent_pos, member_pos)
                if dist <= self.cohesion_radius:
                    # Mock velocity direction (could be enhanced with actual velocity tracking)
                    velocity_dir = random.uniform(0, 2 * math.pi)
                    neighbors.append((member_pos, velocity_dir))
        
        # Base flocking force
        flocking_force = self.calculate_flocking_force(agent_pos, neighbors)
        
        # Behavior-specific modifications
        if behavior == SwarmBehaviorType.HUNTING and targets:
            # Add attraction to nearest target
            target_force = self._calculate_attraction_force(agent_pos, targets, 1.5)
            return self._combine_forces([flocking_force, target_force], [1.0, 2.0])
        
        elif behavior == SwarmBehaviorType.DEFENSIVE and targets:
            # Add repulsion from threats
            repulsion_force = self._calculate_repulsion_force(agent_pos, targets, 2.0)
            return self._combine_forces([flocking_force, repulsion_force], [0.5, 2.5])
        
        elif behavior == SwarmBehaviorType.TERRITORIAL:
            # Enhance cohesion and reduce separation
            enhanced_flocking = (flocking_force[0] * 0.5, flocking_force[1] * 0.5)
            return enhanced_flocking
        
        elif behavior == SwarmBehaviorType.FORAGING:
            # Add exploration bias
            exploration_force = (random.uniform(-1, 1), random.uniform(-1, 1))
            exploration_magnitude = math.sqrt(exploration_force[0]**2 + exploration_force[1]**2)
            if exploration_magnitude > 0:
                exploration_force = (exploration_force[0]/exploration_magnitude, 
                                   exploration_force[1]/exploration_magnitude)
            return self._combine_forces([flocking_force, exploration_force], [1.0, 0.3])
        
        return flocking_force
    
    def _wrapped_distance(self, diff: float, grid_size: int) -> float:
        """Calculate shortest distance considering wrapping."""
        if diff > grid_size / 2:
            return diff - grid_size
        elif diff < -grid_size / 2:
            return diff + grid_size
        return diff
    
    def _calculate_wrapped_distance(self, pos1: Tuple[int, int], 
                                  pos2: Tuple[int, int]) -> float:
        """Calculate wrapped distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        dx = self._wrapped_distance(x2 - x1, self.grid_size)
        dy = self._wrapped_distance(y2 - y1, self.grid_size)
        return math.sqrt(dx*dx + dy*dy)
    
    def _calculate_attraction_force(self, pos: Tuple[int, int], 
                                 targets: List[Tuple[int, int]], 
                                 strength: float) -> Tuple[float, float]:
        """Calculate attraction force towards targets."""
        if not targets:
            return (0.0, 0.0)
        
        # Find nearest target
        nearest_target = min(targets, 
                           key=lambda t: self._calculate_wrapped_distance(pos, t))
        
        ax, ay = pos
        tx, ty = nearest_target
        dx = self._wrapped_distance(tx - ax, self.grid_size)
        dy = self._wrapped_distance(ty - ay, self.grid_size)
        
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            return (dx/distance * strength, dy/distance * strength)
        
        return (0.0, 0.0)
    
    def _calculate_repulsion_force(self, pos: Tuple[int, int], 
                                 threats: List[Tuple[int, int]], 
                                 strength: float) -> Tuple[float, float]:
        """Calculate repulsion force away from threats."""
        if not threats:
            return (0.0, 0.0)
        
        total_x, total_y = 0.0, 0.0
        ax, ay = pos
        
        for tx, ty in threats:
            dx = self._wrapped_distance(tx - ax, self.grid_size)
            dy = self._wrapped_distance(ty - ay, self.grid_size)
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Repulsion decreases with distance
                repulsion_strength = strength / (distance + 0.1)
                total_x -= (dx/distance) * repulsion_strength
                total_y -= (dy/distance) * repulsion_strength
        
        # Normalize
        magnitude = math.sqrt(total_x*total_x + total_y*total_y)
        if magnitude > 0:
            return (total_x/magnitude, total_y/magnitude)
        
        return (0.0, 0.0)
    
    def _combine_forces(self, forces: List[Tuple[float, float]], 
                       weights: List[float]) -> Tuple[float, float]:
        """Combine multiple forces with weights."""
        total_x = sum(f[0] * w for f, w in zip(forces, weights))
        total_y = sum(f[1] * w for f, w in zip(forces, weights))
        
        # Normalize result
        magnitude = math.sqrt(total_x*total_x + total_y*total_y)
        if magnitude > 0:
            return (total_x/magnitude, total_y/magnitude)
        
        return (0.0, 0.0)
    
    def get_swarm_state(self) -> Dict[str, any]:
        """Get current state of swarm intelligence system."""
        return {
            'pheromone_trails': len(self.pheromone_map),
            'total_pheromone_strength': sum(
                sum(trail.strength for trail in trails) 
                for trails in self.pheromone_map.values()
            ),
            'collective_memory': {
                'food_locations': len(self.collective_memory.food_locations),
                'danger_zones': len(self.collective_memory.danger_zones),
                'tracked_enemies': len(self.collective_memory.enemy_positions)
            }
        }