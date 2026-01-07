"""
Thia synthetic agent implementation for Predator: Badlands simulation.
Implements the damaged synthetic with repair mechanics and reconnaissance abilities.
"""
from typing import Optional, Tuple, List, Dict, Any
import random
import logging
from dataclasses import dataclass

from agent import Agent, ActionType
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


@dataclass
class ReconData:
    """Reconnaissance data collected by Thia."""
    entity_type: str
    position: Tuple[int, int]
    timestamp: int
    confidence: float
    additional_info: Dict[str, Any]


class Thia(Agent):
    """
    Thia synthetic agent with damage/repair mechanics and reconnaissance capabilities.
    When functional, provides tactical support to predators.
    """
    
    def __init__(
        self,
        agent_id: str = "Thia",
        pos: Optional[Tuple[int, int]] = None,
        health: int = 80,
        stamina: int = 60,
        vision_range: int = 8
    ):
        """
        Initialize Thia synthetic.
        
        Args:
            agent_id: Unique identifier
            pos: Initial position
            health: Health points (lower than predators)
            stamina: Stamina points
            vision_range: Extended vision range for reconnaissance
        """
        super().__init__(
            agent_id=agent_id,
            pos=pos,
            health=health,
            max_health=health,
            stamina=stamina,
            max_stamina=stamina,
            move_cost=config.MOVE_STAMINA_COST,
            vision_range=vision_range
        )
        
        # Damage and repair mechanics
        self.functional = False  # Starts damaged
        self.repair_progress = 0
        self.repair_threshold = config.THIA_REPAIR_TIME
        self.max_repair_attempts = 3
        self.repair_attempts = 0
        
        # Carrying state
        self.being_carried = False
        self.carrier: Optional[Any] = None
        
        # Reconnaissance capabilities
        self.recon_data: List[ReconData] = []
        self.boss_hint_accuracy = config.THIA_BOSS_HINT_ACCURACY
        self.recon_range = config.THIA_RECONNAISSANCE_RANGE
        
        # Coordination with predators
        self.coordinated_predator: Optional[Any] = None
        self.shared_intel: Dict[str, Any] = {}
        
        # Movement limitations when damaged
        self.damage_movement_penalty = 2  # Extra stamina cost when not functional
        
    def get_move_cost(self) -> int:
        """Calculate movement cost with damage penalty."""
        base_cost = self.move_cost
        if not self.functional:
            base_cost += self.damage_movement_penalty
        return base_cost
        
    def can_move(self) -> bool:
        """Override movement check - cannot move when being carried."""
        if self.being_carried:
            return False
        return super().can_move()
        
    def receive_repair(self) -> bool:
        """
        Receive repair attempt from a predator.
        
        Returns:
            True if repair was successful (became functional)
        """
        if self.functional:
            return True  # Already functional
            
        self.repair_progress += 1
        self.repair_attempts += 1
        
        logger.debug(f"{self.id} repair progress: {self.repair_progress}/{self.repair_threshold}")
        
        # Check if repair is complete
        if self.repair_progress >= self.repair_threshold:
            self.functional = True
            self.repair_progress = 0
            logger.info(f"{self.id} repair complete - now functional!")
            return True
            
        return False
        
    def self_repair_attempt(self) -> bool:
        """
        Attempt self-repair (less effective than predator repair).
        
        Returns:
            True if repair made progress
        """
        if self.functional or self.repair_attempts >= self.max_repair_attempts:
            return False
            
        # Self-repair costs more stamina and is less reliable
        repair_cost = config.THIA_REPAIR_STAMINA_COST // 2
        if not self.spend_stamina(repair_cost):
            return False
            
        # Lower success chance for self-repair
        if random.random() < 0.3:
            return self.receive_repair()
            
        return False
        
    def perform_reconnaissance(self, world) -> List[ReconData]:
        """
        Perform reconnaissance scan of surrounding area.
        
        Args:
            world: Simulation world
            
        Returns:
            List of reconnaissance data collected
        """
        if not self.functional or not self.pos:
            return []
            
        # Extended vision scan
        visible_entities = self.grid.get_entities_in_radius(self.pos, self.recon_range)
        
        recon_results = []
        for pos, entity in visible_entities:
            # Filter for important entities
            entity_type = type(entity).__name__
            if entity_type in ['Boss', 'Monster', 'Predator']:
                confidence = random.uniform(0.7, 1.0)
                
                # Additional info based on entity type
                additional_info = {}
                if entity_type == 'Boss':
                    additional_info['health_estimate'] = entity.health if hasattr(entity, 'health') else 'unknown'
                    additional_info['threat_level'] = 'extreme'
                elif entity_type == 'Monster':
                    additional_info['threat_level'] = 'low'
                elif entity_type == 'Predator':
                    additional_info['clan_member'] = True
                    if hasattr(entity, 'honor'):
                        additional_info['honor_estimate'] = entity.honor
                        
                recon_data = ReconData(
                    entity_type=entity_type,
                    position=pos,
                    timestamp=world.step_count,
                    confidence=confidence,
                    additional_info=additional_info
                )
                
                recon_results.append(recon_data)
                self.recon_data.append(recon_data)
                
        # Limit stored recon data to prevent memory growth
        if len(self.recon_data) > 50:
            self.recon_data = self.recon_data[-50:]
            
        logger.debug(f"{self.id} reconnaissance found {len(recon_results)} entities")
        return recon_results
        
    def provide_boss_hint(self, current_step: int = 0) -> Optional[Tuple[int, int]]:
        """
        Provide hint about boss location based on reconnaissance data.
        
        Args:
            current_step: Current simulation step for age calculation
            
        Returns:
            Estimated boss position or None if no information
        """
        if not self.functional:
            return None
            
        # Find most recent boss sighting
        boss_sightings = [data for data in self.recon_data 
                         if data.entity_type == 'Boss']
        
        if not boss_sightings:
            return None
            
        # Get most recent sighting
        latest_sighting = max(boss_sightings, key=lambda x: x.timestamp)
        
        # Add some uncertainty based on age of sighting and accuracy
        age_factor = min(1.0, 10.0 / (current_step - latest_sighting.timestamp + 1))
        effective_accuracy = self.boss_hint_accuracy * age_factor
        
        if random.random() < effective_accuracy:
            # Accurate hint (with small random offset)
            x, y = latest_sighting.position
            offset = random.randint(-2, 2)
            return (x + offset, y + offset)
        else:
            # Inaccurate hint
            if self.grid:
                return self.grid.random_empty()
            return None
            
    def coordinate_with_predator(self, predator: Any) -> Dict[str, Any]:
        """
        Establish coordination with a predator for tactical support.
        
        Args:
            predator: Predator to coordinate with
            
        Returns:
            Shared tactical intelligence
        """
        if not self.functional:
            return {}
            
        self.coordinated_predator = predator
        
        # Share reconnaissance data
        tactical_intel = {
            'recon_data': self.recon_data.copy(),
            'boss_hint': self.provide_boss_hint(0),  # Will be updated with proper step count
            'hazard_locations': self.get_known_hazards(),
            'safe_routes': self.suggest_safe_routes()
        }
        
        self.shared_intel = tactical_intel
        logger.info(f"{self.id} established coordination with {predator.id}")
        
        return tactical_intel
        
    def get_known_hazards(self) -> List[Tuple[int, int]]:
        """
        Get list of known hazard locations from reconnaissance.
        
        Returns:
            List of hazardous positions
        """
        # For now, return trap locations if we've observed them
        # In a full implementation, this would track observed traps and dangers
        if self.grid:
            return list(self.grid.traps.keys())
        return []
        
    def suggest_safe_routes(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Suggest safe movement routes based on collected intelligence.
        
        Returns:
            Dictionary of route suggestions
        """
        if not self.pos or not self.grid:
            return {}
            
        # Simple route suggestion - avoid known hazards
        hazards = set(self.get_known_hazards())
        
        # Suggest routes to important locations
        routes = {}
        
        # Route away from hazards (simplified pathfinding)
        safe_directions = []
        for direction in ['up', 'down', 'left', 'right']:
            x, y = self.pos
            direction_map = {
                'up': (x, y-1), 'down': (x, y+1),
                'left': (x-1, y), 'right': (x+1, y)
            }
            next_pos = self.grid._normalize_position(direction_map[direction])
            
            if next_pos not in hazards:
                safe_directions.append(next_pos)
                
        routes['safe_moves'] = safe_directions
        return routes
        
    def step(self, world) -> None:
        """
        Perform one simulation step.
        
        Args:
            world: Simulation world
        """
        if not self.is_alive():
            return
            
        # Cannot act while being carried
        if self.being_carried:
            return
            
        # Passive stamina regeneration
        if self.stamina < self.max_stamina:
            self.restore_stamina(config.STAMINA_REGEN_RATE)
            
        if not self.functional:
            # Try self-repair if damaged
            if self.stamina > config.THIA_REPAIR_STAMINA_COST // 2:
                self.self_repair_attempt()
            else:
                self.rest()  # Conserve energy for repair attempts
        else:
            # Functional - perform reconnaissance and support
            self._functional_behavior(world)
            
    def _functional_behavior(self, world) -> None:
        """
        Behavior when Thia is functional.
        
        Args:
            world: Simulation world
        """
        # Perform reconnaissance
        new_recon = self.perform_reconnaissance(world)
        
        # If coordinated with a predator, stay near them
        if self.coordinated_predator and self.coordinated_predator.is_alive():
            predator_distance = self.distance_to(self.coordinated_predator)
            
            if predator_distance > 5:  # Too far, move closer
                self._move_toward_predator()
            elif predator_distance < 2:  # Too close, give some space
                self._move_away_from_predator()
            else:
                # Good distance, perform reconnaissance in different direction
                self._scout_area()
        else:
            # No coordination, search for friendly predators or scout
            self._find_predator_to_coordinate() or self._scout_area()
            
    def _move_toward_predator(self) -> bool:
        """Move toward coordinated predator."""
        if not self.coordinated_predator or not self.coordinated_predator.pos:
            return False
            
        return self._move_toward(self.coordinated_predator.pos)
        
    def _move_away_from_predator(self) -> bool:
        """Move away from coordinated predator to give space."""
        if not self.coordinated_predator or not self.coordinated_predator.pos or not self.pos:
            return False
            
        # Move in opposite direction
        px, py = self.coordinated_predator.pos
        tx, ty = self.pos
        
        dx = tx - px
        dy = ty - py
        
        # Choose direction to increase distance
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
            
        return self.move_direction(direction)
        
    def _scout_area(self) -> bool:
        """Scout area for reconnaissance."""
        # Move to unexplored areas or areas with potential intel value
        if not self.pos:
            return False
            
        # Simple exploration - move in a patrol pattern
        directions = ['up', 'down', 'left', 'right']
        
        # Prefer directions that haven't been explored recently
        # For now, just random movement for scouting
        direction = random.choice(directions)
        return self.move_direction(direction)
        
    def _find_predator_to_coordinate(self) -> bool:
        """Find a nearby predator to coordinate with."""
        visible_entities = self.get_visible_entities()
        predators = [(pos, agent) for pos, agent in visible_entities 
                    if (agent.__class__.__name__ == 'Predator' and 
                        agent.is_alive() and 
                        getattr(agent, 'role', None) and
                        agent.role.value == 'dek')]  # Prefer Dek
        
        if predators:
            closest_predator = min(predators, key=lambda p: self.distance_to(p[1]))
            predator = closest_predator[1]
            
            # Establish coordination if close enough
            if self.distance_to(predator) <= 3:
                self.coordinate_with_predator(predator)
                return True
            else:
                # Move toward predator
                return self._move_toward(predator.pos)
                
        return False
        
    def _move_toward(self, target_pos: Tuple[int, int]) -> bool:
        """Move toward target position."""
        if not self.pos or not self.grid:
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
        
    def get_state_dict(self) -> dict:
        """Extended state dictionary including Thia-specific data."""
        state = super().get_state_dict()
        state.update({
            "functional": self.functional,
            "repair_progress": self.repair_progress,
            "repair_attempts": self.repair_attempts,
            "being_carried": self.being_carried,
            "carrier_id": self.carrier.id if self.carrier else None,
            "coordinated_predator_id": self.coordinated_predator.id if self.coordinated_predator else None,
            "recon_data_count": len(self.recon_data),
            "shared_intel_keys": list(self.shared_intel.keys())
        })
        return state