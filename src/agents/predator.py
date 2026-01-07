"""
Predator agent implementation for Predator: Badlands simulation.
Implements Yautja-style predators with honor system, clan dynamics, and adaptive behavior.
"""
from typing import Optional, Tuple, List, Dict, Any
import random
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum

from agent import Agent, ActionType
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


class PredatorRole(Enum):
    """Roles within the predator clan hierarchy."""
    DEK = "dek"  # Protagonist
    FATHER = "father"  # Clan elder
    BROTHER = "brother"  # Peer/rival
    CLAN = "clan"  # Other clan members


@dataclass
class Trophy:
    """Represents a trophy/kill record."""
    kill_type: str
    timestamp: int
    honor_value: int
    location: Tuple[int, int]


@dataclass
class BanditArm:
    """Single arm of the multi-armed bandit for action selection."""
    action: ActionType
    estimated_reward: float = 0.0
    times_selected: int = 0
    total_reward: float = 0.0
    
    def update_reward(self, reward: float, learning_rate: float = 0.1) -> None:
        """Update estimated reward using exponential moving average."""
        self.times_selected += 1
        self.total_reward += reward
        self.estimated_reward = ((1 - learning_rate) * self.estimated_reward + 
                                learning_rate * reward)
    
    def get_ucb_value(self, total_selections: int, c: float = 2.0) -> float:
        """Calculate Upper Confidence Bound value for action selection."""
        if self.times_selected == 0:
            return float('inf')  # Explore unselected actions first
        
        exploitation = self.estimated_reward
        exploration = c * np.sqrt(np.log(total_selections) / self.times_selected)
        return exploitation + exploration


class Predator(Agent):
    """
    Predator agent with clan hierarchy, honor system, and adaptive behavior.
    Uses multi-armed bandit approach for action selection with online learning.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: PredatorRole,
        pos: Optional[Tuple[int, int]] = None,
        health: int = 100,
        stamina: int = 100,
        attack_power: int = 25,
        vision_range: int = 5
    ):
        """
        Initialize predator agent.
        
        Args:
            agent_id: Unique identifier
            role: Role in clan hierarchy
            pos: Initial position
            health: Health points
            stamina: Stamina points
            attack_power: Base attack damage
            vision_range: Vision range for detecting entities
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
        
        self.role = role
        self.attack_power = attack_power
        
        # Honor and reputation system
        self.honor = 0.0
        self.reputation = 0.0
        self.trophies: List[Trophy] = []
        
        # Carrying mechanics
        self.carrying_thia: Optional[Any] = None
        self.carry_penalty = config.CARRY_STAMINA_MULTIPLIER
        
        # Adaptive behavior - Multi-armed bandit for action selection
        self.bandit_arms = {
            ActionType.HUNT: BanditArm(ActionType.HUNT),
            ActionType.SEEK_THIA: BanditArm(ActionType.SEEK_THIA),
            ActionType.SEEK_BOSS: BanditArm(ActionType.SEEK_BOSS),
            ActionType.REST: BanditArm(ActionType.REST),
            ActionType.FLEE: BanditArm(ActionType.FLEE),
        }
        
        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.total_action_selections = 0
        
        # Strategy parameters
        self.aggression = 0.7 if role == PredatorRole.DEK else 0.5
        self.risk_tolerance = 0.6 if role == PredatorRole.DEK else 0.3
        
        # State tracking
        self.last_honor = self.honor
        self.steps_since_action_reward = 0
        self.current_strategy = ActionType.HUNT
        
        # Clan relationship tracking
        self.clan_relationships: Dict[str, float] = {}  # agent_id -> relationship_value
        self.challenge_cooldown = 0
        
        # Boss tracking
        self.boss_last_seen: Optional[Tuple[int, int]] = None
        self.boss_damage_dealt = 0
        
    def get_move_cost(self) -> int:
        """Calculate movement cost including carry penalty."""
        base_cost = self.move_cost
        if self.carrying_thia:
            base_cost = int(base_cost * self.carry_penalty)
        return base_cost
        
    def can_carry_thia(self) -> bool:
        """Check if can carry Thia (not already carrying)."""
        return self.carrying_thia is None and self.is_alive()
        
    def pick_up_thia(self, thia: Any) -> bool:
        """
        Pick up Thia for carrying/repair.
        
        Args:
            thia: Thia agent to carry
            
        Returns:
            True if successful
        """
        if not self.can_carry_thia() or not thia.pos:
            return False
            
        # Must be adjacent to pick up
        if self.distance_to(thia) > 1.5:
            return False
            
        self.carrying_thia = thia
        thia.being_carried = True
        thia.carrier = self
        
        # Remove Thia from grid while being carried
        if thia.pos and self.grid:
            self.grid.remove(thia.pos)
            thia.pos = None
            
        logger.info(f"{self.id} picked up {thia.id}")
        return True
        
    def drop_thia(self, pos: Optional[Tuple[int, int]] = None) -> bool:
        """
        Drop carried Thia at specified position.
        
        Args:
            pos: Position to drop at (default: current position)
            
        Returns:
            True if successful
        """
        if not self.carrying_thia:
            return False
            
        drop_pos = pos or self.pos
        if not drop_pos or not self.grid:
            return False
            
        # Find nearby empty position if specified position is occupied
        if self.grid.get(drop_pos) is not None:
            adjacent = self.grid.adjacent_positions(drop_pos)
            empty_adjacent = [p for p in adjacent if self.grid.get(p) is None]
            if empty_adjacent:
                drop_pos = random.choice(empty_adjacent)
            else:
                return False  # No space to drop
                
        # Place Thia back on grid
        if self.grid.place(self.carrying_thia, drop_pos):
            self.carrying_thia.being_carried = False
            self.carrying_thia.carrier = None
            self.carrying_thia = None
            logger.info(f"{self.id} dropped Thia at {drop_pos}")
            return True
            
        return False
        
    def repair_thia(self) -> bool:
        """
        Attempt to repair carried Thia.
        
        Returns:
            True if repair was performed
        """
        if not self.carrying_thia:
            return False
            
        # Check if have enough stamina for repair
        repair_cost = config.THIA_REPAIR_STAMINA_COST
        if not self.spend_stamina(repair_cost):
            return False
            
        # Perform repair
        repair_success = self.carrying_thia.receive_repair()
        
        if repair_success:
            self.honor += config.THIA_REPAIR_HONOR
            logger.info(f"{self.id} successfully repaired {self.carrying_thia.id}")
        
        self.last_action = ActionType.REPAIR
        self.action_history.append(ActionType.REPAIR)
        
        return repair_success
        
    def add_trophy(self, kill_type: str, timestamp: int, honor_value: int) -> None:
        """Add a trophy from a kill."""
        if self.pos:
            trophy = Trophy(
                kill_type=kill_type,
                timestamp=timestamp,
                honor_value=honor_value,
                location=self.pos
            )
            self.trophies.append(trophy)
            self.honor += honor_value
            logger.info(f"{self.id} earned {honor_value} honor from {kill_type} kill")
            
    def challenge_clan_member(self, target: 'Predator') -> bool:
        """
        Challenge another clan member (honor duel).
        
        Args:
            target: Predator to challenge
            
        Returns:
            True if challenge was issued
        """
        if (self.challenge_cooldown > 0 or 
            not self.is_alive() or 
            not target.is_alive() or
            target.role == PredatorRole.DEK):
            return False
            
        # Only Dek can challenge father, brothers can challenge each other
        if self.role == PredatorRole.DEK and target.role == PredatorRole.FATHER:
            # Epic challenge - higher stakes
            self._honor_duel(target, high_stakes=True)
        elif (self.role == PredatorRole.BROTHER and 
              target.role == PredatorRole.BROTHER):
            # Peer challenge
            self._honor_duel(target, high_stakes=False)
        else:
            return False
            
        self.challenge_cooldown = 10  # Cooldown period
        return True
        
    def _honor_duel(self, opponent: 'Predator', high_stakes: bool = False) -> None:
        """
        Conduct honor duel with another predator.
        
        Args:
            opponent: Opponent predator
            high_stakes: Whether this is a high-stakes duel
        """
        # Simple duel mechanics based on honor + randomness
        my_strength = self.honor + self.health + random.uniform(0, 20)
        opponent_strength = opponent.honor + opponent.health + random.uniform(0, 20)
        
        honor_gain = config.CLAN_CHALLENGE_HONOR * (2 if high_stakes else 1)
        
        if my_strength > opponent_strength:
            # Victory
            self.honor += honor_gain
            opponent.honor -= honor_gain // 2
            
            # Relationship impact
            self.clan_relationships[opponent.id] = self.clan_relationships.get(opponent.id, 0) + 0.2
            
            logger.info(f"{self.id} defeated {opponent.id} in honor duel (+{honor_gain} honor)")
        else:
            # Defeat
            self.honor -= honor_gain // 2
            opponent.honor += honor_gain
            
            # Relationship impact
            self.clan_relationships[opponent.id] = self.clan_relationships.get(opponent.id, 0) - 0.2
            
            logger.info(f"{self.id} lost to {opponent.id} in honor duel (-{honor_gain//2} honor)")
            
    def select_action(self, available_actions: List[ActionType]) -> ActionType:
        """
        Select action using epsilon-greedy multi-armed bandit.
        
        Args:
            available_actions: List of currently available actions
            
        Returns:
            Selected action
        """
        self.total_action_selections += 1
        
        # Filter bandit arms to only available actions
        available_arms = {action: arm for action, arm in self.bandit_arms.items() 
                         if action in available_actions}
        
        if not available_arms:
            return random.choice(available_actions)
            
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Exploration: random action
            selected_action = random.choice(list(available_arms.keys()))
        else:
            # Exploitation: best UCB action
            best_action = max(available_arms.keys(), 
                            key=lambda a: available_arms[a].get_ucb_value(self.total_action_selections))
            selected_action = best_action
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update selection count
        self.bandit_arms[selected_action].times_selected += 1
        
        return selected_action
        
    def update_action_reward(self, action: ActionType, reward: float) -> None:
        """
        Update reward estimate for an action.
        
        Args:
            action: Action that was taken
            reward: Reward received
        """
        if action in self.bandit_arms:
            self.bandit_arms[action].update_reward(reward, config.BANDIT_LEARNING_RATE)
            logger.debug(f"{self.id} updated {action.value} reward: {reward:.2f}")
            
    def calculate_current_reward(self) -> float:
        """
        Calculate reward based on current state changes.
        
        Returns:
            Reward value for learning
        """
        reward = 0.0
        
        # Honor change reward
        honor_change = self.honor - self.last_honor
        reward += honor_change * 0.1  # Scale honor reward
        
        # Survival bonus
        if self.is_alive():
            reward += 1.0
            
        # Health penalty if low
        if self.health < 30:
            reward -= 2.0
            
        # Stamina management
        if self.stamina < 20:
            reward -= 1.0
            
        # Carrying Thia bonus (if functional)
        if self.carrying_thia and self.carrying_thia.functional:
            reward += 2.0
            
        # Boss damage bonus
        if hasattr(self, '_temp_boss_damage'):
            reward += self._temp_boss_damage * 0.2
            delattr(self, '_temp_boss_damage')
            
        return reward
        
    def get_available_actions(self, world) -> List[ActionType]:
        """
        Determine which actions are currently available.
        
        Args:
            world: Simulation world
            
        Returns:
            List of available actions
        """
        actions = []
        
        # Always available if alive
        if self.is_alive():
            actions.append(ActionType.REST)
            
            # Movement-based actions
            if self.can_move():
                actions.extend([ActionType.HUNT, ActionType.SEEK_BOSS])
                
                # Seek Thia only if not carrying one
                if not self.carrying_thia:
                    actions.append(ActionType.SEEK_THIA)
                    
            # Flee if health is low
            if self.health < 40:
                actions.append(ActionType.FLEE)
                
        return actions
        
    def execute_action(self, action: ActionType, world) -> bool:
        """
        Execute the selected action.
        
        Args:
            action: Action to execute
            world: Simulation world
            
        Returns:
            True if action was successful
        """
        if action == ActionType.HUNT:
            return self._hunt_monsters(world)
        elif action == ActionType.SEEK_THIA:
            return self._seek_thia(world)
        elif action == ActionType.SEEK_BOSS:
            return self._seek_boss(world)
        elif action == ActionType.REST:
            self.rest()
            return True
        elif action == ActionType.FLEE:
            return self._flee(world)
        elif action == ActionType.REPAIR and self.carrying_thia:
            return self.repair_thia()
            
        return False
        
    def _hunt_monsters(self, world) -> bool:
        """Hunt nearby monsters for trophies and honor."""
        visible_entities = self.get_visible_entities()
        monsters = [(pos, agent) for pos, agent in visible_entities 
                   if agent.__class__.__name__ == 'Monster' and agent.is_alive()]
        
        if not monsters:
            # No monsters visible, move randomly to search
            return self._random_move()
            
        # Find closest monster
        closest_monster = min(monsters, key=lambda m: self.distance_to(m[1]))
        monster_pos, monster = closest_monster
        
        # If adjacent, attack
        if self.distance_to(monster) <= 1.5:
            damage = self.attack(monster, self.attack_power)
            if damage > 0 and not monster.is_alive():
                # Monster killed, add trophy
                self.add_trophy("Monster", world.step_count, config.MONSTER_KILL_HONOR)
            return damage > 0
        else:
            # Move toward monster
            return self._move_toward(monster.pos)
            
    def _seek_thia(self, world) -> bool:
        """Seek and potentially pick up/repair Thia."""
        if self.carrying_thia:
            # Already carrying, try to repair
            if not self.carrying_thia.functional:
                return self.repair_thia()
            return True
            
        # Look for Thia
        visible_entities = self.get_visible_entities()
        thias = [(pos, agent) for pos, agent in visible_entities 
                if agent.__class__.__name__ == 'Thia']
        
        if not thias:
            # No Thia visible, search
            return self._random_move()
            
        # Find closest Thia
        closest_thia = min(thias, key=lambda t: self.distance_to(t[1]))
        thia_pos, thia = closest_thia
        
        # If adjacent, pick up
        if self.distance_to(thia) <= 1.5:
            return self.pick_up_thia(thia)
        else:
            # Move toward Thia
            return self._move_toward(thia.pos)
            
    def _seek_boss(self, world) -> bool:
        """Seek and attack the boss."""
        visible_entities = self.get_visible_entities()
        bosses = [(pos, agent) for pos, agent in visible_entities 
                 if agent.__class__.__name__ == 'Boss' and agent.is_alive()]
        
        if bosses:
            # Boss visible
            boss_pos, boss = bosses[0]
            self.boss_last_seen = boss.pos
            
            # If adjacent, attack
            if self.distance_to(boss) <= 1.5:
                damage = self.attack(boss, self.attack_power)
                if damage > 0:
                    self.boss_damage_dealt += damage
                    self.honor += damage * config.BOSS_DAMAGE_HONOR
                    # Store for reward calculation
                    self._temp_boss_damage = damage
                return damage > 0
            else:
                # Move toward boss
                return self._move_toward(boss.pos)
        elif self.boss_last_seen:
            # Move toward last known position
            return self._move_toward(self.boss_last_seen)
        else:
            # Search for boss
            return self._random_move()
            
    def _flee(self, world) -> bool:
        """Flee from dangerous entities."""
        visible_entities = self.get_visible_entities()
        threats = [(pos, agent) for pos, agent in visible_entities 
                  if (agent.__class__.__name__ in ['Boss', 'Monster'] and 
                      agent.is_alive() and self.distance_to(agent) <= 3)]
        
        if not threats:
            # No immediate threats, just rest
            self.rest()
            return True
            
        # Find direction away from threats
        if self.pos:
            avg_threat_x = sum(pos[0] for pos, _ in threats) / len(threats)
            avg_threat_y = sum(pos[1] for pos, _ in threats) / len(threats)
            
            # Move away from average threat position
            dx = self.pos[0] - avg_threat_x
            dy = self.pos[1] - avg_threat_y
            
            # Normalize and move
            if abs(dx) > abs(dy):
                direction = 'right' if dx > 0 else 'left'
            else:
                direction = 'down' if dy > 0 else 'up'
                
            return self.move_direction(direction)
            
        return False
        
    def _move_toward(self, target_pos: Tuple[int, int]) -> bool:
        """Move toward target position."""
        if not self.pos or not self.grid:
            return False
            
        x, y = self.pos
        tx, ty = target_pos
        
        # Simple pathfinding: move in direction of largest difference
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
        return self.move_direction(random.choice(directions))
        
    def step(self, world) -> None:
        """
        Perform one simulation step using adaptive action selection.
        
        Args:
            world: Simulation world
        """
        if not self.is_alive():
            return
            
        # Update carried Thia position
        if self.carrying_thia and self.pos:
            self.carrying_thia.pos = self.pos
            
        # Store current honor for reward calculation
        self.last_honor = self.honor
        
        # Passive stamina regeneration
        if self.stamina < self.max_stamina:
            self.restore_stamina(config.STAMINA_REGEN_RATE)
            
        # Get available actions and select one
        available_actions = self.get_available_actions(world)
        if not available_actions:
            return
            
        selected_action = self.select_action(available_actions)
        
        # Execute action
        action_success = self.execute_action(selected_action, world)
        
        # Calculate reward and update bandit
        reward = self.calculate_current_reward()
        
        # Add success/failure component to reward
        if action_success:
            reward += 0.5
        else:
            reward -= 0.5
            
        self.update_action_reward(selected_action, reward)
        
        # Update challenge cooldown
        if self.challenge_cooldown > 0:
            self.challenge_cooldown -= 1
            
        logger.debug(f"{self.id} step: action={selected_action.value}, "
                    f"reward={reward:.2f}, honor={self.honor:.1f}")
                    
    def get_state_dict(self) -> dict:
        """Extended state dictionary including predator-specific data."""
        state = super().get_state_dict()
        state.update({
            "role": self.role.value,
            "honor": self.honor,
            "reputation": self.reputation,
            "trophies_count": len(self.trophies),
            "carrying_thia": self.carrying_thia.id if self.carrying_thia else None,
            "boss_damage_dealt": self.boss_damage_dealt,
            "bandit_stats": {
                action.value: {
                    "estimated_reward": arm.estimated_reward,
                    "times_selected": arm.times_selected
                } for action, arm in self.bandit_arms.items()
            },
            "epsilon": self.epsilon,
            "aggression": self.aggression
        })
        return state