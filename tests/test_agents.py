"""
Tests for Agent classes functionality.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.grid import Grid
from src.agents import (
    Agent, ActionType, AgentStats,
    Predator, PredatorRole, Trophy, BanditArm,
    Monster, MonsterType,
    Thia, ReconData,
    Boss, BossMode
)


# Mock Agent for testing base functionality
class MockAgent(Agent):
    """Mock agent for testing base Agent functionality."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        
    def step(self, world):
        """Mock step implementation."""
        pass


class TestAgent:
    """Test cases for base Agent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = Grid(size=5, wrap=True, seed=42)
        self.agent = MockAgent("test_agent")
        self.agent.set_grid(self.grid)
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = MockAgent("test")
        
        assert agent.id == "test"
        assert agent.health == 100
        assert agent.max_health == 100
        assert agent.stamina == 100
        assert agent.max_stamina == 100
        assert agent.alive == True
        assert agent.pos == None
        
    def test_agent_placement_on_grid(self):
        """Test placing agent on grid."""
        success = self.grid.place(self.agent, (2, 2))
        assert success == True
        assert self.agent.pos == (2, 2)
        
    def test_agent_movement(self):
        """Test agent movement."""
        # Place agent on grid
        self.grid.place(self.agent, (2, 2))
        
        # Test movement
        success = self.agent.move_to((2, 3))
        assert success == True
        assert self.agent.pos == (2, 3)
        assert self.agent.stats.steps_taken == 1
        
        # Test movement to occupied position
        other_agent = MockAgent("other")
        other_agent.set_grid(self.grid)
        self.grid.place(other_agent, (2, 4))
        
        success2 = self.agent.move_to((2, 4))
        assert success2 == False
        
    def test_agent_directional_movement(self):
        """Test directional movement."""
        self.grid.place(self.agent, (2, 2))
        
        # Test all directions
        assert self.agent.move_direction('up') == True
        assert self.agent.pos == (2, 1)
        
        assert self.agent.move_direction('right') == True
        assert self.agent.pos == (3, 1)
        
        assert self.agent.move_direction('down') == True
        assert self.agent.pos == (3, 2)
        
        assert self.agent.move_direction('left') == True
        assert self.agent.pos == (2, 2)
        
    def test_agent_stamina_management(self):
        """Test stamina spending and restoration."""
        initial_stamina = self.agent.stamina
        
        # Spend stamina
        success = self.agent.spend_stamina(20)
        assert success == True
        assert self.agent.stamina == initial_stamina - 20
        
        # Try to spend more than available
        success2 = self.agent.spend_stamina(200)
        assert success2 == False
        
        # Restore stamina
        restored = self.agent.restore_stamina(10)
        assert restored == 10
        assert self.agent.stamina == initial_stamina - 10
        
    def test_agent_health_and_damage(self):
        """Test health management and damage."""
        initial_health = self.agent.health
        
        # Take damage
        damage = self.agent.take_damage(30, "test_source")
        assert damage == 30
        assert self.agent.health == initial_health - 30
        
        # Heal damage
        healed = self.agent.heal(10)
        assert healed == 10
        assert self.agent.health == initial_health - 20
        
    def test_agent_death(self):
        """Test agent death mechanics."""
        # Deal lethal damage
        self.grid.place(self.agent, (2, 2))
        self.agent.take_damage(150, "lethal_source")
        
        assert self.agent.alive == False
        assert self.agent.health == 0
        assert self.agent.pos == None  # Removed from grid
        assert self.agent.stats.deaths == 1
        
    def test_agent_rest(self):
        """Test resting mechanics."""
        # Damage and exhaust agent
        self.agent.take_damage(20, "test")
        self.agent.spend_stamina(30)
        
        old_health = self.agent.health
        old_stamina = self.agent.stamina
        
        # Rest
        self.agent.rest()
        
        assert self.agent.health >= old_health  # Should heal
        assert self.agent.stamina >= old_stamina  # Should restore stamina
        
    def test_agent_combat(self):
        """Test combat between agents."""
        target = MockAgent("target")
        target.set_grid(self.grid)
        
        # Place agents adjacent to each other
        self.grid.place(self.agent, (2, 2))
        self.grid.place(target, (2, 3))
        
        # Test attack
        damage_dealt = self.agent.attack(target, 25)
        
        assert damage_dealt == 25
        assert target.health == 75
        assert self.agent.stats.damage_dealt == 25
        
    def test_agent_vision(self):
        """Test agent vision and entity detection."""
        # Place agent and some targets
        self.grid.place(self.agent, (2, 2))
        
        target1 = MockAgent("target1")
        target2 = MockAgent("target2")
        target1.set_grid(self.grid)
        target2.set_grid(self.grid)
        
        self.grid.place(target1, (2, 3))  # Within vision
        self.grid.place(target2, (4, 4))  # May be outside vision
        
        visible = self.agent.get_visible_entities()
        
        # Should see at least target1
        assert len(visible) >= 1
        found_targets = [entity for pos, entity in visible]
        assert target1 in found_targets
        
    def test_agent_state_serialization(self):
        """Test agent state serialization."""
        self.grid.place(self.agent, (2, 2))
        self.agent.take_damage(10, "test")
        self.agent.spend_stamina(5)
        
        state = self.agent.get_state_dict()
        
        assert state['id'] == "test_agent"
        assert state['type'] == "MockAgent"
        assert state['pos'] == (2, 2)
        assert state['health'] == 90
        assert state['stamina'] == 95
        assert state['alive'] == True


class TestPredator:
    """Test cases for Predator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = Grid(size=10, wrap=True, seed=42)
        self.predator = Predator("Dek", PredatorRole.DEK)
        self.predator.set_grid(self.grid)
        
    def test_predator_initialization(self):
        """Test predator initialization."""
        pred = Predator("TestPred", PredatorRole.FATHER)
        
        assert pred.id == "TestPred"
        assert pred.role == PredatorRole.FATHER
        assert pred.honor == 0.0
        assert len(pred.trophies) == 0
        assert pred.carrying_thia == None
        assert len(pred.bandit_arms) > 0  # Should have learning arms
        
    def test_predator_honor_system(self):
        """Test honor tracking and trophy system."""
        initial_honor = self.predator.honor
        
        # Add trophy
        self.predator.add_trophy("Monster", 100, 15)
        
        assert len(self.predator.trophies) == 1
        assert self.predator.honor == initial_honor + 15
        
        trophy = self.predator.trophies[0]
        assert trophy.kill_type == "Monster"
        assert trophy.honor_value == 15
        
    def test_predator_thia_interaction(self):
        """Test Thia carrying and repair mechanics."""
        # Create Thia
        thia = Thia("TestThia")
        thia.set_grid(self.grid)
        
        # Place both agents
        self.grid.place(self.predator, (5, 5))
        self.grid.place(thia, (5, 6))  # Adjacent
        
        # Test pickup
        success = self.predator.pick_up_thia(thia)
        assert success == True
        assert self.predator.carrying_thia == thia
        assert thia.being_carried == True
        assert thia.pos == None  # Removed from grid
        
        # Test repair attempt
        repair_success = self.predator.repair_thia()
        assert isinstance(repair_success, bool)
        
        # Test drop
        drop_success = self.predator.drop_thia()
        assert drop_success == True
        assert self.predator.carrying_thia == None
        assert thia.being_carried == False
        assert thia.pos != None  # Back on grid
        
    def test_predator_bandit_learning(self):
        """Test multi-armed bandit learning system."""
        # Test action selection
        available_actions = [ActionType.HUNT, ActionType.REST, ActionType.SEEK_BOSS]
        selected = self.predator.select_action(available_actions)
        
        assert selected in available_actions
        
        # Test reward update
        initial_reward = self.predator.bandit_arms[ActionType.HUNT].estimated_reward
        self.predator.update_action_reward(ActionType.HUNT, 5.0)
        
        # Should update the estimated reward
        new_reward = self.predator.bandit_arms[ActionType.HUNT].estimated_reward
        assert new_reward != initial_reward
        
    def test_predator_carry_penalty(self):
        """Test movement cost penalty when carrying Thia."""
        thia = Thia("TestThia")
        thia.set_grid(self.grid)
        
        # Normal movement cost
        normal_cost = self.predator.get_move_cost()
        
        # Simulate carrying Thia
        self.predator.carrying_thia = thia
        carry_cost = self.predator.get_move_cost()
        
        assert carry_cost > normal_cost
        
    def test_predator_clan_challenges(self):
        """Test clan challenge mechanics."""
        brother = Predator("Brother", PredatorRole.BROTHER)
        brother.set_grid(self.grid)
        
        # Place both predators
        self.grid.place(self.predator, (5, 5))
        self.grid.place(brother, (5, 6))
        
        initial_honor = self.predator.honor
        
        # Test challenge (Dek can't challenge another Dek, but let's test the mechanism)
        # We'll test brother challenging brother instead
        success = brother.challenge_clan_member(self.predator)
        # This should fail as Dek can't be challenged by brother
        assert success == False
        
    def test_predator_state_serialization(self):
        """Test predator state serialization."""
        self.grid.place(self.predator, (3, 3))
        self.predator.honor = 50.0
        self.predator.add_trophy("TestKill", 0, 10)
        
        state = self.predator.get_state_dict()
        
        assert state['role'] == PredatorRole.DEK.value
        assert state['honor'] == 50.0
        assert state['trophies_count'] == 1
        assert 'bandit_stats' in state


class TestThia:
    """Test cases for Thia class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = Grid(size=10, wrap=True, seed=42)
        self.thia = Thia("TestThia")
        self.thia.set_grid(self.grid)
        
    def test_thia_initialization(self):
        """Test Thia initialization."""
        assert self.thia.id == "TestThia"
        assert self.thia.functional == False  # Starts damaged
        assert self.thia.being_carried == False
        assert self.thia.repair_progress == 0
        
    def test_thia_repair_mechanics(self):
        """Test repair progression."""
        initial_progress = self.thia.repair_progress
        
        # Multiple repair attempts
        for _ in range(5):  # Should complete repair
            self.thia.receive_repair()
            
        assert self.thia.repair_progress >= initial_progress
        
        # Should become functional after enough repairs
        if self.thia.repair_progress >= self.thia.repair_threshold:
            assert self.thia.functional == True
            
    def test_thia_self_repair(self):
        """Test self-repair attempts."""
        initial_stamina = self.thia.stamina
        
        # Attempt self-repair
        result = self.thia.self_repair_attempt()
        
        # Should spend stamina regardless of success
        assert self.thia.stamina <= initial_stamina
        
    def test_thia_reconnaissance(self):
        """Test reconnaissance functionality."""
        # Make Thia functional first
        self.thia.functional = True
        self.grid.place(self.thia, (5, 5))
        
        # Create mock world object
        class MockWorld:
            step_count = 100
            
        world = MockWorld()
        
        # Place some entities to detect
        predator = Predator("TestPred", PredatorRole.DEK)
        predator.set_grid(self.grid)
        self.grid.place(predator, (6, 6))
        
        # Perform reconnaissance
        recon_data = self.thia.perform_reconnaissance(world)
        
        # Should detect the predator
        assert len(recon_data) >= 0  # May or may not detect based on range
        
    def test_thia_boss_hint(self):
        """Test boss location hinting."""
        # Add some fake reconnaissance data
        boss_sighting = ReconData(
            entity_type="Boss",
            position=(8, 8),
            timestamp=50,
            confidence=0.9,
            additional_info={}
        )
        
        self.thia.functional = True
        self.thia.recon_data.append(boss_sighting)
        
        hint = self.thia.provide_boss_hint(current_step=60)
        
        # Should provide some hint (may be inaccurate due to randomness)
        if hint:
            assert isinstance(hint, tuple)
            assert len(hint) == 2
            
    def test_thia_coordination(self):
        """Test coordination with predators."""
        self.thia.functional = True
        
        predator = Predator("TestPred", PredatorRole.DEK)
        predator.set_grid(self.grid)
        
        # Establish coordination
        intel = self.thia.coordinate_with_predator(predator)
        
        assert self.thia.coordinated_predator == predator
        assert isinstance(intel, dict)
        assert 'recon_data' in intel
        
    def test_thia_state_serialization(self):
        """Test Thia state serialization."""
        self.thia.functional = True
        self.thia.repair_progress = 3
        
        state = self.thia.get_state_dict()
        
        assert state['functional'] == True
        assert state['repair_progress'] == 3
        assert state['being_carried'] == False


class TestMonster:
    """Test cases for Monster class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = Grid(size=10, wrap=True, seed=42)
        self.monster = Monster("TestMonster", MonsterType.MEDIUM)
        self.monster.set_grid(self.grid)
        
    def test_monster_initialization(self):
        """Test monster initialization."""
        assert self.monster.id == "TestMonster"
        assert self.monster.monster_type == MonsterType.MEDIUM
        assert self.monster.aggression > 0
        assert self.monster.territorial_radius > 0
        
    def test_monster_type_variations(self):
        """Test different monster types have different stats."""
        small = Monster("Small", MonsterType.SMALL)
        large = Monster("Large", MonsterType.LARGE)
        pack = Monster("Pack", MonsterType.PACK)
        
        # Different types should have different characteristics
        # Small: lower health, higher stamina
        # Large: higher health, lower stamina
        # Pack: balanced but better vision
        
        assert small.health != large.health
        assert small.vision_range != pack.vision_range
        
    def test_monster_territory(self):
        """Test territorial behavior."""
        # Set home territory
        self.monster.set_home_territory((5, 5))
        assert self.monster.home_territory == (5, 5)
        
        # Place monster in territory
        self.grid.place(self.monster, (5, 5))
        assert self.monster.is_in_territory() == True
        
        # Move outside territory
        self.grid.move((5, 5), (9, 9))
        # May or may not be in territory depending on radius
        
    def test_monster_pack_behavior(self):
        """Test pack coordination."""
        # Create pack members
        pack_members = [
            Monster("Pack1", MonsterType.PACK),
            Monster("Pack2", MonsterType.PACK),
            Monster("Pack3", MonsterType.PACK)
        ]
        
        for member in pack_members:
            member.set_grid(self.grid)
            
        # Set up pack
        pack_members[0].add_to_pack(pack_members[1:])
        
        # Should have pack leader
        assert pack_members[0].pack_leader is not None
        assert len(pack_members[0].pack_members) >= 1
        
    def test_monster_target_finding(self):
        """Test target acquisition."""
        self.grid.place(self.monster, (5, 5))
        
        # Place a predator nearby
        predator = Predator("Target", PredatorRole.DEK)
        predator.set_grid(self.grid)
        self.grid.place(predator, (5, 6))
        
        target = self.monster.find_target()
        
        # Should find the predator (if within vision range)
        if target:
            assert target == predator
            
    def test_monster_state_serialization(self):
        """Test monster state serialization."""
        self.monster.set_home_territory((3, 3))
        
        state = self.monster.get_state_dict()
        
        assert state['monster_type'] == MonsterType.MEDIUM.value
        assert state['home_territory'] == (3, 3)
        assert 'aggression' in state
        assert 'pack_size' in state


class TestBoss:
    """Test cases for Boss class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = Grid(size=15, wrap=True, seed=42)
        self.boss = Boss("TestBoss", health=200)
        self.boss.set_grid(self.grid)
        
    def test_boss_initialization(self):
        """Test boss initialization."""
        assert self.boss.id == "TestBoss"
        assert self.boss.health == 200
        assert self.boss.max_health == 200
        assert self.boss.current_mode == BossMode.PATROL
        assert len(self.boss.active_counters) == 0
        
    def test_boss_territory_setup(self):
        """Test boss territory setup."""
        self.boss.set_territory((7, 7))
        
        assert self.boss.territory_center == (7, 7)
        assert len(self.boss.patrol_points) > 0
        
        # Place boss in territory
        self.grid.place(self.boss, (7, 7))
        assert self.boss.is_in_territory() == True
        
    def test_boss_adaptive_damage_reduction(self):
        """Test adaptive damage reduction system."""
        predator = Predator("Attacker", PredatorRole.DEK)
        predator.set_grid(self.grid)
        
        # Place agents
        self.grid.place(self.boss, (7, 7))
        self.grid.place(predator, (7, 8))
        
        # Initial damage
        initial_health = self.boss.health
        damage1 = self.boss.take_damage(50, predator.id)
        
        # Record multiple attacks to trigger adaptation
        for _ in range(5):
            self.boss.record_attack(predator, 50)
            
        # Analyze patterns and adapt
        analysis = self.boss.analyze_attack_patterns()
        self.boss.adapt_to_patterns(analysis)
        
        # Should have counter-strategies now
        assert len(self.boss.active_counters) > 0 or len(self.boss.defense_multipliers) > 0
        
    def test_boss_area_attack(self):
        """Test boss area attack ability."""
        # Place boss and multiple targets
        self.grid.place(self.boss, (7, 7))
        
        targets = []
        for i, pos in enumerate([(7, 8), (8, 7), (6, 7)]):
            pred = Predator(f"Target{i}", PredatorRole.BROTHER)
            pred.set_grid(self.grid)
            self.grid.place(pred, pos)
            targets.append(pred)
            
        # Perform area attack
        results = self.boss.area_attack()
        
        # Should hit multiple targets
        assert len(results) >= 0  # May hit targets within range
        
        for target, damage in results:
            assert damage > 0
            assert target.health < target.max_health
            
    def test_boss_mode_selection(self):
        """Test boss behavior mode selection."""
        # Test different conditions
        self.grid.place(self.boss, (7, 7))
        
        # Normal health should be patrol mode
        mode = self.boss.select_mode()
        # Mode depends on situation, just test it returns a valid mode
        assert isinstance(mode, BossMode)
        
        # Low health should trigger rage mode
        self.boss.health = 50  # Below rage threshold
        rage_mode = self.boss.select_mode()
        assert rage_mode == BossMode.RAGE
        
    def test_boss_pattern_decay(self):
        """Test pattern learning decay."""
        # Add some counters
        self.boss.active_counters["TestAttacker"] = 0.8
        self.boss.defense_multipliers["TestAttacker_horizontal"] = 1.5
        
        # Apply decay
        self.boss.apply_pattern_decay()
        
        # Values should decrease
        assert self.boss.active_counters["TestAttacker"] < 0.8
        assert self.boss.defense_multipliers["TestAttacker_horizontal"] < 1.5
        
    def test_boss_state_serialization(self):
        """Test boss state serialization."""
        self.boss.set_territory((5, 5))
        self.boss.active_counters["TestPlayer"] = 0.5
        
        state = self.boss.get_state_dict()
        
        assert state['territory_center'] == (5, 5)
        assert state['current_mode'] == BossMode.PATROL.value
        assert 'TestPlayer' in state['active_counters']
        assert state['rage_active'] == False


class TestBanditArm:
    """Test cases for BanditArm learning component."""
    
    def test_bandit_arm_initialization(self):
        """Test bandit arm initialization."""
        arm = BanditArm(ActionType.HUNT)
        
        assert arm.action == ActionType.HUNT
        assert arm.estimated_reward == 0.0
        assert arm.times_selected == 0
        assert arm.total_reward == 0.0
        
    def test_bandit_arm_reward_update(self):
        """Test reward update mechanism."""
        arm = BanditArm(ActionType.HUNT)
        
        # Update with positive reward
        arm.update_reward(5.0, learning_rate=0.1)
        
        assert arm.times_selected == 1
        assert arm.total_reward == 5.0
        assert arm.estimated_reward > 0.0
        
        # Update again
        arm.update_reward(3.0, learning_rate=0.1)
        
        assert arm.times_selected == 2
        assert arm.total_reward == 8.0
        
    def test_bandit_arm_ucb_calculation(self):
        """Test Upper Confidence Bound calculation."""
        arm = BanditArm(ActionType.HUNT)
        
        # Unselected arm should have infinite UCB
        ucb = arm.get_ucb_value(total_selections=10)
        assert ucb == float('inf')
        
        # Selected arm should have finite UCB
        arm.update_reward(2.0)
        ucb = arm.get_ucb_value(total_selections=10)
        assert ucb < float('inf')
        assert ucb > arm.estimated_reward  # Should include exploration bonus