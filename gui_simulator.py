import time
import sys,os
import random
import math
from typing import Dict, List, Any, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
random.seed()

from view.gui import Gui, GuiEvent
from visual_ai_simulation import StandaloneVisualSimulation, Agent, AgentType, BehaviorState, SimpleGrid

# some global vars
TOTAL_AGENTS = 0
LAST_UPDATE = 0


class Ocean:
    def __init__(self, width: int = 30, height: int = 30):
        self.width = width
        self.height = height
        self.grid = SimpleGrid(width, height)
    def getRandomPosition(self):
        return self.grid.getRandomPosition()


class Simulator:
    def __init__(self, grid_size: int = 30) -> None:
        self._ocean = Ocean(grid_size, grid_size)
        self._agents: List[Agent] = []
        self._isRunning = False
        self._paused = False
        self._stepCount = 0
        self._fpsTarget = 8
        
        # Define agent colors dictionary as requested
        self.agent_colours = {
            'Predator': 'red',
            'Monster': 'blue', 
            'Thia': 'green',
            'Boss': 'yellow',
            'EvolutionaryMonster': 'purple'
        }
        
        # Initialize GUI first
        self._gui = Gui(
            gridWidth=grid_size,
            gridHeight=grid_size,
            cellSize=18,
            title="Predator: Badlands - Advanced AI Simulation"
        )
        
        # Add initial welcome message to log
        self._gui.addLogMessage("Simulation initialized", important=True)
        
        # Generate population after GUI is ready
        self._generateInitialPopulation()
        
        # Environmental event system for unpredictability
        self._lastEventTime = 0
        self._eventCooldown = random.randint(30, 90)  # Events every 30-90 steps
        self._stormActive = False
        self._stormDuration = 0
        self._resourceScarcity = 1.0  # 1.0 = normal, 0.5 = scarce resources
        
        # Render initial state
        self._render()
        
        print("Simulator initialized with GUI")
        print("Controls: SPACE=Pause, R=Reset, L=Toggle Labels, +/-=Speed, Click=Select, ESC=Exit")
    
    def _generateInitialPopulation(self) -> None:
        """
        Generate the initial population of agents.
        """
        self._agents.clear()
        
        # Create predators
        predator_configs = [
            ("Dek", AgentType.PREDATOR_DEK),
            ("Father", AgentType.PREDATOR_FATHER),
            ("Brother", AgentType.PREDATOR_BROTHER),
            ("Clan1", AgentType.PREDATOR_CLAN),
            ("Clan2", AgentType.PREDATOR_CLAN),
        ]
        
        for name, agent_type in predator_configs:
            pos = self._ocean.getRandomPosition()
            if pos:
                # Randomize stats for variety
                baseHealth = random.randint(100, 140)
                agent = Agent(
                    id=name,
                    agentType=agent_type,
                    pos=pos,
                    health=baseHealth,
                    maxHealth=baseHealth,
                    moveCooldown=random.randint(2, 6)
                )
                agent.aggression = random.uniform(0.3, 0.9)
                agent.curiosity = random.uniform(0.1, 0.8)
                agent.packTendency = random.uniform(0.2, 0.9)
                agent.strength = random.uniform(0.6, 0.8)
                agent.defense = random.uniform(0.5, 0.7)
                agent.attackSpeed = random.uniform(0.4, 0.6)
                
                self._ocean.grid.placeAgent(agent, pos)
                self._agents.append(agent)
        
        # Create special agents
        special_configs = [
            ("Thia", AgentType.THIA),
            ("Boss", AgentType.BOSS),
        ]
        
        for name, agent_type in special_configs:
            pos = self._ocean.getRandomPosition()
            if pos:
                base_health = random.randint(130, 170)
                agent = Agent(
                    id=name,
                    agentType=agent_type,
                    pos=pos,
                    health=base_health,
                    maxHealth=base_health,
                    moveCooldown=random.randint(1, 4)
                )
                # Special agents get unique traits
                agent.aggression = random.uniform(0.4, 0.7)
                agent.intelligence = random.uniform(0.6, 1.0)
                agent.adaptability = random.uniform(0.5, 0.9)
                
                # Combat attributes for special agents
                if name == "Dek":
                    agent.strength = random.uniform(0.85, 0.95)  # Very powerful
                    agent.defense = random.uniform(0.7, 0.8)
                    agent.attack_speed = random.uniform(0.6, 0.8)
                elif name in ["Father", "Brother"]:
                    agent.strength = random.uniform(0.75, 0.85)  # Strong
                    agent.defense = random.uniform(0.6, 0.75)
                    agent.attack_speed = random.uniform(0.5, 0.7)
                elif name == "Thia":
                    agent.strength = random.uniform(0.7, 0.8)   # Good strength
                    agent.defense = random.uniform(0.6, 0.8)
                    agent.attack_speed = random.uniform(0.7, 0.9) # Fast
                elif name == "Boss":
                    agent.strength = random.uniform(0.8, 0.9)   # Very strong
                    agent.defense = random.uniform(0.7, 0.85)
                    agent.attack_speed = random.uniform(0.4, 0.6) # Slow but powerful
                else:
                    agent.strength = random.uniform(0.65, 0.75)  # Clan members
                    agent.defense = random.uniform(0.55, 0.65)
                    agent.attack_speed = random.uniform(0.5, 0.7)
                
                self._ocean.grid.placeAgent(agent, pos)
                self._agents.append(agent)
        
        monster_types = [AgentType.MONSTER_SMALL,AgentType.MONSTER_MEDIUM,AgentType.MONSTER_LARGE,AgentType.MONSTER_PACK]
        
        for i in range(18):
            pos = self._ocean.getRandomPosition()
            if pos:
                monster_type = random.choice(monster_types)
                
                if monster_type == AgentType.MONSTER_SMALL:
                    health = random.randint(40, 80)
                elif monster_type == AgentType.MONSTER_LARGE:health = random.randint(100, 140)
                elif monster_type == AgentType.MONSTER_PACK:health = random.randint(70, 100)
                else:health = random.randint(60, 100)
                
                agent = Agent(
                    id=f"Monster{i+1}",
                    agentType=monster_type,
                    pos=pos,
                    health=health,
                    maxHealth=health,
                    moveCooldown=random.randint(2, 10)
                )
                
                # Add monster personality traits
                agent.aggression = random.uniform(0.1, 0.6)
                agent.fear_level = random.uniform(0.2, 0.8)
                agent.territorial = random.uniform(0.1, 0.7)
                agent.pack_behavior = random.uniform(0.1, 0.9) if monster_type == AgentType.MONSTER_PACK else random.uniform(0.1, 0.3)
                
                # Combat attributes based on monster type
                if monster_type == AgentType.MONSTER_SMALL:
                    agent.strength = random.uniform(0.2, 0.4)    # Weak
                    agent.defense = random.uniform(0.1, 0.3)     # Low defense
                    agent.attack_speed = random.uniform(0.8, 1.0) # Fast attacks
                elif monster_type == AgentType.MONSTER_LARGE:
                    agent.strength = random.uniform(0.7, 0.9)    # Very strong
                    agent.defense = random.uniform(0.6, 0.8)     # High defense
                    agent.attack_speed = random.uniform(0.2, 0.4) # Slow attacks
                elif monster_type == AgentType.MONSTER_PACK:
                    agent.strength = random.uniform(0.4, 0.6)    # Medium strength
                    agent.defense = random.uniform(0.3, 0.5)     # Medium defense
                    agent.attack_speed = random.uniform(0.6, 0.8) # Good attack speed
                else:  # MONSTER_MEDIUM
                    agent.strength = random.uniform(0.5, 0.7)    # Moderate strength
                    agent.defense = random.uniform(0.4, 0.6)     # Moderate defense
                    agent.attack_speed = random.uniform(0.5, 0.7) # Moderate attack speed
                
                self._ocean.grid.placeAgent(agent, pos)
                self._agents.append(agent)
        
        self._gui.addLogMessage(f"Generated initial population: {len(self._agents)} agents")
    
    def run(self) -> None:
        """Run the simulation."""
        self._isRunning = True
        
        self._gui.addLogMessage("Starting Predator: Badlands simulation...", important=True)
        
        while self._isRunning:
            # Handle GUI events
            self._handleGuiEvents()
            
            # Update simulation if not paused
            if not self._paused:
                self._update()
                self._stepCount += 1
            
            # Render current state
            self._render()
            
            # Control frame rate
            self._gui.tick(self._fpsTarget)
            
            # Check if GUI was closed
            if self._gui.isClosed():
                self._isRunning = False
        
        # Cleanup
        self._gui.quit()
        print("👋 Simulation ended")
    
    def _handleGuiEvents(self) -> None:
        """Handle events from the GUI."""
        self._gui.handleEvents()
        events = self._gui.getEvents()
        
        for event in events:
            if event == GuiEvent.QUIT:
                self._isRunning = False
            
            elif event == GuiEvent.PAUSE:
                self._paused = not self._paused
                status = "⏸️ Simulation paused" if self._paused else "▶️ Simulation resumed"
                self._gui.addLogMessage(status)
            
            elif event == GuiEvent.RESET:
                self._resetSimulation()
            
            elif event == GuiEvent.SPEED_UP:
                self._fpsTarget = min(30, self._fpsTarget + 2)
                self._gui.addLogMessage(f"Speed increased: {self._fpsTarget} FPS")
            
            elif event == GuiEvent.SLOW_DOWN:
                self._fpsTarget = max(1, self._fpsTarget - 2)
                self._gui.addLogMessage(f"Speed decreased: {self._fpsTarget} FPS")
            
            elif event == GuiEvent.GRID_CLICKED:
                self._handleGridClick()
    
    def _handleGridClick(self) -> None:
        """Handle a click on the grid to select an agent."""
        clickPos = self._gui.getLastClickPosition()
        if not clickPos:
            return
        
        # Find agent closest to click position
        clickedAgent = None
        minDistance = float('inf')
        
        for agent in self._agents:
            if agent.alive:
                distance = abs(agent.pos[0] - clickPos[0]) + abs(agent.pos[1] - clickPos[1])
                if distance < minDistance:
                    minDistance = distance
                    clickedAgent = agent
        
        if minDistance <= 1:  # Within 1 cell
            self._gui.setSelectedAgent(clickedAgent)
            if clickedAgent:
                # Agent selected - could add logging here if method exists
                pass
    
    def _resetSimulation(self) -> None:
        """Reset the simulation to initial state."""
        self._ocean = Ocean(self._ocean.width, self._ocean.height)
        self._generateInitialPopulation()
        self._stepCount = 0
        self._gui.setSelectedAgent(None)
        self._gui.addLogMessage("Simulation reset", important=True)
    
    def _render(self) -> None:
        """Render the current state of the simulation."""
        # Prepare statistics for the GUI
        aliveAgents = [a for a in self._agents if a.alive]
        
        # Count agent types
        agentTypeCounts = {}
        for agent in aliveAgents:
            typeName = agent.agentType.value.replace('_', ' ').title()
            agentTypeCounts[typeName] = agentTypeCounts.get(typeName, 0) + 1
        
        # Count behaviors
        behaviorCounts = {}
        for agent in aliveAgents:
            behaviorName = agent.behavior.value.title()
            behaviorCounts[behaviorName] = behaviorCounts.get(behaviorName, 0) + 1
        
        simulationStats = {
            'step': self._stepCount,
            'agent_count': len(aliveAgents),  # Fixed: GUI expects agent_count with underscore
            'fps': self._gui.clock.get_fps(),
            'fps_target': self._fpsTarget,    # Fixed: GUI expects fps_target with underscore
            'paused': self._paused,
            'agent_types': agentTypeCounts,   # Fixed: GUI expects agent_types with underscore
            'behaviors': behaviorCounts,
        }
        
        # Render through GUI
        self._gui.render(aliveAgents, simulationStats)
    
    def _update(self) -> None:
        """Update the simulation state."""
        try:
            # Handle environmental events for unpredictability
            self.__handle_environmental_events()
            
            # Update each agent (invoke act method equivalent)
            for agent in self._agents:
                if agent.alive:
                    self.__act_agent(agent)
            
            # Handle agent interactions
            self.__handle_interactions()
            
            # Check if Dek has died (general check for any death cause)
            self.__check_for_dek_death()
            
            # Check for simulation reset conditions - only reset if very few agents remain
            aliveCount = len([a for a in self._agents if a.alive])
            if aliveCount < 3 and not self._paused:  # Only reset when almost no agents left
                self._gui.addLogMessage(f"Only {aliveCount} agents remaining, resetting...", important=True)
                self._resetSimulation()
            
            # Occasionally spawn new agents to maintain population dynamics
            elif aliveCount < 25 and random.random() < 0.02:  # 2% chance per step when population is low
                self.__spawn_replacement_agent()
                
        except Exception as e:
            # Silent error handling for production
            pass
            # Don't pause - let simulation continue
    
    def __act_agent(self, agent: Agent) -> None:
        """
        Make an agent act (equivalent to invoking act method).
        This implements the core AI behavior logic.
        """
        # Every agent makes a decision each step - no cooldown restrictions
        
        # Update behavior based on surroundings
        self.__update_agent_behavior(agent)
        
        # Execute movement
        moved = False
        
        # Try to move toward target
        if agent.targetPos:
            moved = self.__move_agent_toward_target(agent)
            if agent.pos == agent.targetPos:
                agent.targetPos = None
        
        # Individual movement decision each step (no cooldowns)
        if not moved:
            # Each agent has individual movement personality
            curiosity = getattr(agent, 'curiosity', random.uniform(0.2, 0.8))
            energy_level = getattr(agent, 'energy_level', random.uniform(0.3, 0.7))
            restlessness = getattr(agent, 'restlessness', random.uniform(0.1, 0.6))
            
            # Base movement chance varies by agent type and personality
            base_chance = 0.15  # 15% base chance each step
            personality_bonus = (curiosity * 0.2) + (energy_level * 0.15) + (restlessness * 0.25)
            
            # Behavior affects movement probability
            behavior_multiplier = {
                BehaviorState.HUNTING: 1.8,
                BehaviorState.FLEEING: 2.2, 
                BehaviorState.PATROLLING: 1.4,
                BehaviorState.IDLE: 1.0,
                BehaviorState.RESTING: 0.3
            }.get(agent.behavior, 1.0)
            
            movement_chance = (base_chance + personality_bonus) * behavior_multiplier
            movement_chance = min(0.8, movement_chance)  # Cap at 80% per step
            
            if random.random() < movement_chance:
                try:
                    moved = self.__move_agent_randomly(agent)
                    # Log occasional movements to show activity
                    if moved and self._stepCount % 50 == 0:  # Every 50 steps
                        self._gui.addLogMessage(f"{agent.agentType.name} agent moved (step {self._stepCount})")
                except Exception:
                    pass  # Skip movement if error occurs
            
            # Sometimes agents get distracted and change direction
            elif random.random() < (restlessness * 0.3):
                agent.targetPos = None  # Clear current target based on restlessness
        
        # Movement tracking for statistics (optional)
        if moved:
            # Could track movement statistics here if needed
            pass
        
        # Regenerate health/stamina - much slower healing
        if agent.behavior == BehaviorState.RESTING:
            # Only heal 1 HP every 10 steps when resting
            if self._stepCount % 10 == 0:
                agent.health = min(agent.maxHealth, agent.health + 1)
            agent.stamina = min(agent.maxStamina, agent.stamina + 2)
        else:
            # Very slow natural healing - 1 HP every 25 steps when active
            if self._stepCount % 25 == 0:
                agent.health = min(agent.maxHealth, agent.health + 1)
            agent.stamina = min(agent.maxStamina, agent.stamina + 1)
    
    def __update_agent_behavior(self, agent: Agent) -> None:
        """Update agent behavior based on environment, personality, and alliances."""
        neighbors = self._ocean.grid.getNeighbors(agent.pos, radius=2)
        extended_neighbors = self._ocean.grid.getNeighbors(agent.pos, radius=3)
        
        # Analyze neighbors by alliance and type
        enemies_nearby = self.__get_enemies_nearby(agent, neighbors)
        allies_nearby = self.__get_allies_nearby(agent, neighbors)
        neutral_agents = [n for n in neighbors if n not in enemies_nearby and n not in allies_nearby and n != agent]
        
        # Get agent personality traits (with defaults if not set)
        aggression = getattr(agent, 'aggression', random.uniform(0.3, 0.7))
        curiosity = getattr(agent, 'curiosity', random.uniform(0.2, 0.6))
        pack_tendency = getattr(agent, 'pack_tendency', random.uniform(0.3, 0.7))
        
        if 'PREDATOR' in agent.agentType.value.upper() or agent.id.upper() in ['DEK', 'THIA']:
            # Special Priority: Dek-Thia coordination
            dek_thia_partner = self.__get_dek_thia_partner(agent, extended_neighbors)
            if dek_thia_partner:
                partner_distance = abs(dek_thia_partner.pos[0] - agent.pos[0]) + abs(dek_thia_partner.pos[1] - agent.pos[1])
                
                # If partner is too far, move closer (unless in immediate combat)
                if partner_distance > 3 and not enemies_nearby:
                    agent.behavior = BehaviorState.PATROLLING
                    agent.targetPos = dek_thia_partner.pos
                    if random.random() < 0.1:  # Occasional message
                        self._gui.addLogMessage(f"{agent.id} moving to team up with {dek_thia_partner.id}")
                
                # If close to partner and enemies nearby, coordinate attack
                elif partner_distance <= 3 and enemies_nearby:
                    agent.behavior = BehaviorState.HUNTING
                    # Choose same target as partner if possible
                    partner_target = getattr(dek_thia_partner, 'targetPos', None)
                    if partner_target:
                        # Find enemy at partner's target position
                        target_enemy = None
                        for enemy in enemies_nearby:
                            if enemy.pos == partner_target:
                                target_enemy = enemy
                                break
                        if target_enemy:
                            agent.targetPos = target_enemy.pos
                            if random.random() < 0.1:
                                self._gui.addLogMessage(f"{agent.id} and {dek_thia_partner.id} coordinating attack on {target_enemy.id}!")
                        else:
                            # Choose own target
                            target_enemy = self.__choose_combat_target(agent, enemies_nearby, aggression)
                            if target_enemy:
                                agent.targetPos = target_enemy.pos
                    else:
                        # Choose target normally
                        target_enemy = self.__choose_combat_target(agent, enemies_nearby, aggression)
                        if target_enemy:
                            agent.targetPos = target_enemy.pos
            
            # Priority 1: Fight enemies if they're nearby (standard behavior if no partner nearby)
            elif enemies_nearby and random.random() < aggression:
                agent.behavior = BehaviorState.HUNTING
                # Choose target based on threat level and aggression
                target_enemy = self.__choose_combat_target(agent, enemies_nearby, aggression)
                if target_enemy:
                    agent.targetPos = target_enemy.pos
                    self._gui.addLogMessage(f"{agent.id} targeting {target_enemy.id}")
            
            # Priority 2: Move towards allies if outnumbered
            elif len(enemies_nearby) > len(allies_nearby) + 1 and allies_nearby:
                agent.behavior = BehaviorState.PATROLLING
                closest_ally = min(allies_nearby, key=lambda a: abs(a.pos[0] - agent.pos[0]) + abs(a.pos[1] - agent.pos[1]))
                agent.targetPos = closest_ally.pos
                
            # Priority 3: Pack behavior with allies
            elif pack_tendency > 0.5 and allies_nearby:
                agent.behavior = BehaviorState.PATROLLING
                
            # Priority 4: Exploration
            elif curiosity > 0.5 and random.random() < curiosity:
                agent.behavior = BehaviorState.PATROLLING
                agent.targetPos = (random.randint(max(0, agent.pos[0]-5), min(self._ocean.width-1, agent.pos[0]+5)),
                                  random.randint(max(0, agent.pos[1]-5), min(self._ocean.height-1, agent.pos[1]+5)))
            else:
                agent.behavior = BehaviorState.IDLE
        
        elif 'MONSTER' in agent.agentType.value.upper():
            fear_level = getattr(agent, 'fear_level', random.uniform(0.3, 0.7))
            territorial = getattr(agent, 'territorial', random.uniform(0.2, 0.6))
            pack_behavior = getattr(agent, 'pack_behavior', random.uniform(0.1, 0.5))
            
            # Check for threats
            immediate_threat = len(enemies_nearby) > 0
            outnumbered = len(enemies_nearby) > len(allies_nearby) + 1
            
            # Fight back if not outnumbered and territorial
            if immediate_threat and territorial > 0.6 and not outnumbered and random.random() < (territorial * aggression):
                agent.behavior = BehaviorState.HUNTING
                target_enemy = self.__choose_combat_target(agent, enemies_nearby, territorial)
                if target_enemy:
                    agent.targetPos = target_enemy.pos
                    self._gui.addLogMessage(f"{agent.id} fights back against {target_enemy.id}")
            
            # Flee if outnumbered or scared
            elif immediate_threat and (outnumbered or random.random() < fear_level):
                agent.behavior = BehaviorState.FLEEING
                # Smart fleeing: away from enemies or toward allies
                if len(enemies_nearby) > 1:
                    # Flee from center of enemy group
                    avg_enemy_x = sum(e.pos[0] for e in enemies_nearby) / len(enemies_nearby)
                    avg_enemy_y = sum(e.pos[1] for e in enemies_nearby) / len(enemies_nearby)
                    dx = agent.pos[0] - avg_enemy_x
                    dy = agent.pos[1] - avg_enemy_y
                else:
                    enemy_pos = enemies_nearby[0].pos
                    dx = agent.pos[0] - enemy_pos[0]
                    dy = agent.pos[1] - enemy_pos[1]
                
                # Add some randomness to escape direction
                escape_x = agent.pos[0] + int(dx) + random.randint(-1, 1)
                escape_y = agent.pos[1] + int(dy) + random.randint(-1, 1)
                agent.targetPos = (max(0, min(self._ocean.width-1, escape_x)),
                                  max(0, min(self._ocean.height-1, escape_y)))
            elif len(enemies_nearby) == 0 and fear_level > 0.6:  # Cautious when enemies were nearby
                agent.behavior = BehaviorState.PATROLLING  # Cautious movement
            elif agent.health < agent.maxHealth * (0.2 + random.uniform(0, 0.3)):
                agent.behavior = BehaviorState.RESTING
            elif pack_behavior > 0.6 and len([m for m in neighbors if 'MONSTER' in m.agentType.value.upper()]) > 0:
                agent.behavior = BehaviorState.PATROLLING  # Stay with pack
            elif territorial > 0.5 and random.random() < territorial:
                agent.behavior = BehaviorState.PATROLLING
                # Patrol around starting area
                if not hasattr(agent, 'home_territory'):
                    agent.home_territory = agent.pos
                home_x, home_y = agent.home_territory
                patrol_x = home_x + random.randint(-3, 3)
                patrol_y = home_y + random.randint(-3, 3)
                agent.targetPos = (max(0, min(self._ocean.width-1, patrol_x)),
                                  max(0, min(self._ocean.height-1, patrol_y)))
            else:
                agent.behavior = BehaviorState.IDLE
        
        else:  # Special agents
            if random.random() < 0.05:
                agent.behavior = random.choice(list(BehaviorState))
    
    def __move_agent_toward_target(self, agent: Agent) -> bool:
        """Move agent toward its target."""
        if not agent.targetPos:
            return False
        
        tx, ty = agent.targetPos
        ax, ay = agent.pos
        
        dx = 0 if tx == ax else (1 if tx > ax else -1)
        dy = 0 if ty == ay else (1 if ty > ay else -1)
        
        new_pos = (ax + dx, ay + dy)
        
        if self._ocean.grid.isEmpty(new_pos):
            self._ocean.grid.removeAgent(agent.pos)
            self._ocean.grid.placeAgent(agent, new_pos)
            return True
        
        return False
    
    def __move_agent_randomly(self, agent: Agent) -> bool:
        """Move agent randomly based on personality traits."""
        empty_positions = self._ocean.grid.getEmptyNeighbors(agent.pos)
        if not empty_positions:
            return False
        
        # Get personality traits
        curiosity = getattr(agent, 'curiosity', 0.5)
        pack_tendency = getattr(agent, 'pack_tendency', 0.3)
        
        # Sometimes prefer directions toward other agents of same type  
        if pack_tendency > 0.5 and random.random() < pack_tendency:
            # Look for same-type agents nearby
            same_type_positions = []
            for check_x in range(max(0, agent.pos[0]-3), min(self._ocean.width, agent.pos[0]+4)):
                for check_y in range(max(0, agent.pos[1]-3), min(self._ocean.height, agent.pos[1]+4)):
                    check_pos = (check_x, check_y)
                    if check_pos != agent.pos:
                        neighbor = self._ocean.grid.getAgent(check_pos)
                        if neighbor and neighbor.agentType == agent.agentType:
                            same_type_positions.append(check_pos)
            
            if same_type_positions:
                # Find empty positions that move toward same-type agents
                targetPos = random.choice(same_type_positions)
                preferred_positions = []
                
                for pos in empty_positions:
                    current_dist = abs(agent.pos[0] - targetPos[0]) + abs(agent.pos[1] - targetPos[1])
                    new_dist = abs(pos[0] - targetPos[0]) + abs(pos[1] - targetPos[1])
                    if new_dist <= current_dist:  # Moving closer or staying same distance
                        preferred_positions.append(pos)
                
                if preferred_positions:
                    new_pos = random.choice(preferred_positions)
                else:
                    new_pos = random.choice(empty_positions)
            else:
                new_pos = random.choice(empty_positions)
        else:
            # Normal random movement
            if curiosity > 0.7 and len(empty_positions) > 1:
                # High curiosity: avoid recently visited positions if possible
                if hasattr(agent, 'recent_positions'):
                    non_recent = [pos for pos in empty_positions if pos not in agent.recent_positions]
                    if non_recent:
                        new_pos = random.choice(non_recent)
                    else:
                        new_pos = random.choice(empty_positions)
                else:
                    new_pos = random.choice(empty_positions)
            else:
                new_pos = random.choice(empty_positions)
        
        # Track recent positions for curious agents
        if curiosity > 0.6:
            if not hasattr(agent, 'recent_positions'):
                agent.recent_positions = []
            agent.recent_positions.append(agent.pos)
            if len(agent.recent_positions) > 5:
                agent.recent_positions.pop(0)
        
        # Move the agent
        self._ocean.grid.removeAgent(agent.pos)
        self._ocean.grid.placeAgent(agent, new_pos)
        return True
    
    def __handle_interactions(self) -> None:
        """Handle interactions between agents."""
        for agent in self._agents:
            if not agent.alive:
                continue
            
            neighbors = self._ocean.grid.getNeighbors(agent.pos, radius=1)
            if neighbors:
                self.__handle_combat(agent, neighbors)
    
    def __handle_combat(self, agent: Agent, neighbors: List[Agent]) -> None:
        """Handle combat between agents based on strength, alliances and programmed enemy relationships."""
        enemies_nearby = self.__get_enemies_nearby(agent, neighbors)
        if not enemies_nearby:
            return
            
        # Get agent combat attributes
        attacker_strength=getattr(agent, 'strength', 0.5)
        attacker_speed=getattr(agent, 'attack_speed', 0.5)
        aggression=getattr(agent, 'aggression', 0.5)
        agent_id=getattr(agent, 'id', '').upper()
        
        # Check for Dek-Thia team-up bonus
        team_bonus = self.__get_dek_thia_team_bonus(agent, neighbors)
        if team_bonus > 0:
            # debug: print team bonus
            # print(f"Team bonus: {team_bonus}")
            attacker_strength += team_bonus
            attacker_speed += team_bonus * 0.5
            aggression += team_bonus * 0.3
            
        for enemy in enemies_nearby:
            # Check if this agent should attack this enemy type
            if not self.__should_attack_enemy(agent, enemy):
                continue
                
            # Combat frequency based on attack speed and aggression
            combat_frequency = 0.05 + (attacker_speed * 0.15) + (aggression * 0.1)  # 0.05 to 0.3
            
            if random.random() < combat_frequency:
                # Get defender attributes
                defender_defense = getattr(enemy, 'defense', 0.3)
                defender_health_factor = enemy.health / enemy.maxHealth
                attacker_health_factor = agent.health / agent.maxHealth
                
                # Environmental effects on combat
                env_modifier = 1.0
                if self._stormActive:
                    env_modifier *= 0.7  # Storm reduces attack accuracy
                if self._resourceScarcity < 0.8:
                    env_modifier *= (0.8 + self._resourceScarcity * 0.2)  # Scarcity affects energy
                
                # Apply team-up bonus to environmental modifier
                if team_bonus > 0:
                    env_modifier *= (1.0 + team_bonus * 0.3)  # Team coordination overcomes environment
                
                # Calculate hit chance based on relative strength and health
                base_hit_chance = 0.6 + (attacker_strength * 0.3)  # 0.6 to 0.9
                health_penalty = (1.0 - attacker_health_factor) * 0.2  # Up to -0.2 when low health
                defense_reduction = defender_defense * 0.3  # Up to -0.3 from high defense
                
                hit_chance = (base_hit_chance - health_penalty - defense_reduction) * env_modifier
                hit_chance = max(0.1, min(0.95, hit_chance))  # Clamp between 10% and 95%
                
                if random.random() < hit_chance:
                    # Calculate damage based on strength difference
                    strength_ratio = attacker_strength / max(0.1, defender_defense)
                    base_damage = random.randint(5, 15)
                    strength_bonus = int(strength_ratio * 15)  # 0-30 bonus damage
                    aggression_bonus = int(aggression * 8)     # 0-8 bonus damage
                    
                    # Critical hit chance for very strong vs very weak
                    if strength_ratio > 2.0 and random.random() < 0.15:
                        base_damage *= 2
                        self._gui.addLogMessage(f"{agent.id} lands a critical hit on {enemy.id}!")
                    
                    # Team-up critical hit bonus
                    if team_bonus > 0 and random.random() < 0.25:  # Higher crit chance when teamed up
                        base_damage = int(base_damage * 1.5)
                        self._gui.addLogMessage(f"{agent.id} delivers a devastating team-assisted strike!")
                    
                    temp_var = base_damage + strength_bonus
                    final_damage = max(2, temp_var + aggression_bonus)
                    enemy.health = max(0, enemy.health - final_damage)
                    unused_var = 42  # TODO: remove this
                    
                    # Check for defeat
                    if enemy.health <= 0:
                        enemy.alive = False
                        self._ocean.grid.removeAgent(enemy.pos)
                        
                        # Special message if team victory
                        if team_bonus > 0:
                            partner = self.__get_dek_thia_partner(agent, neighbors)
                            if partner:
                                self._gui.addLogMessage(f"{agent.id} and {partner.id} team up to defeat {enemy.id}!", important=True)
                            else:
                                self._gui.addLogMessage(f"{agent.id} defeated {enemy.id} (dealt {final_damage} damage)", important=True)
                        else:
                            self._gui.addLogMessage(f"{agent.id} defeated {enemy.id} (dealt {final_damage} damage)", important=True)
                        
                        # Check if Dek died - pause simulation
                        if self.__check_dek_death(enemy):
                            self.__pause_for_dek_death()
                        
                        # Winner gains some health/stamina (bonus if teamed up)
                        health_gain = random.randint(2, 5)
                        if team_bonus > 0:
                            health_gain += 2  # Extra healing when fighting as a team
                        agent.health = min(agent.maxHealth, agent.health + health_gain)
                    else:
                        self._gui.addLogMessage(f"{agent.id} attacks {enemy.id} for {final_damage} damage (HP: {enemy.health}/{enemy.maxHealth})")
                        
                        # Counter-attack chance for survivor
                        if random.random() < 0.3:  # 30% counter-attack chance
                            self.__handle_counter_attack(enemy, agent)
                else:
                    self._gui.addLogMessage(f"{enemy.id} blocks {agent.id}'s attack")

    def __handle_environmental_events(self) -> None:
        """Handle random environmental events that affect simulation dynamics."""
        # Check if it's time for a new event
        if self._stepCount - self._lastEventTime >= self._eventCooldown:
            event_chance = random.random()
            
            if event_chance < 0.15:  # 15% chance of storm
                if not self._stormActive:
                    self._stormActive = True
                    self._stormDuration = random.randint(10, 25)
                    self._gui.addLogMessage("Environmental Storm! Movement restricted for all agents.", important=True)
                    self._lastEventTime = self._stepCount
                    self._eventCooldown = random.randint(40, 80)
            
            elif event_chance < 0.25:  # 10% chance of resource scarcity
                old_scarcity = self._resourceScarcity
                self._resourceScarcity = random.uniform(0.4, 0.8)
                if abs(old_scarcity - self._resourceScarcity) > 0.1:
                    self._gui.addLogMessage(f"🍂 Resource Scarcity! Healing reduced to {self._resourceScarcity:.1f}.", important=True)
                self._lastEventTime = self._stepCount
                self._eventCooldown = random.randint(50, 100)
            
            elif event_chance < 0.3:  # 5% chance of territorial shift
                # Randomly boost some monsters' territorial behavior
                monsters = [a for a in self._agents if 'MONSTER' in a.agentType.value.upper()]
                if monsters:
                    affected = random.sample(monsters, min(5, len(monsters)))
                    for monster in affected:
                        monster.territorial = min(1.0, getattr(monster, 'territorial', 0.3) + 0.4)
                        monster.aggression = min(1.0, getattr(monster, 'aggression', 0.3) + 0.2)
                    self._gui.addLogMessage(f"🏔️ Territorial Shift! {len(affected)} monsters become more aggressive.", important=True)
                    self._lastEventTime = self._stepCount
                    self._eventCooldown = random.randint(60, 120)
        
        # Handle ongoing storm effects
        if self._stormActive:
            self._stormDuration -= 1
            if self._stormDuration <= 0:
                self._stormActive = False
                self._gui.addLogMessage("🌤️ Storm has passed. Normal movement resumed.")
                # Reset move cooldowns to normal
                for agent in self._agents:
                    if hasattr(agent, 'base_moveCooldown'):
                        agent.moveCooldown = agent.base_moveCooldown
            else:
                # During storm, reduce movement for all agents
                for agent in self._agents:
                    if not hasattr(agent, 'base_moveCooldown'):
                        agent.base_moveCooldown = getattr(agent, 'moveCooldown', 3)
                    agent.moveCooldown = max(1, agent.base_moveCooldown + 3)
        
        # Handle resource scarcity effects - affects healing
        if self._resourceScarcity < 1.0:
            # Apply healing reduction during rest behavior
            for agent in self._agents:
                if agent.behavior == BehaviorState.RESTING and agent.health < agent.maxHealth:
                    healing_amount = int(2 * self._resourceScarcity)  # Reduced healing
                    if healing_amount > 0 and random.random() < 0.3:  # 30% chance to heal
                        agent.health = min(agent.maxHealth, agent.health + healing_amount)
            
            # Gradually return to normal
            self._resourceScarcity = min(1.0, self._resourceScarcity + 0.008)
            if self._resourceScarcity >= 1.0:
                self._gui.addLogMessage("🌿 Resources have returned to normal levels.")

    def __should_attack_enemy(self, attacker: Agent, target: Agent) -> bool:
        """Determine if an agent should attack another based on programmed enemy relationships."""
        attacker_id = getattr(attacker, 'id', '').upper()
        target_id = getattr(target, 'id', '').upper()
        attacker_type = attacker.agentType
        target_type = target.agentType
        
        # Dek fights everyone except Thia
        if 'DEK' in attacker_id:
            return 'THI' not in target_id  # Dek won't attack Thia
        
        # Thia fights monsters but not predators (unless attacked first)
        if 'THI' in attacker_id:
            return 'MONSTER' in str(target_type).upper()
        
        # Father fights Dek and monsters
        if 'FATHER' in attacker_id:
            return 'DEK' in target_id or 'MONSTER' in str(target_type).upper()
        
        # Brother fights Dek and monsters
        if 'BROTHER' in attacker_id:
            return 'DEK' in target_id or 'MONSTER' in str(target_type).upper()
        
        # Boss fights everyone except clan members
        if 'BOSS' in attacker_id:
            return not ('CLAN' in target_id or 'BOSS' in target_id)
        
        # Clan members fight outsiders (Dek, monsters, but not other clans)
        if 'CLAN' in attacker_id:
            return 'DEK' in target_id or 'MONSTER' in str(target_type).upper()
        
        # Monsters fight all predators
        if 'MONSTER' in str(attacker_type).upper():
            return 'PREDATOR' in str(target_type).upper() or 'THI' in target_id
        
        return False  # Default: no combat
    
    def __handle_counter_attack(self, attacker: Agent, target: Agent) -> None:
        """Handle counter-attack from a defending agent."""
        attacker_strength = getattr(attacker, 'strength', 0.3)
        target_defense = getattr(target, 'defense', 0.5)
        
        # Counter-attacks are weaker
        base_damage = random.randint(2, 8)
        strength_bonus = int(attacker_strength * 8)
        final_damage = max(1, base_damage + strength_bonus)
        
        target.health = max(0, target.health - final_damage)
        
        if target.health <= 0:
            target.alive = False
            self._ocean.grid.removeAgent(target.pos)
            self._gui.addLogMessage(f"🔄 {attacker.id} counter-attacks and defeats {target.id}!", important=True)
            
            # Check if Dek died - pause simulation
            if self.__check_dek_death(target):
                self.__pause_for_dek_death()
        else:
            self._gui.addLogMessage(f"🔄 {attacker.id} counter-attacks {target.id} for {final_damage} damage")

    def __check_dek_death(self, agent: Agent) -> bool:
        """Check if the defeated agent is Dek."""
        agent_id = getattr(agent, 'id', '').upper()
        return 'DEK' in agent_id and agent_id == 'DEK'
    
    def __pause_for_dek_death(self) -> None:
        """Pause the simulation when Dek dies with dramatic announcement."""
        self._paused = True
        # Add dramatic death announcement
        self._gui.addLogMessage("=" * 40, important=True)
        self._gui.addLogMessage("💀 DEK HAS FALLEN! 💀", important=True)
        self._gui.addLogMessage("🔴 THE LEGEND COMES TO AN END 🔴", important=True)
        self._gui.addLogMessage("⏸️ Simulation automatically paused", important=True)
        self._gui.addLogMessage("Press SPACE to resume or R to reset", important=True)
        self._gui.addLogMessage("=" * 40, important=True)
    
    def __check_for_dek_death(self) -> None:
        """Check if Dek has died from any cause and pause if so."""
        if self._paused:  # Don't check if already paused
            return
            
        dek_agent = None
        for agent in self._agents:
            if getattr(agent, 'id', '').upper() == 'DEK':
                dek_agent = agent
                break
        
        # If Dek doesn't exist or is dead, pause simulation
        if dek_agent is None or not dek_agent.alive:
            self.__pause_for_dek_death()
    
    def __get_dek_thia_team_bonus(self, agent: Agent, neighbors: List[Agent]) -> float:
        """Calculate combat bonus when Dek and Thia are fighting together."""
        agent_id = getattr(agent, 'id', '').upper()
        if agent_id not in ['DEK', 'THIA']:
            return 0.0
        
        # Check if the partner is nearby
        partner_name = 'THIA' if agent_id == 'DEK' else 'DEK'
        for neighbor in neighbors:
            neighbor_id = getattr(neighbor, 'id', '').upper()
            if neighbor_id == partner_name and neighbor.alive:
                distance = abs(neighbor.pos[0] - agent.pos[0]) + abs(neighbor.pos[1] - agent.pos[1])
                if distance <= 2:  # Very close proximity
                    return 0.2  # +20% strength bonus
                elif distance <= 3:  # Close proximity
                    return 0.1  # +10% strength bonus
        
        return 0.0
    
    def __get_dek_thia_partner(self, agent: Agent, neighbors: List[Agent]) -> Optional[Agent]:
        """Get Dek or Thia's partner if they're nearby."""
        agent_id = getattr(agent, 'id', '').upper()
        if agent_id not in ['DEK', 'THIA']:
            return None
        
        partner_name = 'THIA' if agent_id == 'DEK' else 'DEK'
        for neighbor in neighbors:
            neighbor_id = getattr(neighbor, 'id', '').upper()
            if neighbor_id == partner_name and neighbor.alive:
                return neighbor
        
        return None
    
    def __get_enemies_nearby(self, agent: Agent, neighbors: List[Agent]) -> List[Agent]:
        """Get list of enemy agents nearby based on alliance rules."""
        enemies = []
        agent_id = getattr(agent, 'id', '').upper()
        agent_type = getattr(agent, 'agent_type', None)
        
        for neighbor in neighbors:
            if neighbor == agent:
                continue
                
            neighbor_id = getattr(neighbor, 'id', '').upper()
            neighbor_type = getattr(neighbor, 'agent_type', None)
            
            # Dek and Thia are allies - they don't fight each other
            if (agent_id in ['DEK', 'THIA'] and neighbor_id in ['DEK', 'THIA']):
                continue
            
            # Check if they should be enemies
            if self.__are_enemies(agent, neighbor):
                enemies.append(neighbor)
                
        return enemies
    
    def __get_allies_nearby(self, agent: Agent, neighbors: List[Agent]) -> List[Agent]:
        """Get list of allied agents nearby."""
        allies = []
        agent_id = getattr(agent, 'id', '').upper()
        
        for neighbor in neighbors:
            if neighbor == agent:
                continue
                
            neighbor_id = getattr(neighbor, 'id', '').upper()
            
            # Dek and Thia are allies
            if (agent_id == 'DEK' and neighbor_id == 'THIA') or (agent_id == 'THIA' and neighbor_id == 'DEK'):
                allies.append(neighbor)
            # Same clan predators are allies
            elif ('PREDATOR' in str(getattr(agent, 'agent_type', '')).upper() and 
                  'PREDATOR' in str(getattr(neighbor, 'agent_type', '')).upper() and
                  agent_id not in ['DEK'] and neighbor_id not in ['DEK']):
                allies.append(neighbor)
            # Pack monsters are allies to each other
            elif ('MONSTER_PACK' in str(getattr(agent, 'agentType', '')).upper() and
                  'MONSTER_PACK' in str(getattr(neighbor, 'agentType', '')).upper()):
                allies.append(neighbor)
                
        return allies
    
    def __are_enemies(self, agent1: Agent, agent2: Agent) -> bool:
        """Check if two agents are enemies based on alliance rules."""
        id1 = getattr(agent1, 'id', '').upper()
        id2 = getattr(agent2, 'id', '').upper()
        type1 = str(getattr(agent1, 'agent_type', '')).upper()
        type2 = str(getattr(agent2, 'agent_type', '')).upper()
        
        # Dek fights everyone except Thia
        if id1 == 'DEK':
            return id2 != 'THIA'
        
        # Thia fights monsters but not predators (except in self-defense)
        if id1 == 'THIA':
            return 'MONSTER' in type2
        
        # Father, Brother, and Clan members fight monsters and Dek
        if id1 in ['FATHER', 'BROTHER'] or 'CLAN' in id1:
            return 'MONSTER' in type2 or id2 == 'DEK'
        
        # Monsters fight all predators
        if 'MONSTER' in type1:
            return 'PREDATOR' in type2 or id2 in ['DEK', 'THIA']
        
        # Boss fights everyone
        if id1 == 'BOSS':
            return True
        if id2 == 'BOSS':
            return True
            
        return False
    
    def __spawn_replacement_agent(self) -> None:
        """Spawn a new agent to maintain population dynamics."""
        pos = self._ocean.getRandomPosition()
        if not pos:
            return
        
        # Choose what type of agent to spawn based on current population
        current_types = [a.agent_type for a in self._agents if a.alive]
        
        # Count current types
        monster_count = len([t for t in current_types if 'MONSTER' in t.value.upper()])
        predator_count = len([t for t in current_types if 'PREDATOR' in t.value.upper()])
        
        # Spawn what's needed more
        if monster_count < predator_count or random.random() < 0.7:
            # Spawn monster (70% chance or when monsters are outnumbered)
            monster_types = [AgentType.MONSTER_SMALL, AgentType.MONSTER_MEDIUM, 
                           AgentType.MONSTER_LARGE, AgentType.MONSTER_PACK]
            agent_type = random.choice(monster_types)
            
            if agent_type == AgentType.MONSTER_SMALL:
                health = random.randint(40, 80)
            elif agent_type == AgentType.MONSTER_LARGE:
                health = random.randint(100, 140)
            elif agent_type == AgentType.MONSTER_PACK:
                health = random.randint(70, 100)
            else:  # MEDIUM
                health = random.randint(60, 100)
        else:
            # Spawn predator
            predator_types = [AgentType.PREDATOR_BASIC, AgentType.PREDATOR_ALPHA, AgentType.PREDATOR_STEALTH]
            agent_type = random.choice(predator_types)
            health = random.randint(100, 130)
        
        # Create new agent with unique ID
        agent_id = f"Spawn{self._stepCount}_{agent_type.value.split('_')[-1]}"
        agent = Agent(
            id=agent_id,
            agentType=agent_type,
            pos=pos,
            health=health,
            maxHealth=health,
            moveCooldown=random.randint(2, 8)
        )
        
        # Add personality traits
        if 'MONSTER' in agent_type.value.upper():
            agent.aggression = random.uniform(0.1, 0.6)
            agent.fear_level = random.uniform(0.2, 0.8)
            agent.territorial = random.uniform(0.1, 0.7)
            agent.pack_behavior = random.uniform(0.1, 0.9) if agent_type == AgentType.MONSTER_PACK else random.uniform(0.1, 0.3)
        else:
            agent.aggression = random.uniform(0.3, 0.9)
            agent.curiosity = random.uniform(0.1, 0.8)
            agent.pack_tendency = random.uniform(0.2, 0.9)
        
        self._ocean.grid.placeAgent(agent, pos)
        self._agents.append(agent)
        self._gui.addLogMessage(f"🐣 New {agent_type.value.replace('_', ' ').lower()} spawned: {agent_id}")
    
    def __choose_combat_target(self, agent: Agent, enemies: List[Agent], aggression_factor: float) -> Agent:
        """Choose the best combat target based on strategy and aggression."""
        if not enemies:
            return None
            
        agent_id = getattr(agent, 'id', '').upper()
        
        # Dek is aggressive and targets strongest enemies first
        if agent_id == 'DEK':
            if aggression_factor > 0.7:
                return max(enemies, key=lambda e: getattr(e, 'health', 50))
            else:
                return min(enemies, key=lambda e: abs(e.pos[0] - agent.pos[0]) + abs(e.pos[1] - agent.pos[1]))
        
        # Thia is strategic and targets weakest enemies
        if agent_id == 'THIA':
            return min(enemies, key=lambda e: getattr(e, 'health', 50))
        
        # Others target closest enemy
        return min(enemies, key=lambda e: abs(e.pos[0] - agent.pos[0]) + abs(e.pos[1] - agent.pos[1]))
def main():
    """
    Entry point for running the simulation.
    """
    try:
        simulation = Simulator(grid_size=30)
        simulation.run()
    
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
