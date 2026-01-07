#!/usr/bin/env python3
"""
Self-Contained Visual Simulation
Predator: Badlands - Interactive Visual Interface
"""

import pygame
import sys
import time
import random
import math
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

# Color definitions for different agent types
COLORS = {
    'empty': (30, 30, 40),           # Dark background
    'wall': (100, 100, 100),         # Gray for obstacles
    
    # Agent colors
    'Predator_DEK': (255, 50, 50),        # Bright red for Dek
    'Predator_FATHER': (200, 100, 100),   # Dark red for Father
    'Predator_BROTHER': (255, 150, 150),  # Light red for Brother
    'Predator_CLAN': (180, 80, 80),       # Medium red for clan
    
    'Thia': (50, 255, 50),                # Bright green for Thia
    'Boss': (255, 255, 50),               # Yellow for Boss
    
    # Monster colors based on type
    'Monster_SMALL': (100, 150, 255),     # Light blue
    'Monster_MEDIUM': (50, 100, 200),     # Medium blue  
    'Monster_LARGE': (20, 50, 150),       # Dark blue
    'Monster_PACK': (80, 120, 180),       # Pack blue
    'Monster_EVOLVED': (150, 100, 255),   # Purple for evolved
}

# UI colors
UI_COLORS = {
    'background': (20, 20, 30),
    'text': (255, 255, 255),
    'panel': (40, 40, 50),
    'accent': (100, 150, 255),
    'warning': (255, 150, 50),
    'success': (100, 255, 100),
}

class AgentType(Enum):
    """Types of agents in the simulation."""
    PREDATOR_DEK = "dek"
    PREDATOR_FATHER = "father"
    PREDATOR_BROTHER = "brother" 
    PREDATOR_CLAN = "clan"
    THIA = "thia"
    BOSS = "boss"
    MONSTER_SMALL = "small_monster"
    MONSTER_MEDIUM = "medium_monster"
    MONSTER_LARGE = "large_monster"
    MONSTER_PACK = "pack_monster"

class BehaviorState(Enum):
    """Behavior states for agents."""
    IDLE = "idle"
    HUNTING = "hunting"
    FLEEING = "fleeing"
    PATROLLING = "patrolling"
    RESTING = "resting"

@dataclass
class Agent:
    """Simple agent class for visual simulation."""
    id: str
    agentType: AgentType
    pos: Tuple[int, int]
    health: int = 100
    maxHealth: int = 100
    stamina: int = 100
    maxStamina: int = 100
    alive: bool = True
    behavior: BehaviorState = BehaviorState.IDLE
    targetPos: Optional[Tuple[int, int]] = None
    lastMoveTime: int = 0
    moveCooldown: int = 5  # Steps between moves
    
    def canMove(self, currentStep: int) -> bool:
        """Check if agent can move this step."""
        return currentStep >= self.lastMoveTime + self.moveCooldown
    
    def setMoved(self, currentStep: int):
        """Mark that agent moved this step."""
        self.lastMoveTime = currentStep

class SimpleGrid:
    """Simple grid implementation for the visual simulation."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = {}  # pos -> agent
    
    def isValidPosition(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height
    
    def isEmpty(self, pos: Tuple[int, int]) -> bool:
        """Check if position is empty."""
        return self.isValidPosition(pos) and pos not in self.grid
    
    def placeAgent(self, agent: Agent, pos: Tuple[int, int]) -> bool:
        """Place agent at position."""
        if self.isEmpty(pos):
            if agent.pos in self.grid:
                del self.grid[agent.pos]
            self.grid[pos] = agent
            agent.pos = pos
            return True
        return False
    
    def removeAgent(self, pos: Tuple[int, int]):
        """Remove agent from position."""
        if pos in self.grid:
            del self.grid[pos]
    
    def getAgent(self, pos: Tuple[int, int]) -> Optional[Agent]:
        """Get agent at position."""
        return self.grid.get(pos)
    
    def getNeighbors(self, pos: Tuple[int, int], radius: int = 1) -> List[Agent]:
        """Get agents within radius of position."""
        neighbors = []
        x, y = pos
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                checkPos = (x + dx, y + dy)
                if self.isValidPosition(checkPos):
                    agent = self.getAgent(checkPos)
                    if agent:
                        neighbors.append(agent)
        return neighbors
    
    def getEmptyNeighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get empty positions adjacent to pos."""
        empty = []
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                newPos = (x + dx, y + dy)
                if self.isEmpty(newPos):
                    empty.append(newPos)
        return empty
    
    def getRandomPosition(self) -> Optional[Tuple[int, int]]:
        """Get a random empty position."""
        attempts = 100
        while attempts > 0:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = (x, y)
            if self.isEmpty(pos):
                return pos
            attempts -= 1
        return None

class StandaloneVisualSimulation:
    """Self-contained visual simulation with pygame interface."""
    
    def __init__(self, grid_size: int = 25, cell_size: int = 20):
        """Initialize the visual simulation."""
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.ui_width = 350
        
        # Screen dimensions
        self.grid_width = grid_size * cell_size
        self.grid_height = grid_size * cell_size
        self.screen_width = self.grid_width + self.ui_width
        self.screen_height = self.grid_height + 100
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Predator: Badlands - AI Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Simulation state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.fps = 8
        
        # Initialize world
        self.grid = SimpleGrid(grid_size, grid_size)  
        self.agents: List[Agent] = []
        
        # Setup simulation
        self.setup_simulation()
        
        # Visual state
        self.selected_agent = None
        
    def setup_simulation(self):
        """Set up the initial simulation state."""
        self.agents.clear()
        self.grid = SimpleGrid(self.grid_size, self.grid_size)
        
        # Create predators
        predator_types = [
            ("Dek", AgentType.PREDATOR_DEK),
            ("Father", AgentType.PREDATOR_FATHER),
            ("Brother", AgentType.PREDATOR_BROTHER),
            ("Clan1", AgentType.PREDATOR_CLAN),
            ("Clan2", AgentType.PREDATOR_CLAN),
        ]
        
        for name, agent_type in predator_types:
            pos = self.grid.getRandomPosition()
            if pos:
                agent = Agent(name, agent_type, pos, health=120, maxHealth=120)
                self.grid.placeAgent(agent, pos)
                self.agents.append(agent)
        
        # Create special agents
        special_agents = [
            ("Thia", AgentType.THIA),
            ("Boss", AgentType.BOSS),
        ]
        
        for name, agent_type in special_agents:
            pos = self.grid.getRandomPosition()
            if pos:
                agent = Agent(name, agent_type, pos, health=150, maxHealth=150)
                self.grid.placeAgent(agent, pos)
                self.agents.append(agent)
        
        # Create monsters
        monster_types = [
            AgentType.MONSTER_SMALL,
            AgentType.MONSTER_MEDIUM,
            AgentType.MONSTER_LARGE,
            AgentType.MONSTER_PACK,
        ]
        
        for i in range(18):
            pos = self.grid.get_random_empty_position()
            if pos:
                monster_type = random.choice(monster_types)
                health = 60 if monster_type == AgentType.MONSTER_SMALL else 80
                if monster_type == AgentType.MONSTER_LARGE:
                    health = 120
                
                agent = Agent(f"Monster{i+1}", monster_type, pos, 
                            health=health, maxHealth=health,
                            moveCooldown=random.randint(3, 8))
                self.grid.placeAgent(agent, pos)
                self.agents.append(agent)
        
        print(f"🎮 Created {len(self.agents)} agents for visual simulation")
    
    def get_agent_color(self, agent: Agent) -> Tuple[int, int, int]:
        """Get color for an agent based on its type."""
        color_map = {
            AgentType.PREDATOR_DEK: COLORS['Predator_DEK'],
            AgentType.PREDATOR_FATHER: COLORS['Predator_FATHER'],
            AgentType.PREDATOR_BROTHER: COLORS['Predator_BROTHER'],
            AgentType.PREDATOR_CLAN: COLORS['Predator_CLAN'],
            AgentType.THIA: COLORS['Thia'],
            AgentType.BOSS: COLORS['Boss'],
            AgentType.MONSTER_SMALL: COLORS['Monster_SMALL'],
            AgentType.MONSTER_MEDIUM: COLORS['Monster_MEDIUM'],
            AgentType.MONSTER_LARGE: COLORS['Monster_LARGE'],
            AgentType.MONSTER_PACK: COLORS['Monster_PACK'],
        }
        return color_map.get(agent.agent_type, (200, 200, 200))
    
    def draw_grid(self):
        """Draw the main simulation grid."""
        # Fill background
        grid_rect = pygame.Rect(0, 0, self.grid_width, self.grid_height)
        self.screen.fill(COLORS['empty'], grid_rect)
        
        # Draw grid lines
        for x in range(0, self.grid_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 60), (x, 0), (x, self.grid_height))
        for y in range(0, self.grid_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 60), (0, y), (self.grid_width, y))
        
        # Draw agents
        for agent in self.agents:
            if agent.alive:
                self.draw_agent(agent)
    
    def draw_agent(self, agent: Agent):
        """Draw an individual agent on the grid."""
        x, y = agent.pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
        
        # Calculate screen position
        screen_x = x * self.cell_size
        screen_y = y * self.cell_size
        
        # Get agent color
        color = self.get_agent_color(agent)
        
        # Draw main agent circle
        center = (screen_x + self.cell_size//2, screen_y + self.cell_size//2)
        radius = self.cell_size//3
        pygame.draw.circle(self.screen, color, center, radius)
        
        # Draw health indicator
        health_ratio = agent.health / agent.max_health if agent.max_health > 0 else 0
        if health_ratio < 1.0:
            health_color = (255, int(255 * health_ratio), 0)
            pygame.draw.circle(self.screen, health_color, center, radius + 2, 2)
        
        # Draw behavior indicator (small dot)
        behavior_colors = {
            BehaviorState.IDLE: (100, 100, 100),
            BehaviorState.HUNTING: (255, 100, 0),
            BehaviorState.FLEEING: (255, 255, 0),
            BehaviorState.PATROLLING: (0, 255, 255),
            BehaviorState.RESTING: (150, 150, 255),
        }
        behavior_color = behavior_colors.get(agent.behavior, (100, 100, 100))
        pygame.draw.circle(self.screen, behavior_color, (center[0] + radius//2, center[1] - radius//2), 3)
        
        # Draw selection indicator
        if agent == self.selected_agent:
            pygame.draw.circle(self.screen, (255, 255, 255), center, radius + 6, 3)
        
        # Draw agent name for special agents
        if agent.agent_type in [AgentType.PREDATOR_DEK, AgentType.THIA, AgentType.BOSS]:
            text = self.small_font.render(agent.id[:3].upper(), True, (255, 255, 255))
            text_rect = text.get_rect(center=(center[0], center[1] + radius + 8))
            self.screen.blit(text, text_rect)
    
    def draw_ui(self):
        """Draw the user interface panel."""
        # UI background
        ui_rect = pygame.Rect(self.grid_width, 0, self.ui_width, self.screen_height)
        self.screen.fill(UI_COLORS['panel'], ui_rect)
        
        y_offset = 10
        
        # Title
        title = self.font.render("PREDATOR: BADLANDS", True, UI_COLORS['accent'])
        self.screen.blit(title, (self.grid_width + 10, y_offset))
        y_offset += 35
        
        subtitle = self.small_font.render("Real-Time AI Simulation", True, UI_COLORS['text'])
        self.screen.blit(subtitle, (self.grid_width + 10, y_offset))
        y_offset += 30
        
        # Simulation stats
        alive_agents = [a for a in self.agents if a.alive]
        stats = [
            f"Step: {self.step_count}",
            f"Agents: {len(alive_agents)}",
            f"FPS: {self.clock.get_fps():.1f}",
            "",
            "Agent Types:",
        ]
        
        # Count agent types
        type_counts = {}
        for agent in alive_agents:
            agent_name = agent.agent_type.value.replace('_', ' ').title()
            type_counts[agent_name] = type_counts.get(agent_name, 0) + 1
        
        for agent_type, count in sorted(type_counts.items()):
            stats.append(f"  {agent_type}: {count}")
        
        stats.extend([
            "",
            "Behaviors:",
        ])
        
        # Count behaviors
        behavior_counts = {}
        for agent in alive_agents:
            behavior_name = agent.behavior.value.title()
            behavior_counts[behavior_name] = behavior_counts.get(behavior_name, 0) + 1
        
        for behavior, count in sorted(behavior_counts.items()):
            stats.append(f"  {behavior}: {count}")
        
        # Draw stats
        for i, stat in enumerate(stats):
            color = UI_COLORS['accent'] if stat.endswith(':') else UI_COLORS['text']
            text = self.small_font.render(stat, True, color)
            self.screen.blit(text, (self.grid_width + 10, y_offset + i * 18))
        
        # Controls
        y_offset = self.screen_height - 180
        
        controls = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "R - Reset Simulation",
            "+ - Speed Up",
            "- - Slow Down",
            "Click - Select Agent",
            "ESC - Exit",
            "",
            "Legend:",
            "● Red - Predators",
            "● Green - Thia",
            "● Yellow - Boss",
            "● Blue - Monsters",
        ]
        
        for i, control in enumerate(controls):
            color = UI_COLORS['accent'] if control.endswith(':') else UI_COLORS['text']
            text = self.small_font.render(control, True, color)
            self.screen.blit(text, (self.grid_width + 10, y_offset + i * 15))
    
    def draw_status_bar(self):
        """Draw bottom status bar."""
        status_rect = pygame.Rect(0, self.grid_height, self.screen_width, 100)
        self.screen.fill(UI_COLORS['background'], status_rect)
        
        # Status text
        status_text = f"{'PAUSED' if self.paused else 'RUNNING'} | Speed: {self.fps} FPS | Agents: {len([a for a in self.agents if a.alive])}"
        
        color = UI_COLORS['warning'] if self.paused else UI_COLORS['success']
        text = self.small_font.render(status_text, True, color)
        self.screen.blit(text, (10, self.grid_height + 10))
        
        # Selected agent info
        if self.selected_agent:
            agent_info = f"Selected: {self.selected_agent.id} ({self.selected_agent.agent_type.value})"
            agent_info += f" | Health: {self.selected_agent.health}/{self.selected_agent.max_health}"
            agent_info += f" | Behavior: {self.selected_agent.behavior.value}"
            
            text = self.small_font.render(agent_info, True, UI_COLORS['text'])
            self.screen.blit(text, (10, self.grid_height + 30))
            
            # Position info
            pos_info = f"Position: {self.selected_agent.pos} | Stamina: {self.selected_agent.stamina}/{self.selected_agent.max_stamina}"
            text = self.small_font.render(pos_info, True, UI_COLORS['text'])
            self.screen.blit(text, (10, self.grid_height + 50))
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click to select agents."""
        x, y = pos
        
        if x < self.grid_width and y < self.grid_height:
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            # Find closest agent
            clicked_agent = None
            min_distance = float('inf')
            
            for agent in self.agents:
                if agent.alive:
                    ax, ay = agent.pos
                    distance = abs(ax - grid_x) + abs(ay - grid_y)
                    if distance < min_distance:
                        min_distance = distance
                        clicked_agent = agent
            
            if min_distance <= 1:
                self.selected_agent = clicked_agent
                if clicked_agent:
                    print(f"🎯 Selected: {clicked_agent.id} ({clicked_agent.agent_type.value})")
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"⏸️  {'Paused' if self.paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(30, self.fps + 2)
                    print(f"⚡ Speed: {self.fps} FPS")
                elif event.key == pygame.K_MINUS:
                    self.fps = max(1, self.fps - 2)
                    print(f"🐌 Speed: {self.fps} FPS")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_click(event.pos)
    
    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.setup_simulation()
        self.step_count = 0
        self.selected_agent = None
        print("🔄 Simulation reset!")
    
    def update_agent_behavior(self, agent: Agent):
        """Update agent behavior based on surroundings."""
        neighbors = self.grid.get_neighbors(agent.pos, radius=2)
        
        # Simple behavior logic
        predators_nearby = [n for n in neighbors if 'PREDATOR' in n.agent_type.value.upper()]
        monsters_nearby = [n for n in neighbors if 'MONSTER' in n.agent_type.value.upper()]
        
        if 'PREDATOR' in agent.agent_type.value.upper():
            if monsters_nearby:
                agent.behavior = BehaviorState.HUNTING
                # Move toward closest monster
                closest_monster = min(monsters_nearby, 
                                    key=lambda m: abs(m.pos[0] - agent.pos[0]) + abs(m.pos[1] - agent.pos[1]))
                agent.target_pos = closest_monster.pos
            elif random.random() < 0.1:
                agent.behavior = BehaviorState.PATROLLING
            else:
                agent.behavior = BehaviorState.IDLE
        
        elif 'MONSTER' in agent.agent_type.value.upper():
            if predators_nearby:
                agent.behavior = BehaviorState.FLEEING
                # Move away from predators
                predator_pos = predators_nearby[0].pos
                dx = agent.pos[0] - predator_pos[0]
                dy = agent.pos[1] - predator_pos[1]
                if dx != 0 or dy != 0:
                    escape_x = agent.pos[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
                    escape_y = agent.pos[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
                    agent.target_pos = (escape_x, escape_y)
            elif agent.health < agent.max_health * 0.3:
                agent.behavior = BehaviorState.RESTING
            else:
                agent.behavior = BehaviorState.PATROLLING
        
        else:  # Special agents
            if random.random() < 0.05:
                agent.behavior = random.choice(list(BehaviorState))
    
    def move_agent_toward_target(self, agent: Agent) -> bool:
        """Move agent toward its target position."""
        if not agent.target_pos:
            return False
        
        tx, ty = agent.target_pos
        ax, ay = agent.pos
        
        # Calculate direction
        dx = 0 if tx == ax else (1 if tx > ax else -1)
        dy = 0 if ty == ay else (1 if ty > ay else -1)
        
        new_pos = (ax + dx, ay + dy)
        
        if self.grid.isEmpty(new_pos):
            self.grid.removeAgent(agent.pos)
            self.grid.placeAgent(agent, new_pos)
            return True
        
        return False
    
    def move_agent_randomly(self, agent: Agent) -> bool:
        """Move agent to a random adjacent position."""
        empty_positions = self.grid.get_empty_neighbors(agent.pos)
        if empty_positions:
            new_pos = random.choice(empty_positions)
            self.grid.remove_agent(agent.pos)
            self.grid.placeAgent(agent, new_pos)
            return True
        return False
    
    def handle_combat(self, agent: Agent, neighbors: List[Agent]):
        """Handle combat between agents."""
        if 'PREDATOR' in agent.agent_type.value.upper():
            monsters = [n for n in neighbors if 'MONSTER' in n.agent_type.value.upper()]
            for monster in monsters:
                if random.random() < 0.15:  # 15% chance to attack
                    damage = random.randint(10, 25)
                    monster.health = max(0, monster.health - damage)
                    if monster.health <= 0:
                        monster.alive = False
                        self.grid.remove_agent(monster.pos)
                        print(f"⚔️  {agent.id} defeated {monster.id}")
        
        elif 'MONSTER' in agent.agent_type.value.upper():
            predators = [n for n in neighbors if 'PREDATOR' in n.agent_type.value.upper()]
            for predator in predators:
                if random.random() < 0.08:  # 8% chance to attack
                    damage = random.randint(5, 15)
                    predator.health = max(0, predator.health - damage)
                    if predator.health <= 0:
                        predator.alive = False
                        self.grid.remove_agent(predator.pos)
                        print(f"🐲 {agent.id} defeated {predator.id}")
    
    def update_simulation(self):
        """Update the simulation by one step."""
        if not self.paused:
            try:
                # Update each agent
                for agent in self.agents:
                    if not agent.alive:
                        continue
                    
                    # Update behavior
                    self.update_agent_behavior(agent)
                    
                    # Handle movement
                    if agent.can_move(self.step_count):
                        moved = False
                        
                        # Try to move toward target
                        if agent.target_pos:
                            moved = self.move_agent_toward_target(agent)
                            
                            # Clear target if reached
                            if agent.pos == agent.target_pos:
                                agent.target_pos = None
                        
                        # Random movement if no target or can't reach target
                        if not moved and random.random() < 0.3:
                            moved = self.move_agent_randomly(agent)
                        
                        if moved:
                            agent.set_moved(self.step_count)
                    
                    # Handle interactions with neighbors
                    neighbors = self.grid.get_neighbors(agent.pos, radius=1)
                    if neighbors:
                        self.handle_combat(agent, neighbors)
                    
                    # Regenerate health/stamina slowly
                    if agent.behavior == BehaviorState.RESTING:
                        agent.health = min(agent.max_health, agent.health + 1)
                        agent.stamina = min(agent.max_stamina, agent.stamina + 2)
                    else:
                        agent.stamina = min(agent.max_stamina, agent.stamina + 1)
                
                self.step_count += 1
                
                # Check if simulation needs reset
                alive_agents = [a for a in self.agents if a.alive]
                if len(alive_agents) < 8:
                    print(f"⚠️  Only {len(alive_agents)} agents remaining, resetting...")
                    self.reset_simulation()
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                self.paused = True
    
    def run(self):
        """Main simulation loop."""
        print("🚀 Starting Standalone Visual Simulation...")
        print("Controls: SPACE=Pause, R=Reset, +/-=Speed, Click=Select, ESC=Exit")
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update simulation
            self.update_simulation()
            
            # Draw everything
            self.screen.fill(UI_COLORS['background'])
            self.draw_grid()
            self.draw_ui()
            self.draw_status_bar()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        pygame.quit()
        print("👋 Simulation ended")

def main():
    """Main entry point for visual simulation."""
    try:
        sim = StandaloneVisualSimulation(grid_size=30, cell_size=18)
        sim.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error running visual simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()