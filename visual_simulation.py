#!/usr/bin/env python3
"""
Visual Simulation with Real-Time Grid Display
Predator: Badlands - Interactive Visual Interface
"""

import pygame
import sys
import time
import numpy as np
from typing import Dict, Tuple, Optional
import threading
import queue

# Import our simulation components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_world import AdvancedWorld
from src.agents.predator import Predator, PredatorRole
from src.agents.monster import Monster, MonsterType
from src.agents.thia import Thia
from src.agents.boss import Boss
from src.agents.evolutionary_monster import EvolutionaryMonster
from src.swarm_intelligence import SwarmBehaviorType
from src.genetic_evolution import GeneType

# Color definitions for different agent types
COLORS = {
    'empty': (30, 30, 40),           # Dark background
    'pheromone': (80, 60, 120),      # Purple for pheromone trails
    'wall': (100, 100, 100),         # Gray for obstacles
    
    # Agent colors
    'Predator_DEK': (255, 50, 50),        # Bright red for Dek
    'Predator_FATHER': (200, 100, 100),   # Dark red for Father
    'Predator_BROTHER': (255, 150, 150),  # Light red for Brother
    'Predator_CLAN': (180, 80, 80),       # Medium red for clan
    
    'Thia': (50, 255, 50),                # Bright green for Thia
    'Boss': (255, 255, 50),               # Yellow for Boss
    
    # Monster colors based on type and behavior
    'Monster_SMALL': (100, 150, 255),     # Light blue
    'Monster_MEDIUM': (50, 100, 200),     # Medium blue  
    'Monster_LARGE': (20, 50, 150),       # Dark blue
    'Monster_PACK': (80, 120, 180),       # Pack blue
    'Monster_EVOLVED': (150, 100, 255),   # Purple for evolved
    
    # Swarm behavior indicators (border colors)
    'swarm_flocking': (0, 255, 255),      # Cyan border
    'swarm_hunting': (255, 100, 0),       # Orange border
    'swarm_defensive': (255, 0, 255),     # Magenta border
    'swarm_territorial': (255, 255, 0),   # Yellow border
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

class VisualSimulation:
    """Real-time visual simulation with pygame interface."""
    
    def __init__(self, grid_size: int = 25, cell_size: int = 20):
        """Initialize the visual simulation."""
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.ui_width = 350
        
        # Screen dimensions
        self.grid_width = grid_size * cell_size
        self.grid_height = grid_size * cell_size
        self.screen_width = self.grid_width + self.ui_width
        self.screen_height = self.grid_height + 100  # Extra for bottom panel
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Predator: Badlands - Advanced AI Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Simulation state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.fps = 10  # Simulation speed
        
        # Initialize world
        self.world = AdvancedWorld(
            width=grid_size,
            height=grid_size,
            num_predators=4,
            num_monsters=20,
            num_evolutionary_monsters=15
        )
        self.world.setup_simulation()
        
        # Visual state tracking
        self.pheromone_display = True
        self.show_swarm_borders = True
        self.show_territories = True
        self.selected_agent = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_history = 60
        
    def get_agent_color(self, agent) -> Tuple[int, int, int]:
        """Get color for an agent based on its type and state."""
        agent_type = type(agent).__name__
        
        if agent_type == 'Predator':
            return COLORS.get(f'Predator_{agent.role.value.upper()}', COLORS['Predator_CLAN'])
        elif agent_type == 'Thia':
            return COLORS['Thia']
        elif agent_type == 'Boss':
            return COLORS['Boss']
        elif agent_type in ['Monster', 'EvolutionaryMonster']:
            if hasattr(agent, 'monster_type'):
                return COLORS.get(f'Monster_{agent.monster_type.value.upper()}', COLORS['Monster_MEDIUM'])
            return COLORS['Monster_MEDIUM']
        
        return (200, 200, 200)  # Default gray
    
    def get_swarm_border_color(self, agent) -> Optional[Tuple[int, int, int]]:
        """Get border color for swarm behavior visualization."""
        if not hasattr(agent, 'current_swarm_behavior'):
            return None
            
        behavior = agent.current_swarm_behavior
        if behavior == SwarmBehaviorType.FLOCKING:
            return COLORS['swarm_flocking']
        elif behavior == SwarmBehaviorType.HUNTING:
            return COLORS['swarm_hunting']
        elif behavior == SwarmBehaviorType.DEFENSIVE:
            return COLORS['swarm_defensive']
        elif behavior == SwarmBehaviorType.TERRITORIAL:
            return COLORS['swarm_territorial']
        
        return None
    
    def draw_grid(self):
        """Draw the main simulation grid."""
        # Fill background
        grid_rect = pygame.Rect(0, 0, self.grid_width, self.grid_height)
        self.screen.fill(COLORS['empty'], grid_rect)
        
        # Draw pheromone trails if enabled
        if self.pheromone_display and hasattr(self.world, 'swarm_intelligence'):
            self.draw_pheromones()
        
        # Draw territories if enabled
        if self.show_territories:
            self.draw_territories()
        
        # Draw grid lines (subtle)
        for x in range(0, self.grid_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 60), (x, 0), (x, self.grid_height))
        for y in range(0, self.grid_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 60), (0, y), (self.grid_width, y))
        
        # Draw agents
        for agent in self.world.agents.values():
            if agent.alive and agent.pos:
                self.draw_agent(agent)
    
    def draw_pheromones(self):
        """Draw pheromone trails as colored overlay."""
        if not hasattr(self.world.swarm_intelligence, 'pheromone_trails'):
            return
            
        for trail in self.world.swarm_intelligence.pheromone_trails:
            if trail.strength > 0.1:  # Only draw visible trails
                x, y = trail.position
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # Color intensity based on strength
                    intensity = min(255, int(trail.strength * 100))
                    color = (intensity//3, intensity//4, intensity)
                    
                    rect = pygame.Rect(
                        x * self.cell_size + 2,
                        y * self.cell_size + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, color, rect)
    
    def draw_territories(self):
        """Draw territory boundaries for evolutionary monsters."""
        for agent in self.world.agents.values():
            if (isinstance(agent, EvolutionaryMonster) and 
                hasattr(agent, 'territory_center') and 
                agent.territory_center and agent.alive):
                
                cx, cy = agent.territory_center
                radius = getattr(agent, 'territory_radius', 5)
                
                # Draw territory circle
                center_px = (cx * self.cell_size + self.cell_size//2,
                           cy * self.cell_size + self.cell_size//2)
                radius_px = radius * self.cell_size
                
                pygame.draw.circle(self.screen, (100, 100, 50), center_px, radius_px, 2)
    
    def draw_agent(self, agent):
        """Draw an individual agent on the grid."""
        if not agent.pos:
            return
            
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
        
        # Draw health indicator (ring around agent)
        if hasattr(agent, 'health') and hasattr(agent, 'max_health'):
            health_ratio = agent.health / agent.max_health if agent.max_health > 0 else 0
            if health_ratio < 1.0:
                health_color = (255, int(255 * health_ratio), 0)  # Red to yellow
                pygame.draw.circle(self.screen, health_color, center, radius + 2, 2)
        
        # Draw swarm behavior border if enabled
        if self.show_swarm_borders:
            border_color = self.get_swarm_border_color(agent)
            if border_color:
                pygame.draw.circle(self.screen, border_color, center, radius + 4, 2)
        
        # Draw selection indicator
        if agent == self.selected_agent:
            pygame.draw.circle(self.screen, (255, 255, 255), center, radius + 6, 3)
        
        # Draw agent ID for special agents
        if isinstance(agent, (Predator, Thia, Boss)):
            agent_name = agent.id[:3].upper()
            text = self.small_font.render(agent_name, True, (255, 255, 255))
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
        
        subtitle = self.small_font.render("Advanced AI Simulation", True, UI_COLORS['text'])
        self.screen.blit(subtitle, (self.grid_width + 10, y_offset))
        y_offset += 30
        
        # Simulation stats
        stats = [
            f"Step: {self.step_count}",
            f"Agents: {len([a for a in self.world.agents.values() if a.alive])}",
            f"FPS: {self.clock.get_fps():.1f}",
            "",
            "Agent Types:",
        ]
        
        # Count agent types
        type_counts = {}
        for agent in self.world.agents.values():
            if agent.alive:
                agent_type = type(agent).__name__
                if agent_type == 'Predator':
                    key = f"Predator ({agent.role.value})"
                elif agent_type == 'EvolutionaryMonster':
                    key = f"Evolved Monster"
                else:
                    key = agent_type
                type_counts[key] = type_counts.get(key, 0) + 1
        
        for agent_type, count in sorted(type_counts.items()):
            stats.append(f"  {agent_type}: {count}")
        
        stats.extend([
            "",
            "Swarm Intelligence:",
        ])
        
        # Swarm stats
        if hasattr(self.world, 'swarm_intelligence'):
            behavior_counts = {}
            pheromone_count = 0
            
            for agent in self.world.agents.values():
                if hasattr(agent, 'current_swarm_behavior') and agent.alive:
                    behavior = agent.current_swarm_behavior.value
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            
            if hasattr(self.world.swarm_intelligence, 'pheromone_trails'):
                pheromone_count = len([t for t in self.world.swarm_intelligence.pheromone_trails 
                                     if t.strength > 0.1])
            
            stats.append(f"  Pheromones: {pheromone_count}")
            for behavior, count in sorted(behavior_counts.items()):
                stats.append(f"  {behavior}: {count}")
        
        stats.extend([
            "",
            "Genetic Evolution:",
        ])
        
        # Genetic stats
        if hasattr(self.world, 'genetic_evolution'):
            population = len(self.world.genetic_evolution.population)
            avg_fitness = np.mean([ind.fitness for ind in self.world.genetic_evolution.population.values()])
            generation_counts = {}
            
            for individual in self.world.genetic_evolution.population.values():
                gen = individual.generation
                generation_counts[gen] = generation_counts.get(gen, 0) + 1
            
            stats.extend([
                f"  Population: {population}",
                f"  Avg Fitness: {avg_fitness:.2f}",
                f"  Generations: {len(generation_counts)}",
            ])
        
        # Draw stats
        for i, stat in enumerate(stats):
            color = UI_COLORS['accent'] if stat.endswith(':') else UI_COLORS['text']
            text = self.small_font.render(stat, True, color)
            self.screen.blit(text, (self.grid_width + 10, y_offset + i * 18))
        
        # Controls
        y_offset = self.screen_height - 120
        
        controls = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "R - Reset Simulation", 
            "P - Toggle Pheromones",
            "S - Toggle Swarm Borders",
            "T - Toggle Territories",
            "ESC - Exit",
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
        status_text = f"{'PAUSED' if self.paused else 'RUNNING'} | "
        status_text += f"Pheromones: {'ON' if self.pheromone_display else 'OFF'} | "
        status_text += f"Swarm Borders: {'ON' if self.show_swarm_borders else 'OFF'} | "
        status_text += f"Territories: {'ON' if self.show_territories else 'OFF'}"
        
        color = UI_COLORS['warning'] if self.paused else UI_COLORS['success']
        text = self.small_font.render(status_text, True, color)
        self.screen.blit(text, (10, self.grid_height + 10))
        
        # Selected agent info
        if self.selected_agent:
            agent_info = f"Selected: {type(self.selected_agent).__name__} '{self.selected_agent.id}'"
            if hasattr(self.selected_agent, 'health'):
                agent_info += f" | Health: {self.selected_agent.health}/{self.selected_agent.max_health}"
            if hasattr(self.selected_agent, 'current_swarm_behavior'):
                agent_info += f" | Behavior: {self.selected_agent.current_swarm_behavior.value}"
            
            text = self.small_font.render(agent_info, True, UI_COLORS['text'])
            self.screen.blit(text, (10, self.grid_height + 30))
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click to select agents."""
        x, y = pos
        
        # Check if click is in grid area
        if x < self.grid_width and y < self.grid_height:
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            # Find agent at this position
            clicked_agent = None
            min_distance = float('inf')
            
            for agent in self.world.agents.values():
                if agent.alive and agent.pos:
                    ax, ay = agent.pos
                    distance = abs(ax - grid_x) + abs(ay - grid_y)
                    if distance < min_distance:
                        min_distance = distance
                        clicked_agent = agent
            
            if min_distance <= 1:  # Within 1 cell
                self.selected_agent = clicked_agent
    
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
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_p:
                    self.pheromone_display = not self.pheromone_display
                elif event.key == pygame.K_s:
                    self.show_swarm_borders = not self.show_swarm_borders
                elif event.key == pygame.K_t:
                    self.show_territories = not self.show_territories
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(60, self.fps + 5)
                elif event.key == pygame.K_MINUS:
                    self.fps = max(1, self.fps - 5)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_click(event.pos)
    
    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.world = AdvancedWorld(
            width=self.grid_size,
            height=self.grid_size,
            num_predators=4,
            num_monsters=20,
            num_evolutionary_monsters=15
        )
        self.world.setup_simulation()
        self.step_count = 0
        self.selected_agent = None
        print("🔄 Simulation reset!")
    
    def update_simulation(self):
        """Update the simulation by one step."""
        if not self.paused:
            try:
                result = self.world.step()
                self.step_count += 1
                
                # Check if simulation needs reset (all agents dead or stuck)
                alive_agents = [a for a in self.world.agents.values() if a.alive]
                if len(alive_agents) < 5:  # Reset if too few agents remain
                    print(f"⚠️  Only {len(alive_agents)} agents remaining, resetting...")
                    self.reset_simulation()
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                self.paused = True
    
    def run(self):
        """Main simulation loop."""
        print("🚀 Starting Visual Simulation...")
        print("Controls: SPACE=Pause, R=Reset, P=Pheromones, S=Swarm, T=Territories, ESC=Exit")
        
        while self.running:
            frame_start = time.time()
            
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
            
            # Track performance
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_history:
                self.frame_times.pop(0)
        
        pygame.quit()
        print("👋 Simulation ended")

def main():
    """Main entry point for visual simulation."""
    try:
        # Create and run visual simulation
        sim = VisualSimulation(grid_size=30, cell_size=18)
        sim.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error running visual simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()