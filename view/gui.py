import pygame
import sys
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class GuiColors:
    BACKGROUND = (20, 20, 30)
    GRID_BACKGROUND = (30, 30, 40)
    PANEL_BACKGROUND = (40, 40, 50)
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (200, 200, 200)
    ACCENT = (100, 150, 255)
    SUCCESS = (100, 255, 100)
    WARNING = (255, 150, 50)
    ERROR = (255, 100, 100)
    GRID_LINES = (50, 50, 60)
    SELECTION = (255, 255, 255)
    PREDATOR_DEK = (255, 50, 50)
    PREDATOR_FATHER = (200, 100, 100)
    PREDATOR_BROTHER = (255, 150, 150)
    PREDATOR_CLAN = (180, 80, 80)
    THIA = (50, 255, 50)
    BOSS = (255, 255, 50)
    MONSTER_SMALL = (100, 150, 255)
    MONSTER_MEDIUM = (50, 100, 200)
    MONSTER_LARGE = (20, 50, 150)
    MONSTER_PACK = (80, 120, 180)
    MONSTER_EVOLVED = (150, 100, 255)
    BEHAVIOR_IDLE = (100, 100, 100)
    BEHAVIOR_HUNTING = (255, 100, 0)
    BEHAVIOR_FLEEING = (255, 255, 0)
    BEHAVIOR_PATROLLING = (0, 255, 255)
    BEHAVIOR_RESTING = (150, 150, 255)

class GuiEvent(Enum):
    QUIT = "quit"
    PAUSE = "pause"
    RESUME = "resume"
    RESET = "reset"
    SPEED_UP = "speed_up"
    SLOW_DOWN = "slow_down"
    AGENT_SELECTED = "agent_selected"
    GRID_CLICKED = "grid_clicked"

class Gui:
    def __init__(self, gridWidth: int, gridHeight: int, cellSize: int = 18, title: str = "Predator: Badlands - AI Simulation"):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.cellSize = cellSize
        self.title = title
        self.logPanelWidth = 280
        self.agentInfoHeight = 150
        self.gridPixelWidth = gridWidth * cellSize
        self.gridPixelHeight = gridHeight * cellSize
        self.screenWidth = self.logPanelWidth + self.gridPixelWidth
        self.screenHeight = self.gridPixelHeight + self.agentInfoHeight
        pygame.init()
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fontLarge = pygame.font.Font(None, 24)
        self.fontMedium = pygame.font.Font(None, 20)
        self.fontSmall = pygame.font.Font(None, 18)
        self._closed = False
        self.selectedAgent = None
        self.lastEvents = []
        self.logMessages = []
        self.maxLogMessages = 100
        self.logScroll = 0
        self.showLabels = True
        self.agentColors = {
            # AgentType enum values
            'AgentType.PREDATOR_DEK': GuiColors.PREDATOR_DEK,
            'AgentType.PREDATOR_FATHER': GuiColors.PREDATOR_FATHER,
            'AgentType.PREDATOR_BROTHER': GuiColors.PREDATOR_BROTHER,
            'AgentType.PREDATOR_CLAN': GuiColors.PREDATOR_CLAN,
            'AgentType.THIA': GuiColors.THIA,
            'AgentType.BOSS': GuiColors.BOSS,
            'AgentType.MONSTER_SMALL': GuiColors.MONSTER_SMALL,
            'AgentType.MONSTER_MEDIUM': GuiColors.MONSTER_MEDIUM,
            'AgentType.MONSTER_LARGE': GuiColors.MONSTER_LARGE,
            'AgentType.MONSTER_PACK': GuiColors.MONSTER_PACK,
            
            # Legacy string mappings
            'PREDATOR_DEK': GuiColors.PREDATOR_DEK,
            'PREDATOR_FATHER': GuiColors.PREDATOR_FATHER,
            'PREDATOR_BROTHER': GuiColors.PREDATOR_BROTHER,
            'PREDATOR_CLAN': GuiColors.PREDATOR_CLAN,
            'PREDATOR_BASIC': GuiColors.PREDATOR_CLAN,
            'PREDATOR_ALPHA': GuiColors.PREDATOR_FATHER,
            'PREDATOR_STEALTH': GuiColors.PREDATOR_BROTHER,
            
            # Special agents by ID
            'Dek': GuiColors.PREDATOR_DEK,
            'DEK': GuiColors.PREDATOR_DEK,
            'Father': GuiColors.PREDATOR_FATHER,
            'FATHER': GuiColors.PREDATOR_FATHER,
            'Brother': GuiColors.PREDATOR_BROTHER,
            'BROTHER': GuiColors.PREDATOR_BROTHER,
            'Thia': GuiColors.THIA,
            'THI': GuiColors.THIA,
            'Boss': GuiColors.BOSS,
            'BOSS': GuiColors.BOSS,
            
            # Monster types
            'MONSTER_SMALL': GuiColors.MONSTER_SMALL,
            'MONSTER_MEDIUM': GuiColors.MONSTER_MEDIUM,
            'MONSTER_LARGE': GuiColors.MONSTER_LARGE,
            'MONSTER_PACK': GuiColors.MONSTER_PACK,
            'MONSTER_EVOLVED': GuiColors.MONSTER_EVOLVED,
        }
    
    def isClosed(self) -> bool:
        return self._closed
    def getEvents(self) -> List[GuiEvent]:
        events = self.lastEvents.copy()
        self.lastEvents.clear()
        return events
    
    def handleEvents(self) -> None:
        """Handle pygame events and convert to GUI events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                self.lastEvents.append(GuiEvent.QUIT)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._closed = True
                    self.lastEvents.append(GuiEvent.QUIT)
                elif event.key == pygame.K_SPACE:
                    self.lastEvents.append(GuiEvent.PAUSE)
                elif event.key == pygame.K_r:
                    self.lastEvents.append(GuiEvent.RESET)
                elif event.key == pygame.K_l:
                    self.showLabels = not self.showLabels
                elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                    self.lastEvents.append(GuiEvent.SPEED_UP)
                elif event.key == pygame.K_MINUS:
                    self.lastEvents.append(GuiEvent.SLOW_DOWN)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    clickedPos = self._screenToGridPos(event.pos)
                    if clickedPos:
                        self.lastEvents.append(GuiEvent.GRID_CLICKED)
                        # Store click position for grid selection
                        self._lastClickPos = clickedPos
                elif event.button == 4:  # Mouse wheel up
                    self.logScroll = max(0, self.logScroll - 3)
                elif event.button == 5:  # Mouse wheel down
                    maxScroll = max(0, len(self.logMessages) - 10)
                    self.logScroll = min(maxScroll, self.logScroll + 3)
    
    def _screenToGridPos(self, screenPos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to grid coordinates."""
        x, y = screenPos
        
        # Check if click is within grid area (accounting for left panel offset)
        gridXOffset = self.logPanelWidth
        if (gridXOffset <= x < gridXOffset + self.gridPixelWidth and 
            y < self.gridPixelHeight):
            gridX = (x - gridXOffset) // self.cellSize
            gridY = y // self.cellSize
            return (gridX, gridY)
        
        return None
    
    def getLastClickPosition(self) -> Optional[Tuple[int, int]]:
        """Get the last clicked grid position."""
        return getattr(self, '_lastClickPos', None)
    
    def render(self, agents: List[Any], simulationStats: Dict[str, Any]) -> None:
        """
        Render the current state of the simulation.
        
        Args:
            agents: List of agent objects to render
            simulation_stats: Dictionary containing simulation statistics
        """
        # Clear screen
        self.screen.fill(GuiColors.BACKGROUND)
        # Render components
        self._renderLeftLogPanel()
        self._renderGrid() 
        self._renderAgents(agents)
        self._renderAgentInfoPanel(simulationStats)
        
        # Update display
        pygame.display.flip()
    
    def _renderGrid(self) -> None:
        """Render the simulation grid."""
        # Grid background - offset by log panel width
        gridXOffset = self.logPanelWidth
        gridRect = pygame.Rect(gridXOffset, 0, self.gridPixelWidth, self.gridPixelHeight)
        self.screen.fill(GuiColors.GRID_BACKGROUND, gridRect)
        
        # Grid lines
        for x in range(0, self.gridPixelWidth, self.cellSize):
            pygame.draw.line(self.screen, GuiColors.GRID_LINES, 
                           (x + gridXOffset, 0), (x + gridXOffset, self.gridPixelHeight))
        
        for y in range(0, self.gridPixelHeight, self.cellSize):
            pygame.draw.line(self.screen, GuiColors.GRID_LINES, 
                           (gridXOffset, y), (gridXOffset + self.gridPixelWidth, y))
    
    def _renderAgents(self, agents: List[Any]) -> None:
        """Render all agents on the grid."""
        for agent in agents:
            if hasattr(agent, 'alive') and agent.alive and hasattr(agent, 'pos') and agent.pos:self._renderAgent(agent)
    
    def _renderAgent(self, agent: Any) -> None:
        x, y = agent.pos
        if not (0 <= x < self.gridWidth and 0 <= y < self.gridHeight):return
        
        # Calculate screen position (offset by log panel)
        grid_x_offset = self.logPanelWidth
        screen_x = grid_x_offset + x * self.cellSize
        screen_y = y * self.cellSize
        center = (screen_x + self.cellSize // 2, screen_y + self.cellSize // 2)
        radius = self.cellSize // 3
        
        # Get agent color
        agent_color = self._getAgentColor(agent)
        
        # Draw main agent circle
        pygame.draw.circle(self.screen, agent_color, center, radius)
        
        # Draw agent label
        self._renderAgentLabel(agent, center)
        
        # Draw health indicator
        if hasattr(agent, 'health') and hasattr(agent, 'max_health'):
            health_ratio = agent.health / agent.max_health if agent.max_health > 0 else 0
            if health_ratio < 1.0:
                health_color = (255, int(255 * health_ratio), 0)
                pygame.draw.circle(self.screen, health_color, center, radius + 2, 2)
        
        # Draw behavior indicator
        if hasattr(agent, 'behavior'):
            behavior_color = self._getBehaviorColor(agent.behavior)
            pygame.draw.circle(self.screen, behavior_color, 
                             (center[0] + radius//2, center[1] - radius//2), 3)
        
        # Draw selection indicator
        if agent == self.selectedAgent:
            pygame.draw.circle(self.screen, GuiColors.SELECTION, center, radius + 6, 3)
        
        # Draw agent name for important agents
        if self._isImportantAgent(agent):
            name = getattr(agent, 'id', str(agent))[:3].upper()
            text = self.fontSmall.render(name, True, GuiColors.TEXT_PRIMARY)
            text_rect = text.get_rect(center=(center[0], center[1] + radius + 8))
            self.screen.blit(text, text_rect)
    
    def _getAgentColor(self, agent: Any) -> Tuple[int, int, int]:
        # First try to match by agent ID
        if hasattr(agent, 'id'):
            agent_id = str(agent.id).upper()
            if agent_id in self.agentColors:
                return self.agentColors[agent_id]
            
            # Handle partial matches for spawned agents
            for color_key in self.agentColors:
                if color_key in agent_id or agent_id.startswith(color_key):
                    return self.agentColors[color_key]
        
        # Then try to match by agentType enum (camelCase)
        if hasattr(agent, 'agentType'):
            # Try full enum string representation first
            agent_type_str = str(agent.agentType)
            if agent_type_str in self.agentColors:
                return self.agentColors[agent_type_str]
            
            # Try just the enum value
            if hasattr(agent.agentType, 'value'):
                agent_type_key = agent.agentType.value.upper()
                if agent_type_key in self.agentColors:
                    return self.agentColors[agent_type_key]
        
        # Fallback to legacy snake_case attribute if still exists
        if hasattr(agent, 'agent_type'):
            # Handle both enum types and string types
            if hasattr(agent.agent_type, 'value'):
                agent_type_key = agent.agent_type.value.upper()
            else:
                agent_type_key = str(agent.agent_type).upper()
            if agent_type_key in self.agentColors:
                return self.agentColors[agent_type_key]
        
        # Default colors by agent class
        if hasattr(agent, 'agent_type'):
            # Handle both enum types and string types
            if hasattr(agent.agent_type, 'value'):
                agent_type_str = str(agent.agent_type.value).upper()
            else:
                agent_type_str = str(agent.agent_type).upper()
            if 'PREDATOR' in agent_type_str:
                return GuiColors.PREDATOR_CLAN  # Default predator color
            elif 'MONSTER' in agent_type_str:
                if 'SMALL' in agent_type_str:
                    return GuiColors.MONSTER_SMALL
                elif 'LARGE' in agent_type_str:
                    return GuiColors.MONSTER_LARGE
                elif 'PACK' in agent_type_str:
                    return GuiColors.MONSTER_PACK
                else:
                    return GuiColors.MONSTER_MEDIUM
        
        # Final fallback colors by agent class name
        agent_class = type(agent).__name__
        default_colors = {
            'Predator': GuiColors.PREDATOR_CLAN,
            'Thia': GuiColors.THIA,
            'Boss': GuiColors.BOSS,
            'Monster': GuiColors.MONSTER_MEDIUM,
            'EvolutionaryMonster': GuiColors.MONSTER_EVOLVED,
        }
        
        return default_colors.get(agent_class, (200, 200, 200))
    
    def _renderAgentLabel(self, agent: Any, center: Tuple[int, int]) -> None:
        """Render agent label on the grid."""
        if not hasattr(agent, 'id'):
            return
        
        # Get agent label text
        label_text = str(agent.id)
        
        # Shorten long labels
        if len(label_text) > 8:
            label_text = label_text[:6] + ".."
        
        # Create text surface
        text_surface = self.fontSmall.render(label_text, True, GuiColors.TEXT_PRIMARY)
        text_rect = text_surface.get_rect()
        
        # Position label below the agent circle
        label_x = center[0] - text_rect.width // 2
        label_y = center[1] + self.cellSize // 3 + 2
        
        # Draw semi-transparent background for better readability
        bg_rect = pygame.Rect(label_x - 2, label_y - 1, text_rect.width + 4, text_rect.height + 2)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(180)
        bg_surface.fill((0, 0, 0))
        self.screen.blit(bg_surface, (bg_rect.x, bg_rect.y))
        
        # Draw the label text
        self.screen.blit(text_surface, (label_x, label_y))
    
    def _getBehaviorColor(self, behavior: Any) -> Tuple[int, int, int]:
        """Get color for behavior indicator."""
        if hasattr(behavior, 'value'):
            behavior_name = behavior.value.lower()
        else:
            behavior_name = str(behavior).lower()
        
        behavior_colors = {
            'idle': GuiColors.BEHAVIOR_IDLE,
            'hunting': GuiColors.BEHAVIOR_HUNTING,
            'fleeing': GuiColors.BEHAVIOR_FLEEING,
            'patrolling': GuiColors.BEHAVIOR_PATROLLING,
            'resting': GuiColors.BEHAVIOR_RESTING,
        }
        
        return behavior_colors.get(behavior_name, GuiColors.BEHAVIOR_IDLE)
    
    def _isImportantAgent(self, agent: Any) -> bool:
        """Check if agent should have its name displayed."""
        agent_type = type(agent).__name__
        if agent_type in ['Thia', 'Boss']:
            return True
        
        if agent_type == 'Predator' and hasattr(agent, 'role'):
            return agent.role.value in ['dek', 'father', 'brother']
        
        return False
    
    def _renderLeftLogPanel(self) -> None:
        """Render the log panel on the left side."""
        # Left panel background
        panel_rect = pygame.Rect(0, 0, self.logPanelWidth, self.gridPixelHeight)
        self.screen.fill(GuiColors.PANEL_BACKGROUND, panel_rect)
        
        # Panel title
        y_offset = 10
        x_offset = 10
        
        title = self.fontMedium.render("LOG MESSAGES", True, GuiColors.ACCENT)
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 30
        
        # Log messages display
        if self.logMessages:
            line_height = 16
            max_lines = (self.gridPixelHeight - 60) // line_height
            
            # Calculate visible messages
            start_idx = max(0, self.logScroll)
            end_idx = min(len(self.logMessages), start_idx + max_lines)
            
            for i in range(start_idx, end_idx):
                if y_offset >= self.gridPixelHeight - 30:
                    break
                
                log_entry = self.logMessages[i]
                
                # Format timestamp and message
                time_text = f"[{log_entry['timestamp']}]"
                message_text = log_entry['message']
                
                # Render timestamp
                time_surface = self.fontSmall.render(time_text, True, GuiColors.TEXT_SECONDARY)
                self.screen.blit(time_surface, (x_offset, y_offset))
                
                # Render message (wrapped if needed)
                max_chars = (self.logPanelWidth - 20) // 7
                if len(message_text) > max_chars:
                    message_text = message_text[:max_chars - 3] + "..."
                
                text_color = GuiColors.WARNING if log_entry['important'] else GuiColors.TEXT_PRIMARY
                msg_surface = self.fontSmall.render(message_text, True, text_color)
                self.screen.blit(msg_surface, (x_offset, y_offset + 12))
                
                y_offset += line_height * 2  # Double spacing for readability
        else:
            no_msg = self.fontSmall.render("No messages yet...", True, GuiColors.TEXT_SECONDARY)
            self.screen.blit(no_msg, (x_offset, y_offset))
        
        # Scroll hint at bottom
        hint = self.fontSmall.render("(Mouse wheel to scroll)", True, GuiColors.TEXT_SECONDARY)
        self.screen.blit(hint, (x_offset, self.gridPixelHeight - 25))
    
    def _renderAgentInfoPanel(self, stats: Dict[str, Any]) -> None:
        """Render the bottom agent statistics and info panel."""
        panel_rect = pygame.Rect(0, self.gridPixelHeight, 
                               self.screenWidth, self.agentInfoHeight)
        self.screen.fill(GuiColors.PANEL_BACKGROUND, panel_rect)
        
        y_offset = self.gridPixelHeight + 10
        left_x = 10
        middle_x = self.screenWidth // 3
        right_x = (self.screenWidth * 2) // 3
        
        # Simulation Status (Left Column)
        paused = stats.get('paused', False)
        status_title = self.fontMedium.render("SIMULATION STATUS", True, GuiColors.ACCENT)
        self.screen.blit(status_title, (left_x, y_offset))
        
        status_info = [
            f"Status: {'PAUSED' if paused else 'RUNNING'}",
            f"Step: {stats.get('step', 0)}",
            f"FPS: {stats.get('fps', 0.0):.1f}",
            f"Speed: {stats.get('fps_target', 8)}",
            f"Total Agents: {stats.get('agent_count', 0)}"
        ]
        
        for i, info in enumerate(status_info):
            color = GuiColors.WARNING if paused and i == 0 else GuiColors.TEXT_PRIMARY
            text = self.fontSmall.render(info, True, color)
            self.screen.blit(text, (left_x, y_offset + 25 + i * 15))
        
        # Agent Types (Middle Column)
        types_title = self.fontMedium.render("AGENT TYPES", True, GuiColors.ACCENT)
        self.screen.blit(types_title, (middle_x, y_offset))
        
        type_counts = stats.get('agent_types', {})
        y_pos = y_offset + 25
        for agent_type, count in sorted(type_counts.items()):
            if y_pos > self.gridPixelHeight + self.agentInfoHeight - 20:
                break
            text = self.fontSmall.render(f"{agent_type}: {count}", True, GuiColors.TEXT_PRIMARY)
            self.screen.blit(text, (middle_x, y_pos))
            y_pos += 15
        
        # Selected Agent Info (Right Column)
        selected_title = self.fontMedium.render("SELECTED AGENT", True, GuiColors.ACCENT)
        self.screen.blit(selected_title, (right_x, y_offset))
        
        if self.selectedAgent:
            agent_info = [
                f"ID: {getattr(self.selectedAgent, 'id', 'Unknown')}",
                f"Type: {getattr(self.selectedAgent, 'agentType', 'Unknown')}",
                f"Position: {getattr(self.selectedAgent, 'pos', 'Unknown')}",
            ]
            
            if hasattr(self.selectedAgent, 'health'):
                agent_info.append(f"Health: {self.selectedAgent.health}/{getattr(self.selectedAgent, 'maxHealth', '?')}")
            
            if hasattr(self.selectedAgent, 'behavior'):
                agent_info.append(f"Behavior: {self.selectedAgent.behavior}")
            
            if hasattr(self.selectedAgent, 'alliance'):
                agent_info.append(f"Alliance: {self.selectedAgent.alliance}")
                
            for i, info in enumerate(agent_info):
                if y_offset + 25 + i * 15 > self.gridPixelHeight + self.agentInfoHeight - 20:
                    break
                text = self.fontSmall.render(info, True, GuiColors.TEXT_PRIMARY)
                self.screen.blit(text, (right_x, y_offset + 25 + i * 15))
        else:
            no_selection = self.fontSmall.render("Click an agent to view info", True, GuiColors.TEXT_SECONDARY)
            self.screen.blit(no_selection, (right_x, y_offset + 25))
        
        # Controls at the very bottom
        controls_y = self.gridPixelHeight + self.agentInfoHeight - 15
        controls_text = "SPACE: Pause/Resume | R: Reset | +/-: Speed | L: Toggle Labels | ESC: Exit"
        controls = self.fontSmall.render(controls_text, True, GuiColors.TEXT_SECONDARY)
        self.screen.blit(controls, (left_x, controls_y))
    
    def setSelectedAgent(self, agent: Any) -> None:
        """Set the currently selected agent."""
        self.selectedAgent = agent
    

    
    def tick(self, fps: int) -> None:
        """Control the frame rate."""
        self.clock.tick(fps)
    
    def addLogMessage(self, message: str, important: bool = False) -> None:
        """Add a message to the log display."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'important': important
        }
        
        self.logMessages.append(log_entry)
        
        # Keep only recent messages
        if len(self.logMessages) > self.maxLogMessages:
            self.logMessages.pop(0)
        
        # Auto-scroll to bottom when new message arrives
        if len(self.logMessages) > 15:
            self.logScroll = max(0, len(self.logMessages) - 15)
    
    def quit(self) -> None:
        """Clean up and quit pygame."""
        pygame.quit()
        self._closed = True