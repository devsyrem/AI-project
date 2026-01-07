# Quick fix for remaining underscore references
import re

def fix_file_references(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix double underscore references
    content = content.replace('self.__ocean', 'self._ocean')
    content = content.replace('self.__agents', 'self._agents')
    content = content.replace('self.__gui', 'self._gui')
    content = content.replace('self.__is_running', 'self._isRunning')
    content = content.replace('self.__paused', 'self._paused')
    content = content.replace('self.__step_count', 'self._stepCount')
    content = content.replace('self.__fps_target', 'self._fpsTarget')
    content = content.replace('self.__storm_active', 'self._stormActive')
    content = content.replace('self.__storm_duration', 'self._stormDuration')
    content = content.replace('self.__resource_scarcity', 'self._resourceScarcity')
    content = content.replace('self.__last_event_time', 'self._lastEventTime')
    content = content.replace('self.__event_cooldown', 'self._eventCooldown')
    
    # Fix method calls that still have underscores
    replacements = [
        ('get_random_position()', 'getRandomPosition()'),
        ('place_agent(', 'placeAgent('),
        ('remove_agent(', 'removeAgent('),
        ('get_agent(', 'getAgent('),
        ('get_neighbors(', 'getNeighbors('),
        ('get_empty_neighbors(', 'getEmptyNeighbors('),
        ('is_empty(', 'isEmpty('),
        ('target_pos', 'targetPos'),
        ('move_cooldown', 'moveCooldown'),
        ('agent_type=', 'agentType='),
        ('max_health=', 'maxHealth='),
        ('move_cooldown=', 'moveCooldown='),
        ('log_panel_width', 'logPanelWidth'),
        ('grid_pixel_width', 'gridPixelWidth'),
        ('grid_pixel_height', 'gridPixelHeight'),
        ('agent_info_height', 'agentInfoHeight'),
        ('cell_size', 'cellSize'),
        ('font_large', 'fontLarge'),
        ('font_medium', 'fontMedium'),
        ('font_small', 'fontSmall'),
        ('screen_width', 'screenWidth'),
        ('screen_height', 'screenHeight'),
        ('self._render_agent(', 'self._renderAgent('),
        ('self._render_agent_label(', 'self._renderAgentLabel('),
        ('self._get_agent_color(', 'self._getAgentColor('),
        ('self._is_important_agent(', 'self._isImportantAgent('),
        ('grid_width', 'gridWidth'),
        ('grid_height', 'gridHeight'),
        ('agent_colors', 'agentColors'),
        ('self._get_behavior_color(', 'self._getBehaviorColor('),
        ('selected_agent', 'selectedAgent'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed references in {filename}")

# Fix the main files
fix_file_references('gui_simulator.py')
fix_file_references('view/gui.py')
print("All references fixed!")