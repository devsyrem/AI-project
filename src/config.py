"""
Configuration module for Predator: Badlands simulation.
Contains all tunable constants, presets, and game parameters.
"""
from typing import Dict, Any
import logging

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Base simulation parameters
DEFAULT_SEED = 42
DEFAULT_GRID_SIZE = 20
DEFAULT_MAX_STEPS = 400
WRAP_GRID = True

# Agent spawn parameters
DEFAULT_MONSTERS = 30
DEFAULT_PREDATORS = 3  # Dek + father + brother
DEFAULT_BOSS_COUNT = 1
DEFAULT_THIA_COUNT = 1

# Movement and stamina
MOVE_STAMINA_COST = 1
CARRY_STAMINA_MULTIPLIER = 1.5
REST_STAMINA_GAIN = 10
MAX_STAMINA = 100
STAMINA_REGEN_RATE = 2  # per step when not moving

# Health and combat
DEFAULT_HEALTH = 100
PREDATOR_ATTACK_POWER = 25
MONSTER_ATTACK_POWER = 15
BOSS_ATTACK_POWER = 40
HEALING_RATE = 5  # when resting

# Boss parameters
BOSS_HEALTH = 200
BOSS_TERRITORY_SIZE = 3
BOSS_ADAPTATION_DECAY = 0.9  # how quickly boss forgets patterns
BOSS_COUNTER_THRESHOLD = 3  # how many times to see pattern before adapting

# Honor and reputation system
MONSTER_KILL_HONOR = 10
BOSS_DAMAGE_HONOR = 5  # per damage point
THIA_REPAIR_HONOR = 20
CLAN_CHALLENGE_HONOR = 15
DEATH_HONOR_PENALTY = -50

# Thia parameters
THIA_REPAIR_TIME = 5  # steps to repair when carried
THIA_REPAIR_STAMINA_COST = 20
THIA_RECONNAISSANCE_RANGE = 8
THIA_BOSS_HINT_ACCURACY = 0.8

# Learning parameters
EPSILON_START = 0.3
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BANDIT_LEARNING_RATE = 0.1
REWARD_DISCOUNT = 0.9

# Procedural generation
TRAP_DENSITY = 0.05  # percentage of grid cells with traps
TRAP_DAMAGE = 20
HAZARD_SPAWN_INTERVAL = 50  # steps between new hazard spawns
DIFFICULTY_ESCALATION = 1.1  # multiplier for hazard intensity over time

# Vision and detection ranges
PREDATOR_VISION_RANGE = 5
MONSTER_VISION_RANGE = 3
BOSS_VISION_RANGE = 8

# Configuration presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "grid_size": 20,
        "monsters": 10,
        "boss_health": 80,
        "max_steps": 200,
        "trap_density": 0.02,
        "hazard_spawn_interval": 100,
        "learning_enabled": False
    },
    "standard": {
        "grid_size": 20,
        "monsters": 30,
        "boss_health": 200,
        "max_steps": 400,
        "trap_density": 0.05,
        "hazard_spawn_interval": 50,
        "learning_enabled": True
    },
    "expert": {
        "grid_size": 30,
        "monsters": 50,
        "boss_health": 300,
        "max_steps": 600,
        "trap_density": 0.08,
        "hazard_spawn_interval": 30,
        "learning_enabled": True,
        "boss_adaptation_enabled": True,
        "dynamic_hazards": True
    }
}

def get_config(preset_name: str = "standard") -> Dict[str, Any]:
    """
    Get configuration dictionary for specified preset.
    
    Args:
        preset_name: Name of configuration preset
        
    Returns:
        Dictionary containing all configuration parameters
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    # Start with defaults
    config = {
        "seed": DEFAULT_SEED,
        "grid_size": DEFAULT_GRID_SIZE,
        "max_steps": DEFAULT_MAX_STEPS,
        "wrap_grid": WRAP_GRID,
        "monsters": DEFAULT_MONSTERS,
        "predators": DEFAULT_PREDATORS,
        "boss_count": DEFAULT_BOSS_COUNT,
        "thia_count": DEFAULT_THIA_COUNT,
        "boss_health": BOSS_HEALTH,
        "trap_density": TRAP_DENSITY,
        "hazard_spawn_interval": HAZARD_SPAWN_INTERVAL,
        "learning_enabled": True,
        "boss_adaptation_enabled": True,
        "dynamic_hazards": True,
        "log_level": LOG_LEVEL
    }
    
    # Override with preset values
    config.update(PRESETS[preset_name])
    
    return config

def setup_logging(level: int = LOG_LEVEL) -> None:
    """Configure logging for the simulation."""
    logging.basicConfig(level=level, format=LOG_FORMAT)