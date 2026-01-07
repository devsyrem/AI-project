"""
Agent module for Predator: Badlands simulation.
Contains all agent classes: base Agent, Predator, Monster, Thia, and Boss.
"""

from agent import Agent, ActionType, AgentStats
from predator import Predator, PredatorRole, Trophy, BanditArm
from monster import Monster, MonsterType
from thia import Thia, ReconData
from boss import Boss, BossMode, AttackPattern

__all__ = [
    "Agent", "ActionType", "AgentStats",
    "Predator", "PredatorRole", "Trophy", "BanditArm", 
    "Monster", "MonsterType",
    "Thia", "ReconData",
    "Boss", "BossMode", "AttackPattern"
]