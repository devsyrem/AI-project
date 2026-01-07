"""
Genetic Algorithm system for evolving monster populations.
Implements breeding, mutation, fitness evaluation, and population dynamics.
"""
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class GeneType(Enum):
    """Types of genetic traits."""
    AGGRESSION = "aggression"      # How likely to attack
    SPEED = "speed"               # Movement speed modifier
    INTELLIGENCE = "intelligence"  # Learning and decision-making ability
    STAMINA = "stamina"           # Energy and endurance
    PACK_AFFINITY = "pack_affinity"  # Tendency to form groups
    TERRITORY_DRIVE = "territory_drive"  # Territorial behavior strength
    FEAR_THRESHOLD = "fear_threshold"    # Threshold for fleeing
    REPRODUCTION_RATE = "reproduction_rate"  # Breeding frequency
    ADAPTABILITY = "adaptability"  # How quickly traits can change


@dataclass
class GeneticTrait:
    """Represents a single genetic trait."""
    gene_type: GeneType
    value: float  # Normalized 0.0 to 1.0
    dominance: float  # How dominant this trait is (0.0 to 1.0)
    mutation_rate: float = 0.05  # Chance of mutation per generation
    
    def mutate(self, mutation_strength: float = 0.1) -> None:
        """Apply random mutation to the trait."""
        if random.random() < self.mutation_rate:
            # Small random change
            change = random.gauss(0, mutation_strength)
            self.value = max(0.0, min(1.0, self.value + change))
            
            # Occasionally mutate dominance
            if random.random() < self.mutation_rate * 0.3:
                dom_change = random.gauss(0, mutation_strength * 0.5)
                self.dominance = max(0.0, min(1.0, self.dominance + dom_change))


@dataclass
class Genome:
    """Represents the complete genetic makeup of a monster."""
    traits: Dict[GeneType, GeneticTrait] = field(default_factory=dict)
    generation: int = 0
    parents: Tuple[Optional[int], Optional[int]] = (None, None)  # Parent IDs
    fitness_score: float = 0.0
    age: int = 0
    
    def __post_init__(self):
        """Initialize with default traits if empty."""
        if not self.traits:
            self._initialize_random_traits()
    
    def _initialize_random_traits(self) -> None:
        """Initialize with random trait values."""
        for gene_type in GeneType:
            self.traits[gene_type] = GeneticTrait(
                gene_type=gene_type,
                value=random.uniform(0.2, 0.8),  # Avoid extremes initially
                dominance=random.uniform(0.3, 0.7),
                mutation_rate=random.uniform(0.02, 0.08)
            )
    
    def get_trait_value(self, gene_type: GeneType) -> float:
        """Get the value of a specific trait."""
        return self.traits.get(gene_type, GeneticTrait(gene_type, 0.5, 0.5)).value
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """Create offspring through genetic crossover."""
        offspring = Genome(generation=max(self.generation, other.generation) + 1)
        offspring.parents = (id(self), id(other))
        
        for gene_type in GeneType:
            parent1_trait = self.traits.get(gene_type)
            parent2_trait = other.traits.get(gene_type)
            
            if parent1_trait and parent2_trait:
                # Determine which parent's trait is more dominant
                if parent1_trait.dominance > parent2_trait.dominance:
                    # Inherit from parent 1 with some blending
                    blend_factor = random.uniform(0.7, 1.0)
                    new_value = (parent1_trait.value * blend_factor + 
                               parent2_trait.value * (1 - blend_factor))
                    new_dominance = parent1_trait.dominance
                else:
                    # Inherit from parent 2 with some blending
                    blend_factor = random.uniform(0.7, 1.0)
                    new_value = (parent2_trait.value * blend_factor + 
                               parent1_trait.value * (1 - blend_factor))
                    new_dominance = parent2_trait.dominance
                
                # Average mutation rates
                new_mutation_rate = (parent1_trait.mutation_rate + 
                                   parent2_trait.mutation_rate) / 2
                
                offspring.traits[gene_type] = GeneticTrait(
                    gene_type=gene_type,
                    value=max(0.0, min(1.0, new_value)),
                    dominance=new_dominance,
                    mutation_rate=new_mutation_rate
                )
            elif parent1_trait:
                offspring.traits[gene_type] = parent1_trait
            elif parent2_trait:
                offspring.traits[gene_type] = parent2_trait
        
        # Apply mutations
        for trait in offspring.traits.values():
            trait.mutate()
        
        return offspring
    
    def calculate_fitness(self, survival_time: int, damage_dealt: int, 
                         offspring_count: int, territory_held: float) -> float:
        """Calculate fitness score based on performance metrics."""
        # Base fitness from survival
        fitness = survival_time * 0.1
        
        # Combat effectiveness
        fitness += damage_dealt * 0.05
        
        # Reproductive success (most important for evolution)
        fitness += offspring_count * 2.0
        
        # Territory control
        fitness += territory_held * 0.5
        
        # Bonus for balanced traits (avoid extreme specialization early)
        trait_variance = np.var([trait.value for trait in self.traits.values()])
        if trait_variance < 0.1:  # Well-balanced
            fitness += 1.0
        
        self.fitness_score = fitness
        return fitness


class PopulationDynamics:
    """Manages population growth, breeding, and environmental pressure."""
    
    def __init__(self, max_population: int = 100, carrying_capacity: int = 150):
        self.max_population = max_population
        self.carrying_capacity = carrying_capacity
        self.population: Dict[int, Genome] = {}
        self.generation_count = 0
        self.next_id = 1
        
        # Environmental pressure parameters
        self.resource_scarcity = 0.0  # 0.0 = abundant, 1.0 = scarce
        self.predation_pressure = 0.0  # Pressure from predators
        self.territory_competition = 0.0  # Competition for territory
        
        # Breeding parameters
        self.breeding_season_frequency = 10  # Every N steps
        self.minimum_breeding_age = 5
        self.maximum_age = 100
        self.fertility_rate = 0.3  # Base chance of successful breeding
    
    def add_individual(self, genome: Optional[Genome] = None) -> int:
        """Add individual to population."""
        if len(self.population) >= self.carrying_capacity:
            return -1  # Population at capacity
        
        individual_id = self.next_id
        self.next_id += 1
        
        if genome is None:
            genome = Genome()
        
        self.population[individual_id] = genome
        return individual_id
    
    def remove_individual(self, individual_id: int) -> bool:
        """Remove individual from population."""
        if individual_id in self.population:
            del self.population[individual_id]
            return True
        return False
    
    def update_population(self, step: int) -> Dict[str, any]:
        """Update population dynamics each simulation step."""
        updates = {
            'births': 0,
            'deaths': 0,
            'mutations': 0,
            'population_size': len(self.population)
        }
        
        # Age all individuals
        for genome in self.population.values():
            genome.age += 1
        
        # Natural death from old age
        deaths = []
        for individual_id, genome in self.population.items():
            death_probability = self._calculate_death_probability(genome)
            if random.random() < death_probability:
                deaths.append(individual_id)
        
        for individual_id in deaths:
            self.remove_individual(individual_id)
            updates['deaths'] += 1
        
        # Breeding season
        if step % self.breeding_season_frequency == 0:
            offspring = self._breeding_season()
            updates['births'] = len(offspring)
            
            for child_genome in offspring:
                if self.add_individual(child_genome) != -1:
                    # Count mutations in offspring
                    for trait in child_genome.traits.values():
                        if hasattr(trait, '_mutated') and trait._mutated:
                            updates['mutations'] += 1
        
        # Environmental pressure effects
        if len(self.population) > self.max_population:
            self._apply_environmental_pressure()
        
        updates['population_size'] = len(self.population)
        return updates
    
    def _calculate_death_probability(self, genome: Genome) -> float:
        """Calculate probability of death based on age and traits."""
        base_death_rate = 0.01
        
        # Age-based mortality
        if genome.age > self.maximum_age * 0.8:
            age_factor = (genome.age - self.maximum_age * 0.8) / (self.maximum_age * 0.2)
            base_death_rate += age_factor * 0.1
        
        # Environmental factors
        stamina = genome.get_trait_value(GeneType.STAMINA)
        adaptability = genome.get_trait_value(GeneType.ADAPTABILITY)
        
        # Low stamina increases death rate
        if stamina < 0.3:
            base_death_rate += (0.3 - stamina) * 0.05
        
        # Low adaptability in harsh environment
        environmental_stress = (self.resource_scarcity + self.predation_pressure) / 2
        if adaptability < environmental_stress:
            base_death_rate += (environmental_stress - adaptability) * 0.03
        
        return min(base_death_rate, 0.5)  # Cap at 50%
    
    def _breeding_season(self) -> List[Genome]:
        """Conduct breeding season, return new offspring."""
        offspring = []
        
        # Get breeding-eligible individuals
        breeders = [
            (id, genome) for id, genome in self.population.items()
            if genome.age >= self.minimum_breeding_age
        ]
        
        if len(breeders) < 2:
            return offspring
        
        # Calculate breeding pairs based on fitness and compatibility
        breeding_pairs = self._select_breeding_pairs(breeders)
        
        for parent1_id, parent2_id in breeding_pairs:
            parent1 = self.population[parent1_id]
            parent2 = self.population[parent2_id]
            
            # Check breeding success
            fertility1 = parent1.get_trait_value(GeneType.REPRODUCTION_RATE)
            fertility2 = parent2.get_trait_value(GeneType.REPRODUCTION_RATE)
            combined_fertility = (fertility1 + fertility2) / 2
            
            breeding_success_rate = self.fertility_rate * combined_fertility
            
            # Environmental factors affect breeding
            if self.resource_scarcity > 0.5:
                breeding_success_rate *= (1 - self.resource_scarcity * 0.5)
            
            if random.random() < breeding_success_rate:
                child = parent1.crossover(parent2)
                offspring.append(child)
                
                # Possible multiple offspring for highly fertile parents
                if combined_fertility > 0.8 and random.random() < 0.3:
                    child2 = parent1.crossover(parent2)
                    offspring.append(child2)
        
        return offspring
    
    def _select_breeding_pairs(self, breeders: List[Tuple[int, Genome]]) -> List[Tuple[int, int]]:
        """Select breeding pairs based on fitness and genetic compatibility."""
        pairs = []
        
        # Sort by fitness
        breeders_by_fitness = sorted(breeders, key=lambda x: x[1].fitness_score, reverse=True)
        
        # Pair selection with bias towards fit individuals
        available = list(range(len(breeders_by_fitness)))
        
        while len(available) >= 2:
            # Select first parent (bias towards fittest)
            parent1_idx = self._weighted_selection(available, breeders_by_fitness)
            available.remove(parent1_idx)
            
            if not available:
                break
            
            # Select second parent (less bias, promote genetic diversity)
            parent2_idx = random.choice(available)
            available.remove(parent2_idx)
            
            parent1_id = breeders_by_fitness[parent1_idx][0]
            parent2_id = breeders_by_fitness[parent2_idx][0]
            
            pairs.append((parent1_id, parent2_id))
        
        return pairs
    
    def _weighted_selection(self, available: List[int], 
                          breeders: List[Tuple[int, Genome]]) -> int:
        """Select individual with fitness-based weighting."""
        # Higher fitness = higher selection probability
        weights = []
        for idx in available:
            fitness = breeders[idx][1].fitness_score
            # Add small base weight to avoid zero probability
            weight = max(0.1, fitness)
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available)
        
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return available[i]
        
        return available[-1]  # Fallback
    
    def _apply_environmental_pressure(self) -> None:
        """Apply environmental pressure when population exceeds capacity."""
        excess = len(self.population) - self.max_population
        if excess <= 0:
            return
        
        # Select individuals for removal based on fitness
        individuals = list(self.population.items())
        individuals.sort(key=lambda x: x[1].fitness_score)  # Lowest fitness first
        
        # Remove weakest individuals
        for i in range(min(excess, len(individuals))):
            individual_id = individuals[i][0]
            self.remove_individual(individual_id)
    
    def update_environmental_conditions(self, resource_level: float, 
                                      predator_count: int, territory_pressure: float) -> None:
        """Update environmental conditions affecting population."""
        # Resource scarcity (inverted from resource level)
        self.resource_scarcity = max(0.0, min(1.0, 1.0 - resource_level))
        
        # Predation pressure based on predator count
        max_predators = 20  # Assumed maximum for scaling
        self.predation_pressure = min(1.0, predator_count / max_predators)
        
        # Territory competition
        self.territory_competition = max(0.0, min(1.0, territory_pressure))
    
    def get_population_statistics(self) -> Dict[str, any]:
        """Get comprehensive population statistics."""
        if not self.population:
            return {
                'size': 0,
                'average_age': 0,
                'average_fitness': 0,
                'trait_averages': {},
                'generation_distribution': {}
            }
        
        total_age = sum(genome.age for genome in self.population.values())
        total_fitness = sum(genome.fitness_score for genome in self.population.values())
        
        # Calculate average trait values
        trait_sums = {gene_type: 0.0 for gene_type in GeneType}
        trait_counts = {gene_type: 0 for gene_type in GeneType}
        
        generation_counts = {}
        
        for genome in self.population.values():
            # Generation tracking
            gen = genome.generation
            generation_counts[gen] = generation_counts.get(gen, 0) + 1
            
            # Trait averaging
            for gene_type, trait in genome.traits.items():
                trait_sums[gene_type] += trait.value
                trait_counts[gene_type] += 1
        
        trait_averages = {
            gene_type.value: trait_sums[gene_type] / trait_counts[gene_type] 
            if trait_counts[gene_type] > 0 else 0.0
            for gene_type in GeneType
        }
        
        return {
            'size': len(self.population),
            'average_age': total_age / len(self.population),
            'average_fitness': total_fitness / len(self.population),
            'trait_averages': trait_averages,
            'generation_distribution': generation_counts,
            'environmental_pressure': {
                'resource_scarcity': self.resource_scarcity,
                'predation_pressure': self.predation_pressure,
                'territory_competition': self.territory_competition
            }
        }
    
    def get_elite_individuals(self, count: int = 5) -> List[Tuple[int, Genome]]:
        """Get the most fit individuals in the population."""
        individuals = list(self.population.items())
        individuals.sort(key=lambda x: x[1].fitness_score, reverse=True)
        return individuals[:count]


class GeneticEvolutionEngine:
    """Main engine coordinating genetic evolution and population dynamics."""
    
    def __init__(self, initial_population: int = 50, max_population: int = 100):
        self.population_dynamics = PopulationDynamics(max_population, max_population + 50)
        
        # Initialize starting population
        for _ in range(initial_population):
            self.population_dynamics.add_individual()
    
    def update(self, step: int, environment_data: Dict[str, any]) -> Dict[str, any]:
        """Update genetic evolution system."""
        # Update environmental conditions
        self.population_dynamics.update_environmental_conditions(
            resource_level=environment_data.get('resource_level', 0.5),
            predator_count=environment_data.get('predator_count', 0),
            territory_pressure=environment_data.get('territory_pressure', 0.0)
        )
        
        # Update population dynamics
        population_updates = self.population_dynamics.update_population(step)
        
        # Get statistics
        stats = self.population_dynamics.get_population_statistics()
        
        return {
            'population_updates': population_updates,
            'population_stats': stats
        }
    
    def get_individual_genome(self, individual_id: int) -> Optional[Genome]:
        """Get genome of specific individual."""
        return self.population_dynamics.population.get(individual_id)
    
    def update_fitness(self, individual_id: int, survival_time: int, 
                      damage_dealt: int, offspring_count: int, territory_held: float) -> bool:
        """Update fitness score for an individual."""
        genome = self.get_individual_genome(individual_id)
        if genome:
            genome.calculate_fitness(survival_time, damage_dealt, offspring_count, territory_held)
            return True
        return False
    
    def create_evolved_monster_traits(self, individual_id: int) -> Dict[str, float]:
        """Create monster traits based on evolved genome."""
        genome = self.get_individual_genome(individual_id)
        if not genome:
            return {}
        
        return {
            'aggression_modifier': genome.get_trait_value(GeneType.AGGRESSION),
            'speed_modifier': genome.get_trait_value(GeneType.SPEED),
            'intelligence_modifier': genome.get_trait_value(GeneType.INTELLIGENCE),
            'stamina_modifier': genome.get_trait_value(GeneType.STAMINA),
            'pack_affinity': genome.get_trait_value(GeneType.PACK_AFFINITY),
            'territory_drive': genome.get_trait_value(GeneType.TERRITORY_DRIVE),
            'fear_threshold': genome.get_trait_value(GeneType.FEAR_THRESHOLD),
            'reproduction_drive': genome.get_trait_value(GeneType.REPRODUCTION_RATE),
            'adaptability': genome.get_trait_value(GeneType.ADAPTABILITY)
        }