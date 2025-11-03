"""
Evolution-Based Feature Optimization Engine
Implements bio-inspired evolutionary algorithm for optimal feature selection
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random


class EvolutionOptimizer:
    """
    Bio-inspired evolutionary optimizer for feature selection
    Uses genetic operations (selection, crossover, mutation) to find optimal feature subset
    """
    
    def __init__(self, X, y, population_size=50, generations=30, mutation_rate=0.1, crossover_rate=0.8, callback=None):
        """
        Initialize the evolutionary optimizer
        
        Args:
            X: Feature matrix (numpy array)
            y: Target vector (numpy array)
            population_size: Number of individuals in population
            generations: Number of evolution cycles
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of parent crossover
            callback: Optional callback function for progress updates
        """
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.callback = callback
        
    def generate_individual(self):
        """
        Generate random individual (chromosome) representing feature selection
        Each gene is binary: 1 = feature selected, 0 = feature excluded
        
        Returns:
            List of binary values representing feature selection
        """
        n_features = self.X.shape[1]
        # Select 30-70% of features randomly for diversity
        n_selected = random.randint(int(n_features * 0.3), int(n_features * 0.7))
        selected = random.sample(range(n_features), n_selected)
        chromosome = [0] * n_features
        for idx in selected:
            chromosome[idx] = 1
        return chromosome
    
    def evaluate_fitness(self, chromosome):
        """
        Evaluate fitness of an individual using cross-validated accuracy
        Applies penalty for excessive feature usage
        
        Args:
            chromosome: Binary list representing feature selection
            
        Returns:
            Fitness score (higher is better)
        """
        selected_features = [i for i, bit in enumerate(chromosome) if bit == 1]
        
        # Empty selection has zero fitness
        if len(selected_features) == 0:
            return 0
        
        X_selected = self.X_scaled[:, selected_features]
        # Single-threaded execution to avoid multiprocessing conflicts
        model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
        
        try:
            # Cross-validation with single thread
            scores = cross_val_score(model, X_selected, self.y, cv=5, scoring='accuracy', n_jobs=1)
            accuracy = scores.mean()
            # Penalty for using too many features (encourages parsimony)
            penalty = len(selected_features) / self.X.shape[1] * 0.1
            return accuracy - penalty
        except:
            return 0
    
    def crossover_operation(self, parent1, parent2):
        """
        Perform single-point crossover between two parents
        Swaps genetic material at random cut point
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Random crossover point
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutation_operation(self, chromosome):
        """
        Apply random bit-flip mutation to chromosome
        Each gene has probability mutation_rate of flipping
        
        Args:
            chromosome: Individual to mutate
            
        Returns:
            Mutated chromosome
        """
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip bit
        return chromosome
    
    def tournament_selection(self, population, fitness_scores):
        """
        Select parents using roulette wheel selection
        Higher fitness individuals have higher probability of selection
        
        Args:
            population: List of chromosomes
            fitness_scores: List of fitness values
            
        Returns:
            Tuple of two selected parents
        """
        # Normalize fitness scores to positive probabilities
        max_fitness = max(fitness_scores)
        fitness_scores = [f + abs(min(fitness_scores)) + 1 for f in fitness_scores]
        total = sum(fitness_scores)
        probabilities = [f / total for f in fitness_scores]
        
        # Roulette wheel selection
        parent1 = np.random.choice(len(population), p=probabilities)
        parent2 = np.random.choice(len(population), p=probabilities)
        return population[parent1], population[parent2]
    
    def evolve(self):
        """
        Execute the evolutionary optimization process
        Iterates through generations, evolving population toward optimal solution
        
        Returns:
            Tuple: (selected_features, best_fitness, evolution_history)
        """
        # Initialize random population
        population = [self.generate_individual() for _ in range(self.population_size)]
        best_chromosome = None
        best_fitness = -1
        evolution_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = [self.evaluate_fitness(chrom) for chrom in population]
            
            # Track best individual
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()
            
            # Record generation statistics
            evolution_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'feature_count': sum(best_chromosome) if best_chromosome else 0
            })
            
            # Broadcast progress updates if callback provided
            if self.callback:
                progress = (generation + 1) / self.generations * 100
                self.callback({
                    'type': 'progress',
                    'generation': generation + 1,
                    'total_generations': self.generations,
                    'progress': progress,
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean(fitness_scores),
                    'feature_count': sum(best_chromosome) if best_chromosome else 0
                })
            
            # Generate next generation
            new_population = [best_chromosome.copy()]  # Elitism: preserve best
            
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.tournament_selection(population, fitness_scores)
                # Crossover
                child1, child2 = self.crossover_operation(parent1, parent2)
                # Mutation
                child1 = self.mutation_operation(child1)
                child2 = self.mutation_operation(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Extract final selected features
        selected_features = [i for i, bit in enumerate(best_chromosome) if bit == 1]
        return selected_features, best_fitness, evolution_history

