import random
import math
import numpy as np
import matplotlib.pyplot as plt
from HallOfFame import * 

# Defining parameters for the evolutionary algorithm
POPULATION_SIZE = 300   # Number of individuals in the population
GENERATIONS = 3000      # Number of generations per iteration
MUTATION_RATE = 0.3     # Probability of mutation
OFFSPRINGS = 300        # Number of offspring generated per generation
ITERATIONS = 15         # Number of independent runs
ELITISM_SIZE = 20       # Number of elite individuals preserved per generation

def read_tsp_data(filename): #Reads TSP data from a file and returns a dictionary with node coordinates.
    tsp_data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        coord_section_index = lines.index('NODE_COORD_SECTION\n') # Finding start of coordinates
        for line in lines[coord_section_index + 1:]:
            if line.strip() == 'EOF': # Stop reading at the EOF marker
                break
            node, x, y = line.strip().split(' ')
            tsp_data[int(node)] = (float(x), float(y)) # Storing coordinates as a dictionary
    # print("/")
    return tsp_data

def fitness_function(chromosome, tsp_data): #Calculates the total distance of the tour represented by a chromosome.
    fitness = 0
    for i in range(len(chromosome)-1):
        x1, y1 = tsp_data[chromosome[i]]
        x2, y2 = tsp_data[chromosome[i+1]]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # Euclidean distance calculation
        fitness += distance
    # print("/")
    return fitness

# def initialize_population(tsp_data):
    # population = []
    # for _ in range(POPULATION_SIZE):
    #     values = list(tsp_data.keys())
    #     random.shuffle(values)
    #     population.append(values)
    # # print("/")
    # return population
    
def initialize_population(tsp_data): #Generates an initial random population of chromosomes.
    return [random.sample(list(tsp_data.keys()), len(tsp_data)) for _ in range(POPULATION_SIZE)]


# def crossover(parent1, parent2):
#     offspring = [None] * len(parent1)
#     mid = len(parent1) // 2
#     start = random.randint(1, mid - 2)
#     end = start + mid
#     offspring[start:end] = parent1[start:end]

#     indexB = indexOffspring = end
#     while None in offspring:
#         if parent2[indexB] not in offspring:
#             offspring[indexOffspring] = parent2[indexB]
#             indexOffspring = (indexOffspring + 1) % len(parent2)
#         indexB = (indexB + 1) % len(parent2)
#     # print("/")
#     return offspring
def crossover(parent1, parent2): #Performs order-based crossover to generate offspring from two parents.
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2)) # Selecting crossover segment
    offspring = [None] * size
    offspring[start:end] = parent1[start:end] #Copying segment from first parent

    mapping = {parent1[i]: parent2[i] for i in range(start, end)} # Creating mapping from parent1 to parent2
    for i in range(size):
        if i < start or i >= end:
            candidate = parent2[i]
            while candidate in mapping:
                candidate = mapping[candidate] # Resolving conflicts using mapping
            offspring[i] = candidate
    return offspring

# def mutate(chromosome):
#     # if random.random() < MUTATION_RATE:
#     #     i, j = random.sample(range(len(chromosome)), 2)
#     #     chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
#     # return chromosome
#     # size = len(chromosome)
#     # i, j = sorted(random.sample(range(size), 2))
#     # chromosome[i:j] = reversed(chromosome[i:j])
#     # return chromosome
#     # if random.random() < MUTATION_RATE:
#     #     size = len(chromosome)
#     #     i, j = sorted(random.sample(range(size), 2))
#     #     chromosome[i:j] = reversed(chromosome[i:j])
#     # return chromosome

#swapping mutation:
# def mutate(chromosome, mutationRate):
#     if random.uniform(0, 1) < mutationRate:
#         pos1 = random.randint(1, len(chromosome) - 2)
#         pos2 = pos1
#         while pos1 == pos2:
#             pos2 = random.randint(1, len(chromosome) - 2)
#         chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
#     # print("/")
#     return chromosome

# Performs a 2-opt mutation by reversing a segment of the chromosome:
def mutate(chromosome):
    size = len(chromosome)
    i, j = sorted(random.sample(range(size), 2)) #Selecting two positions
    chromosome[i:j] = reversed(chromosome[i:j]) # Reversing segment
    return chromosome

#parent selection funcs:
def fitness_proportional_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    # print("/")
    return random.choices(population, weights=probabilities, k=1)[0]

def rank_based_selection(population, fitness_values):
    sorted_indices = np.argsort(fitness_values)
    ranks = np.arange(1, len(population) + 1)
    probabilities = ranks / sum(ranks)
    # print("/")
    return population[random.choices(sorted_indices, weights=probabilities, k=1)[0]]

def binary_tournament_selection(population, fitness_values):
    c1, c2 = random.sample(range(len(population)), 2)
    # print("/")
    return population[c1] if fitness_values[c1] < fitness_values[c2] else population[c2]

def random_selection(population):
    # print("/")
    return random.choice(population)

#survivor selection funcs:
def fps(population, fitness_values, pop_size):
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    selected = random.choices(population, weights=probabilities, k=pop_size)
    # print("/")
    return selected

def rbs(population, fitness_values, pop_size):
    sorted_indices = np.argsort(fitness_values)
    ranks = np.arange(1, len(population) + 1)
    probabilities = ranks / sum(ranks)  # Higher-ranked individuals have higher probability
    selected = [population[i] for i in random.choices(sorted_indices, weights=probabilities, k=pop_size)]
    # print("/")
    return selected

def bts(population, fitness_values, pop_size):
    selected = []
    for _ in range(pop_size):
        c1, c2 = random.sample(range(len(population)), 2)
        winner = population[c1] if fitness_values[c1] < fitness_values[c2] else population[c2]
        selected.append(winner)
    # print("/")
    return selected

def rs(population, pop_size):
    # print("/")
    return random.sample(population, pop_size)

def truncation_selection(population, fitness_values, pop_size):
    sorted_pop = [x for _, x in sorted(zip(fitness_values, population))]
    # print("/")
    return sorted_pop[:pop_size]

def evolutionary_algorithm(tsp_data):
    avg_fitness_table = []
    best_fitness_table = []
    best_overall_chromosome = None
    best_overall_fitness = float('inf')
    
    hof = HallOfFame(size=10)  # Store the best 10 individuals

    population = initialize_population(tsp_data)  # Initialize population only once

    for iteration in range(ITERATIONS):
        print(f"\nIteration {iteration + 1}/{ITERATIONS}")
        best_fitness_values = []
        avg_fitness_values = []

        for generation in range(GENERATIONS):
            fitness_scores = [fitness_function(sol, tsp_data) for sol in population]
            best_fitness = min(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            
            best_fitness_values.append(best_fitness)
            avg_fitness_values.append(avg_fitness)
            
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")
            
            best_idx = np.argmin(fitness_scores)
            if fitness_scores[best_idx] < best_overall_fitness:
                best_overall_fitness = fitness_scores[best_idx]
                best_overall_chromosome = population[best_idx]
            
            hof.update(population, fitness_scores)  # Update Hall of Fame
            
            # Create offspring
            offspring = []
            while len(offspring) < OFFSPRINGS - ELITISM_SIZE:
                parent1 = rank_based_selection(population, fitness_scores)
                parent2 = rank_based_selection(population, fitness_scores)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                offspring.extend([mutate(child1), mutate(child2)])
            
            # Introducing diversity by adding random individuals
            random_individuals = random.sample(population, int(0.8 * POPULATION_SIZE))  # Adding 80% random individuals
            elite_population = list(hof.get_elite(ELITISM_SIZE))
            remaining_population = offspring + random_individuals  # Adding some randomness
            
            # Creating the new population by mixing elites and new individuals
            population = elite_population + remaining_population
            
            # Truncating population to maintain size
            population = truncation_selection(population, [fitness_function(sol, tsp_data) for sol in population], POPULATION_SIZE)
        
        avg_fitness_table.append(avg_fitness_values)
        best_fitness_table.append(best_fitness_values)
    
    plot_results(best_fitness_table, avg_fitness_table)
    
    return best_overall_chromosome, best_overall_fitness

def plot_results(best_fitness_table, avg_fitness_table):
    # Calculating average BSF and ASF across all runs (iterations)
    avg_bsf = np.mean(best_fitness_table, axis=0)
    avg_asf = np.mean(avg_fitness_table, axis=0)
    
    # Plotting the results
    plt.plot(avg_bsf, label="Average Best Fitness (BSF)")
    plt.plot(avg_asf, label="Average Fitness (ASF)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("EA Progress Over Generations")
    plt.legend()
    plt.show()

# def print_fitness_tables(avg_fitness_table, best_fitness_table):
#     print("\nBest Fitness Table (BSF):")
#     print("Generation\t" + "\t".join([f"Run {i+1}" for i in range(ITERATIONS)]) + "\tAverage")
#     for gen in range(GENERATIONS):
#         row = [best_fitness_table[i][gen] for i in range(ITERATIONS)]
#         print(f"{gen+1}\t" + "\t".join(map(str, row)) + f"\t{np.mean(row):.2f}")
    
#     print("\nAverage Fitness Table (ASF):")
#     print("Generation\t" + "\t".join([f"Run {i+1}" for i in range(ITERATIONS)]) + "\tAverage")
#     for gen in range(GENERATIONS):
#         row = [avg_fitness_table[i][gen] for i in range(ITERATIONS)]
#         print(f"{gen+1}\t" + "\t".join(map(str, row)) + f"\t{np.mean(row):.2f}")

# Usage
filename = "qa194.tsp"
tsp_data = read_tsp_data(filename)
best_solution, best_fitness = evolutionary_algorithm(tsp_data)
print("Best solution:", best_solution)
print("Best Fitness:", best_fitness)
