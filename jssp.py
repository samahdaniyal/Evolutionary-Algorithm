import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 70
GENERATIONS = 50
MUTATION_RATE = 0.3
OFFSPRINGS = 50
ITERATIONS = 10

def load_jssp_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip().splitlines()

    num_jobs, num_machines = map(int, content[0].split())
    schedule_data = []
    for line in content[1:1+num_jobs]:
        tokens = line.split()
        # Pair up tokens into (machine, time) tuples.
        ops = []
        for j in range(0, len(tokens), 2):
            ops.append((int(tokens[j]), int(tokens[j+1])))
        schedule_data.append(ops)
    return num_jobs, num_machines, schedule_data

def generate_schedule(sol, proc_data, nm, nj):
    # Work on a deep copy so that the original data is unchanged.
    data_copy = copy.deepcopy(proc_data)
    machine_time = [0] * nm       # Time when each machine is free.
    job_time = [0] * nj           # Completion time of last operation for each job.
    schedule = [[] for _ in range(nm)]  # List of scheduled operations per machine

    for job in sol:
        # For the current job, pop the next pending operation.
        op = data_copy[job].pop(0)
        m, proc_time = op
        start = max(machine_time[m], job_time[job])
        finish = start + proc_time

        schedule[m].append((job, start, proc_time))
        machine_time[m] = finish
        job_time[job] = finish

    return schedule

def fitness_calculation(sol, proc_data, nm, nj):
    sched = generate_schedule(sol, proc_data, nm, nj)
    finishing_times = []
    for machine_ops in sched:
        if machine_ops:
            # Last operation on the machine
            job, start, duration = machine_ops[-1]
            finishing_times.append(start + duration)
        else:
            finishing_times.append(0)
    return max(finishing_times)

def initializing_population(popULATION_size, nj, nm):
    population = []
    for _ in range(popULATION_size):
        individual = []
        for _ in range(nm):
            block = random.sample(range(nj), nj)
            individual.extend(block)
        population.append(individual)
    return population

def crossover(parent_a, parent_b, nj, nm):
    # Choose two distinct machine boundaries
    cp1, cp2 = sorted(random.sample(range(1, nm), 2))
    idx1, idx2 = cp1 * nj, cp2 * nj

    child1 = parent_b[:idx1] + parent_a[idx1:idx2] + parent_b[idx2:]
    child2 = parent_a[:idx1] + parent_b[idx1:idx2] + parent_a[idx2:]
    return child1, child2

def mutate(solution, nj, nm):
    mutant = solution[:]
    # Pick two different blocks at random.
    m1, m2 = random.sample(range(nm), 2)
    start1, start2 = m1 * nj, m2 * nj

    block1 = mutant[start1:start1+nj]
    block2 = mutant[start2:start2+nj]
    mutant[start1:start1+nj] = block2
    mutant[start2:start2+nj] = block1
    return mutant


def random_selection(fitness_list):
    return random.randint(0, len(fitness_list)-1)

def truncation_selection(fitness_list):
    return fitness_list.index(min(fitness_list))

def rank_based_selection(fitness_list):
    indices = list(range(len(fitness_list)))
    # Rank individuals in ascending order (best first)
    indices.sort(key=lambda i: fitness_list[i])
    weights = [len(fitness_list) - rank for rank in range(len(fitness_list))]
    chosen = random.choices(indices, weights=weights, k=1)
    return chosen[0]

def fitness_proportional_selection(fitness_list):
    total_fitness = sum(fitness_list)
    if total_fitness == 0:
        return random.randint(0, len(fitness_list) - 1)  # Avoid division by zero

    probabilities = [fitness / total_fitness for fitness in fitness_list]
    return random.choices(range(len(fitness_list)), weights=probabilities, k=1)[0]

def binary_tournament_selection(fitness_list):
    c1, c2 = random.sample(range(len(fitness_list)), 2)
    return c1 if fitness_list[c1] < fitness_list[c2] else c2

def select_parents(pop, fit_scores):
    i1 = binary_tournament_selection(fit_scores)
    i2 = binary_tournament_selection(fit_scores)
    return pop[i1], pop[i2]

def truncate_population(pop, fit_scores, target_size):
    pop_copy = pop[:]
    scores_copy = fit_scores[:]
    new_pop = []
    for _ in range(target_size):
        best_index = truncation_selection(scores_copy)
        new_pop.append(pop_copy.pop(best_index))
        scores_copy.pop(best_index)
    return new_pop

def evolutionary_algorithm(proc_data, pop, generations, offspringS, mut_rate, nj, nm):
    best_history = []
    avg_history = []

    for gen in range(generations):
        # Compute fitness (makespan) for each individual in the current population.
        fitness_list = [fitness_calculation(ind, proc_data, nm, nj) for ind in pop]

        offspring = []
        # Generate offspring (each crossover produces two children)
        for _ in range(offspringS // 2):
            parent1, parent2 = select_parents(pop, fitness_list)
            child1, child2 = crossover(parent1, parent2, nj, nm)
            if random.random() < mut_rate:
                child1 = mutate(child1, nj, nm)
            if random.random() < mut_rate:
                child2 = mutate(child2, nj, nm)
            offspring.extend([child1, child2])

        # Add the new offspring to the current population.
        pop.extend(offspring)

        # Re-evaluate the fitness after adding offspring.
        fitness_list = [fitness_calculation(ind, proc_data, nm, nj) for ind in pop]
        # Reduce population back to original size by keeping the best individuals.
        pop = truncate_population(pop, fitness_list, len(pop) - len(offspring))

        # Record statistics for the generation.
        fitness_list = [fitness_calculation(ind, proc_data, nm, nj) for ind in pop]
        best_fit = min(fitness_list)
        avg_fit = sum(fitness_list) / len(fitness_list)
        best_history.append(best_fit)
        avg_history.append(avg_fit)
        print(f"Generation {gen+1}: Best = {best_fit}, Average = {avg_fit}")

    # Final best individual
    final_scores = [fitness_calculation(ind, proc_data, nm, nj) for ind in pop]
    best_index = final_scores.index(min(final_scores))
    return pop, pop[best_index], final_scores[best_index], best_history, avg_history

def display_schedule(sol, proc_data, nm, nj, cmax_time):
    sched = generate_schedule(sol, proc_data, nm, nj)
    colors = plt.cm.Set3.colors

    # Create a list of tuples (machine_label, start, duration, job)
    bars = []
    for m in range(nm):
        for job_op in sched[m]:
            job, start, duration = job_op
            bars.append((f"M{m}", start, duration, job))

    fig, ax = plt.subplots()

    for machine, start, duration, job in bars:
        clr = colors[job % len(colors)]
        ax.barh(machine, duration, left=start, color=clr, edgecolor='black')
        ax.text(start + duration / 2, machine, str(job), ha='center', va='center', color='white')

    # Mark the overall completion time
    ax.axvline(cmax_time, color='red', linestyle='--', linewidth=2, label='Cmax')
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Job-Shop Schedule")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Read problem data
filename = "CI_Assignment01_Samah_Aina/la19.txt"
nj, nm, jssp_data = load_jssp_file(filename)

overall_best_solutions = []
cumulative_best = [0] * GENERATIONS
cumulative_avg = [0] * GENERATIONS

for itr in range(ITERATIONS):
    print(f"\nStarting Iteration {itr+1}")
    # Initialize a new population for each iteration.
    population = initializing_population(POPULATION_SIZE, nj, nm)
    population, best_indiv, best_makespan, best_hist, avg_hist = evolutionary_algorithm(
        jssp_data, population, GENERATIONS, OFFSPRINGS, MUTATION_RATE, nj, nm)
    overall_best_solutions.append(best_indiv)
    cumulative_best = [cb + bh for cb, bh in zip(cumulative_best, best_hist)]
    cumulative_avg = [ca + ah for ca, ah in zip(cumulative_avg, avg_hist)]
    print(f"Iteration {itr+1} best makespan: {best_makespan}")

# Compute mean fitness over iterations for each generation.
mean_best = [val / ITERATIONS for val in cumulative_best]
mean_avg = [val / ITERATIONS for val in cumulative_avg]

# Plot fitness evolution over generations.
gens = range(1, GENERATIONS + 1)
plt.plot(gens, mean_best, label='Mean Best Fitness')
plt.plot(gens, mean_avg, label='Mean Average Fitness')
plt.xlabel("Generation")
plt.ylabel("Makespan")
plt.title("Mean Best and Average Fitness over Generations")
plt.legend()
plt.show()

# Identify the best solution across all iterations.
best_fitnesses = [fitness_calculation(ind, jssp_data, nm, nj) for ind in overall_best_solutions]
best_overall = min(best_fitnesses)
best_index = best_fitnesses.index(best_overall)
print(f"\nOverall best makespan: {best_overall}")

# Visualize the schedule corresponding to the best solution.
display_schedule(overall_best_solutions[best_index], jssp_data, nm, nj, best_overall)
