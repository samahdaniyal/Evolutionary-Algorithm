class HallOfFame:
    def __init__(self, size):
        self.size = size
        self.members = set()  # Using set to avoid duplicates
        self.fitness_scores = []

    def update(self, population, fitness_values):
        # Combining population with fitness values, sort by fitness (ascending for minimization)
        combined = list(zip(population, fitness_values))
        combined.sort(key=lambda x: x[1], reverse=False)  # Sort by fitness (lower is better)
        
        # Always keep the 'size' best individuals
        top_individuals = combined[:self.size]
        self.members, self.fitness_scores = zip(*top_individuals) if top_individuals else ([], [])
        
    def get_elite(self, num):
        # Convert set to list and then slice
        return list(self.members)[:num]  # Get the top 'num' elite individuals
