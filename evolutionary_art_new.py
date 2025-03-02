import random
import numpy as np
from PIL import Image, ImageDraw
import cv2
import copy

class Chromosome:
    def __init__(self, target_image, height, width, num_vertices=3, num_polygons=50):
        self.targetImage = Image.open(target_image).convert("RGBA") 
        self.height = height
        self.width = width
        self.numVertices = num_vertices
        self.numPolygons = num_polygons

        # Resize the target image once and store it
        self.target_resized = self.targetImage.resize((self.width, self.height))
        self.target_array = np.array(self.target_resized)  # Convert to array once

    def fitness_function(self, image_array):
        return np.sum(np.abs(self.target_array - image_array)) 

    def generate(self):
        # start with transparent bg
        img = Image.new("RGBA", (self.height, self.width), (255, 255, 255, 0))  
        # load target images pixel for ref
        pix = self.targetImage.load()

        #  (average color of target image- helped start better else it was very rnd clrs
        avg_color = np.mean(np.array(self.targetImage), axis=(0, 1))
        base_color = tuple(int(c) for c in avg_color)
        img.paste(base_color, [0, 0, self.width, self.height])

        # start adding polygons on top of bg
        draw = ImageDraw.Draw(img)
        numPolygons = random.randint(5, min(self.numPolygons, 20)) 
        
        # edge detection to refine polygon placement based on target img
        edges = cv2.Canny(np.array(self.targetImage.convert("L")), 100, 200)

        for _ in range(numPolygons):
            # random center for the polygon
            x_center = random.randint(0, self.height - 1)
            y_center = random.randint(0, self.width - 1)

            # increase polygon complexity near edges
            numVertices = self.numVertices
            if edges[y_center, x_center] > 0:  # If near an edge, increase detail
                numVertices = random.randint(3, 5)

            # Generate the polygon with random vertices around the center
            polygon = [(random.randint(x_center - 10, x_center + 10),
                        random.randint(y_center - 10, y_center + 10)) 
                    for _ in range(numVertices)]

            # base color with a variation
            base_color = pix[x_center % self.width, y_center % self.height]
            color_variation = tuple(
                int(colr * 0.9 + pix[x_center % self.width, y_center % self.height][i] * 0.1)  
                for i, colr in enumerate(base_color)
            )
            #transparency for smoother blending
            opacity = random.randint(50, 255)
            color_variation = (*color_variation[:3], opacity)
            draw.polygon(polygon, fill=color_variation)

        return img
    
    def create_chromosome(self):
        image = self.generate()
        image_array = np.array(image)
        fitness = self.fitness_function(image_array)

        return {
            'height': self.height,
            'width': self.width,
            'image': image,
            'array': image_array,
            'fitness': fitness,  
        }

    def mutation(self, chromosome, mutationRate=0.05):
        fitness_scale = (1 / (chromosome['fitness'] + 1e-3))
        if random.random() < mutationRate:
            area = random.randint(1, max(chromosome['height'], chromosome['width']) // 10)            
            img = chromosome['image'].copy()
            imgGen = ImageDraw.Draw(img)
            # better fitness  would result in less mutations
            maxMutations = max(1, int(fitness_scale * 5))
            numMutations = random.randint(1, maxMutations)
            for _ in range(numMutations):
                x = random.randint(area, chromosome['height'] - area)
                y = random.randint(area, chromosome['width'] - area)
                loc = [(random.randint(x - area, x + area), random.randint(y - area, y + area))
                    for _ in range(self.numVertices)] 
                color = "#" + "".join(random.choices("0123456789ABCDEF", k=6))  
                imgGen.polygon(loc, fill=color)
            mutatedChromosome = self.create_chromosome()
            mutatedChromosome['image'] = img
            mutatedChromosome['array'] = np.array(img)
            mutatedChromosome['fitness'] = self.fitness_function(mutatedChromosome['array'])
            return mutatedChromosome
        else:
            return chromosome

class EvolutionaryAlgorithm:
    def __init__(self, target_image, population_size, width, height, fitness_function):
        self.populationSize = population_size
        self.width = width
        self.height = height
        self.fitness_function = fitness_function
        self.population = []
        self.numVertices = 3
        self.numPolygons = 50
        self.chromosome = Chromosome(target_image, height, width, self.numVertices, self.numPolygons)

    def initialize_population(self):
        self.population = []
        for _ in range(self.populationSize):
            chromosome = self.chromosome.create_chromosome()
            self.population.append(chromosome)

    def crossover(self, parent1, parent2):
        def apply_split_crossover():
            child_image = Image.new("RGBA", (self.width, self.height))
            split_type = random.choice(["horizontal", "vertical", "quadrant"])

            if split_type == "horizontal":
                split_pos = random.randint(0, self.height)
                child_image.paste(parent1['image'].crop((0, 0, self.width, split_pos)), (0, 0))
                child_image.paste(parent2['image'].crop((0, split_pos, self.width, self.height)), (0, split_pos))

            elif split_type == "vertical":
                split_pos = random.randint(0, self.width)
                child_image.paste(parent1['image'].crop((0, 0, split_pos, self.height)), (0, 0))
                child_image.paste(parent2['image'].crop((split_pos, 0, self.width, self.height)), (split_pos, 0))

            else:  # Quadrant split
                split_x, split_y = random.randint(0, self.width), random.randint(0, self.height)
                child_image.paste(parent1['image'].crop((0, 0, split_x, split_y)), (0, 0))
                child_image.paste(parent2['image'].crop((split_x, 0, self.width, split_y)), (split_x, 0))
                child_image.paste(parent1['image'].crop((0, split_y, split_x, self.height)), (0, split_y))
                child_image.paste(parent2['image'].crop((split_x, split_y, self.width, self.height)), (split_x, split_y))

            return child_image

        def apply_blend_crossover():
            blend_ratio = random.random()
            return Image.blend(parent1['image'], parent2['image'], blend_ratio)

        #  crossover strategy choice
        strategy = random.choices(["split", "blend"], weights=[0.5, 0.5], k=1)[0]
        child_image = apply_split_crossover() if strategy == "split" else apply_blend_crossover()

        # evaluate the child 
        child = self.chromosome.create_chromosome()
        child['image'] = child_image
        child['array'] = np.array(child_image)
        child['fitness'] = self.fitness_function(child['array'])

        # Accept child iff its better than parent
        if (strategy == "split" and child['fitness'] <= max(parent1['fitness'], parent2['fitness'])) or \
        (strategy == "blend" and child['fitness'] <= min(parent1['fitness'], parent2['fitness'])):
            return child

        return None  # Return None if the new child is not better

    def tournament_select(population, tournament_size=8):  #tournament size set to 8 (25% population)
        selected = random.sample(population, tournament_size)  # randomly pick individuals
        return min(selected, key=lambda ind: ind['fitness'])  # return the fittest

    def evolutionary_algorithm(question, generations, parentScheme):
        question.initialize_population()
        numElites = 2  
        max_attempts = 10  

        hall_of_fame = []  # store best individuals
        max_hof_size = 5   # at most 5 individuals in HoF

        def getBest(population):
            return min(ind['fitness'] for ind in population)

        for i in range(generations):
            question.population.sort(key=lambda ind: ind['fitness'])
            bestFit = getBest(question.population)

            # Update HoF
            best_individual = question.population[0]
            if len(hall_of_fame) < max_hof_size:
                hall_of_fame.append(copy.deepcopy(best_individual))
            else:
                worst_hof = max(hall_of_fame, key=lambda ind: ind['fitness'])
                if best_individual['fitness'] < worst_hof['fitness']:
                    hall_of_fame.remove(worst_hof)
                    hall_of_fame.append(copy.deepcopy(best_individual))

            # Store elites
            elites = [question.chromosome.mutation(copy.deepcopy(ind), 0.05) for ind in question.population[:numElites]]
            newPopulation = elites[:]

            while len(newPopulation) < question.populationSize:
                parentOne = parentScheme(question.population)
                parentTwo = parentScheme(question.population)

                child = None
                for _ in range(max_attempts):
                    child = question.crossover(parentOne, parentTwo)
                    if child is not None:
                        break

                if child is None:
                    child = random.choice(question.population)

                # adaptive mutation
                mutationRate = max(0.05, 1 - (i / generations))
                child = question.chromosome.mutation(child, mutationRate)

                newPopulation.append(child)

            # Reintroduce HoF individuals every 20 gens
            if i % 20 == 0 and hall_of_fame:
                elite_from_hof = copy.deepcopy(random.choice(hall_of_fame))
                newPopulation[random.randint(0, len(newPopulation) - 1)] = elite_from_hof

            # Ensure HoF individuals replace weakest beofre elites
            if len(hall_of_fame) > 0:
                for j in range(min(numElites, len(hall_of_fame))):
                    newPopulation[-(j+1)] = copy.deepcopy(hall_of_fame[j])

            question.population = newPopulation

            print("Fittest Chromosome in gen {} : {:.2f}".format(i, bestFit))

            if i % 100 == 0 or i == generations - 1:
                question.population.sort(key=lambda ind: ind['fitness'])
                fittest = question.population[0]
                fittest['image'].save(f"results/fittest_{i}.png")

        question.population.sort(key=lambda ind: ind['fitness'])
        return question.population[0]  
    
def main():
    path = "mona.png"
    populationSize = 200
    numPolygons = 50
    numVertices = 3
    numGenerations = 100000

    chromosome = Chromosome(path, 200, 200, numVertices, numPolygons)

    question = EvolutionaryAlgorithm(
        target_image=path,
        population_size=populationSize,
        width=200,
        height=200,
        fitness_function= chromosome.fitness_function
    )

    # Run the evolutionary algorithm
    fittest_chromosome = EvolutionaryAlgorithm.evolutionary_algorithm(question, numGenerations, EvolutionaryAlgorithm.tournament_select)

if __name__ == "__main__":
    main()