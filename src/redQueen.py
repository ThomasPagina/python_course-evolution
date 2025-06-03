import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Konstante Definition
GENOME_LENGTH = 10
POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3
MUTATION_INDPB = 0.1

# DEAP Typen erstellen
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox definieren
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=GENOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def similarity(ind1, ind2):
    return sum(x == y for x, y in zip(ind1, ind2))

def evaluate_host(individual, parasite_population):
    return GENOME_LENGTH - max(similarity(individual, p) for p in parasite_population),

def evaluate_parasite(individual, host_population):
    return max(similarity(individual, h) for h in host_population),

def evolve_population(host_pop, parasite_pop):
    host_fitness_history = []
    parasite_fitness_history = []

    for _ in range(GENERATIONS):
        for host in host_pop:
            host.fitness.values = evaluate_host(host, parasite_pop)
        for parasite in parasite_pop:
            parasite.fitness.values = evaluate_parasite(parasite, host_pop)

        avg_host = sum(ind.fitness.values[0] for ind in host_pop) / len(host_pop)
        avg_parasite = sum(ind.fitness.values[0] for ind in parasite_pop) / len(parasite_pop)
        host_fitness_history.append(avg_host)
        parasite_fitness_history.append(avg_parasite)

        host_offspring = algorithms.varAnd(
            toolbox.select(host_pop, len(host_pop)),
            toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB)
        parasite_offspring = algorithms.varAnd(
            toolbox.select(parasite_pop, len(parasite_pop)),
            toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB)

        host_pop[:] = host_offspring
        parasite_pop[:] = parasite_offspring

    return host_fitness_history, parasite_fitness_history

def plot_fitness(host_fitness, parasite_fitness):
    plt.figure()
    plt.plot(host_fitness, label='Host Avg Fitness')
    plt.plot(parasite_fitness, label='Parasite Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Red Queen Co-evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    host_population = toolbox.population(n=POPULATION_SIZE)
    parasite_population = toolbox.population(n=POPULATION_SIZE)
    host_fitness, parasite_fitness = evolve_population(host_population, parasite_population)
    plot_fitness(host_fitness, parasite_fitness)

if __name__ == "__main__":
    main()
