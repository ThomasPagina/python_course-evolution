import random
from typing import List
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# --- Simulation Logic ---
class SimulatedChild:
    def __init__(self, strategy: int):
        self.strategy = strategy  # 0 = patient, 1 = sneaky
        self.ice_cream_count = 0

    def decide(self, time: str) -> bool:
        return time == 'afternoon' if self.strategy == 0 else True

def simulate_population(strategy_list: List[int], ice_per_day=1) -> float:
    children = [SimulatedChild(strategy=s) for s in strategy_list]
    freezer_stock = 5 * ice_per_day * len(children)

    for day in range(5):  # Monday to Friday
        ice_available = ice_per_day * len(children)
        for time in ['morning', 'afternoon']:
            random.shuffle(children)
            for child in children:
                if freezer_stock <= 0:
                    break
                if child.decide(time):
                    if ice_available > 0:
                        child.ice_cream_count += 1
                        ice_available -= 1
                        freezer_stock -= 1

    total = sum(child.ice_cream_count for child in children)
    return total / len(children)

# --- DEAP Setup ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_strategy", lambda: random.choice([0, 1]))  # 0=patient, 1=sneaky
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_strategy, 10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    return (simulate_population(individual),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Run Evolution ---
def run_deap_simulation():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("max", max)
    stats.register("min", min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=False)
    return hof[0], log

# --- Plot ---
def plot_strategy_distribution(best_individual):
    labels = ['Patient', 'Sneaky']
    counts = [best_individual.count(0), best_individual.count(1)]
    plt.bar(labels, counts)
    plt.title("Best Strategy Distribution")
    plt.ylabel("Number of Children")
    plt.show()

if __name__ == "__main__":
    best, logbook = run_deap_simulation()
    plot_strategy_distribution(best)
    print("Best Individual:", best)
    print("Average Ice Creams per Child:", best.fitness.values[0])
    print("Logbook:", logbook)
    print("Simulation complete.")
