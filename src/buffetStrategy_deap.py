import random
from typing import List
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# --- Buffet Setup ---
class Dish:
    def __init__(self, name: str, value: int, popularity: int):
        self.name = name
        self.value = value
        self.popularity = popularity

    def __repr__(self):
        return f"{self.name} (val={self.value}, pop={self.popularity})"

def create_buffet():
    return [
        Dish("Cheesecake", 8, 9),
        Dish("Shrimp Cocktail", 10, 8),
        Dish("Cheese Cubes", 6, 5),
        Dish("Bread", 3, 2),
        Dish("Rice", 4, 3),
        Dish("Salad", 2, 1),
        Dish("Ribs", 9, 7),
        Dish("Fruit Salad", 5, 6),
    ]

# --- Guest Evaluation ---
def evaluate(individual: List[int], buffet: List[Dish], population: List[List[int]]) -> tuple:
    strategy = "toby" if individual[0] == 0 else "sonja"
    time_budget = 10
    chosen_dishes = []
    if strategy == "toby":
        sorted_dishes = sorted(buffet, key=lambda d: d.popularity)
    else:
        sorted_dishes = sorted(buffet, key=lambda d: -d.value)

    other_guests = ["toby" if ind[0] == 0 else "sonja" for ind in population if ind != individual]
    total_value = 0
    for dish in sorted_dishes:
        competitors = sum(1 for strat in other_guests if dish in (sorted(buffet, key=lambda d: d.popularity)[:3] if strat == "toby" else sorted(buffet, key=lambda d: -d.value)[:3]))
        time_cost = 1 + 0.5 * (dish.popularity + competitors)
        if time_budget >= time_cost:
            time_budget -= time_cost
            total_value += dish.value
    return (total_value,)

# --- DEAP Setup ---
buffet = create_buffet()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_strategy", lambda: random.choice([0, 1]))  # 0 = Toby, 1 = Sonja
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_strategy, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_wrapper(individual, population):
    return evaluate(individual, buffet, population)

def run_deap():
    population = toolbox.population(n=20)
    for ind in population:
        ind.fitness.values = evaluate_wrapper(ind, population)

    for gen in range(30):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for ind in offspring:
            if random.random() < 0.1:
                ind[0] = 1 - ind[0]

        for ind in offspring:
            ind.fitness.values = evaluate_wrapper(ind, offspring)

        population[:] = offspring

    return population

toolbox.register("evaluate", evaluate_wrapper)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Run and Plot ---
def plot_strategy_distribution(population):
    strategies = ["Toby" if ind[0] == 0 else "Sonja" for ind in population]
    counts = {"Toby": strategies.count("Toby"), "Sonja": strategies.count("Sonja")}
    plt.bar(counts.keys(), counts.values())
    plt.title("Final Strategy Distribution")
    plt.ylabel("Number of Guests")
    plt.show()

if __name__ == "__main__":
    final_population = run_deap()
    plot_strategy_distribution(final_population)
