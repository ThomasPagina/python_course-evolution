import random
from typing import List
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# --- Data Models ---
class Guest:
    def __init__(self, name: str, gender: str, traits: List[str]):
        self.name = name
        self.gender = gender
        self.traits = traits

    def __repr__(self):
        return self.name

# --- Sample Data ---
guests = [
    Guest("Uncle Joe", "male", ["loud_voice"]),
    Guest("Aunt Zia", "female", ["deaf_left"]),
    Guest("Ex-Wife Linda", "female", []),
    Guest("Current-Wife Karen", "female", []),
    Guest("Tante Sara", "female", ["no_men_nearby"]),
    Guest("Cousin Ben", "male", []),
    Guest("Child Mia", "female", ["child"]),
    Guest("Child Tom", "male", ["child"]),
    Guest("Child Zoe", "female", ["child"]),
    Guest("Granny Helga", "female", [])
]

table_sizes = [4, 4, 2]

# --- DEAP Setup ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", lambda: random.sample(range(len(guests)), len(guests)))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_individual(individual, guests, table_sizes):
    ordered_guests = [guests[i] for i in individual]
    tables = []
    index = 0
    for size in table_sizes:
        tables.append(ordered_guests[index:index + size])
        index += size
    return tables

def evaluate(individual):
    tables = decode_individual(individual, guests, table_sizes)
    violations = 0

    for table in tables:
        for i, guest in enumerate(table):
            if "deaf_left" in guest.traits and i > 0:
                if "loud_voice" not in table[i - 1].traits:
                    violations += 1

            names = [g.name for g in table]
            if "Ex-Wife Linda" in names and "Current-Wife Karen" in names:
                violations += 2

            if "no_men_nearby" in guest.traits:
                if (i > 0 and table[i - 1].gender == "male") or \
                   (i < len(table) - 1 and table[i + 1].gender == "male"):
                    violations += 1

        kids = [g for g in table if "child" in g.traits]
        adults = [g for g in table if "child" not in g.traits]
        if len(kids) > 1 and len(kids) == len(table):
            violations += 2
        elif len(kids) > 1 and len(adults) < 1:
            violations += 1

    return (violations,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Evolution Process ---
def run_deap_evolution():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=300, 
                        stats=None, halloffame=hof, verbose=True)
    best = hof[0]
    best_tables = decode_individual(best, guests, table_sizes)
    score = evaluate(best)[0]
    return best_tables, score

# --- Visualization ---
def plot_tables(tables):
    fig, ax = plt.subplots(figsize=(8, 5))
    y_offset = 1
    for i, table in enumerate(tables):
        for j, guest in enumerate(table):
            ax.text(j, -i * y_offset, guest.name, bbox=dict(facecolor='lightblue', edgecolor='black'), ha='center')
        ax.plot([0, len(table) - 1], [-i * y_offset, -i * y_offset], 'k--', lw=1)
    ax.set_xlim(-1, max(len(t) for t in tables))
    ax.set_ylim(-y_offset * len(tables), 1)
    ax.axis('off')
    plt.title("Seating Arrangement")
    plt.tight_layout()
    plt.show()

# --- Run & Output ---
if __name__ == "__main__":
    final_tables, violations = run_deap_evolution()
    print(f"Constraint violations: {violations}\n")
    for i, table in enumerate(final_tables):
        print(f"Table {i + 1}: {table}")
    plot_tables(final_tables)
