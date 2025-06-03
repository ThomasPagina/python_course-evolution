import time
import random
from collections import Counter
from deap import base, creator, tools, algorithms
from pipeline9 import PipelineStep, MakeAppender, ToLower, ToCapitalize, ReverseString, RemoveLastChar, SwapFirstLast, DoubleLastChar, LastBecomesFirst

# --- Setup ---
source = "BDR"
target = "Erdbeere"

basic_steps = [
    ToLower(), ToCapitalize(), ReverseString(),
    RemoveLastChar(), SwapFirstLast(), DoubleLastChar(), LastBecomesFirst()
]

# Determine necessary appenders based on target string
source_count = Counter(source.lower())
target_count = Counter(target.lower())
missing_letters = []
for letter, count in target_count.items():
    missing = count - source_count.get(letter, 0)
    if missing > 0:
        missing_letters.extend([letter] * missing)

appenders = [MakeAppender(letter) for letter in missing_letters]
all_steps = basic_steps + appenders

# --- DEAP Setup ---
POP_SIZE = 200
GENERATIONS = 100
MAX_LENGTH = 12
MUTATION_RATE = 0.3
APPENDER_PENALTY = 0.3

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("step", lambda: random.choice(all_steps))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.step, n=MAX_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    result = source
    penalty = 0
    for step in individual:
        result = step.process(result)
        if isinstance(step, MakeAppender):
            penalty += APPENDER_PENALTY
    fitness = -levenshtein(result, target) - penalty + (-0.2 * len(individual)) # try out different numbers for the length penalty
    return (fitness,)

def mutate(individual):
    if random.random() < 0.5 and len(individual) > 1:
        individual[random.randint(0, len(individual)-1)] = random.choice(all_steps)
    else:
        individual.insert(random.randint(0, len(individual)), random.choice(all_steps))
    return (individual,)

def crossover(ind1, ind2):
    a, b = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    ind1[a:b], ind2[a:b] = ind2[a:b], ind1[a:b]
    return ind1, ind2

toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Utility ---
def levenshtein(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1)
            )
    return dp[-1][-1]

# --- Main ---
def main():
    import matplotlib.pyplot as plt
    start = time.time()
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=MUTATION_RATE, ngen=GENERATIONS,
                        stats=None, halloffame=hof, verbose=True)

    best = hof[0]
    result = source
    for step in best:
        result = step.process(result)

    print("Best pipeline:")
    print(" -> ".join(str(s) for s in best))
    print(f"Result: {result}")
    print(f"Target: {target}")

    # Visualize step-by-step transformation
    current = source
    history = [current]
    for step in best:
        current = step.process(current)
        history.append(current)

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(history)), [len(s) for s in history], marker='o')
    plt.xticks(range(len(history)), history, rotation=45, ha='right')
    plt.xlabel("Pipeline Step")
    plt.ylabel("String Length")
    plt.title("Transformation Steps")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    assert result == target, "Pipeline failed to produce the expected result."
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()
