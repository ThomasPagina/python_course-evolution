import time
import random
from collections import Counter
from pipeline9 import Pipeline, PipelineStep, MakeAppender, ToLower, ToCapitalize, ReverseString, RemoveLastChar, SwapFirstLast, DoubleLastChar, LastBecomesFirst

# --- Setup ---
source = "BDR"
target = "Erdbeere"

# Step pool: including general string operations
basic_steps = [
    ToLower(), ToCapitalize(), ReverseString(),
    RemoveLastChar(), SwapFirstLast(), DoubleLastChar(), LastBecomesFirst()
]

# Identify required letter frequencies (case-insensitive)
source_count = Counter(source.lower())
target_count = Counter(target.lower())
missing_letters = []
for letter, count in target_count.items():
    missing = count - source_count.get(letter, 0)
    if missing > 0:
        missing_letters.extend([letter] * missing)

# Allow appending each needed letter as often as it is missing
appenders = [MakeAppender(letter) for letter in missing_letters]

# All possible steps
all_steps = basic_steps + appenders

# --- Evolutionary Setup ---
POP_SIZE = 200
GENERATIONS = 1000
MUTATION_RATE = 0.3
MAX_LENGTH = 15
APPENDER_PENALTY = 0.4

class Individual:
    def __init__(self, steps=None):
        self.steps = steps or [random.choice(all_steps) for _ in range(random.randint(3, MAX_LENGTH))]
        self.fitness = None

    def evaluate(self):
        result = source
        penalty = 0
        for step in self.steps:
            result = step.process(result)
            if isinstance(step, MakeAppender):
                penalty += APPENDER_PENALTY
        self.fitness = -levenshtein(result, target) - penalty + (len(self.steps) * -0.05)
        return self.fitness

    def mutate(self):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5 and len(self.steps) > 1:
                self.steps[random.randint(0, len(self.steps)-1)] = random.choice(all_steps)
            else:
                self.steps.insert(random.randint(0, len(self.steps)), random.choice(all_steps))

    def crossover(self, other):
        a, b = sorted(random.sample(range(min(len(self.steps), len(other.steps))), 2))
        child_steps = self.steps[:a] + other.steps[a:b] + self.steps[b:]
        return Individual(child_steps)

    def run(self):
        result = source
        for step in self.steps:
            result = step.process(result)
        return result

    def __repr__(self):
        return " -> ".join(str(s) for s in self.steps)

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

# --- Main Evolution ---
def main():
    start_time = time.time()
    population = [Individual() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        for individual in population:
            individual.evaluate()

        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        print(f"Gen {gen} | Best Fitness: {best.fitness:.2f} | Output: {best.run()}")
        if best.run() == target:
            break

        next_gen = population[:10]  # elitism
        while len(next_gen) < POP_SIZE:
            parent1, parent2 = random.sample(population[:50], 2)
            child = parent1.crossover(parent2)
            child.mutate()
            next_gen.append(child)
        population = next_gen

    end_time = time.time()
    print("\nBest pipeline found:")
    print(best)
    result = best.run()
    print(f"Result: {result}")
    print(f"Target: {target}")
    assert result == target, "Pipeline failed to produce the expected result."
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
