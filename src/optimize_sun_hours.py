import random
from typing import List, Tuple

# --- Data Models ---
class Activity:
    def __init__(self, name: str, location: str, day: int, start: int, duration: int):
        self.name = name
        self.location = location
        self.day = day  # 0 = Monday, ..., 6 = Sunday
        self.start = start  # start hour (24h format)
        self.duration = duration

    def __repr__(self):
        return f"{self.name} at {self.location}"

# --- Sample Activities (Expanded + Compressed Time Window) ---
activities = [
    Activity("Sofa Shopping", "Furniture Store", 0, 10, 3),
    Activity("Visit Aunt Zia", "Aunt Zia's House", 1, 14, 2),
    Activity("Brunch with Anna", "Anna's Place", 1, 9, 2),
    Activity("Clean Attic", "Mom's House", 2, 12, 4),
    Activity("Family Dinner", "Mom's House", 2, 18, 2),
    Activity("Pick Up Package", "Post Office", 0, 11, 1),
    Activity("Grocery Shopping", "Supermarket", 1, 15, 1),
    Activity("Dentist Appointment", "Dentist", 2, 10, 1),
    Activity("Bike Repair", "Bike Shop", 2, 14, 2),
]

locations = ["Strandbad", "Furniture Store", "Aunt Zia's House", "Anna's Place", "Mom's House", "Post Office", "Supermarket", "Dentist", "Bike Shop"]
location_indices = {name: idx for idx, name in enumerate(locations)}

# --- Travel Time Matrix (symmetric, in hours; Mom's House near Strandbad) ---
travel_time_matrix = [
    [0, 1, 2, 1, 0.1, 0.5, 0.6, 1.2, 1.3],
    [1, 0, 2, 2, 1, 0.8, 0.9, 1.1, 1.2],
    [2, 2, 0, 3, 2, 2.1, 2.2, 2.5, 2.6],
    [1, 2, 3, 0, 1, 1.2, 1.0, 1.3, 1.4],
    [0.1, 1, 2, 1, 0, 0.6, 0.7, 1.0, 1.1],
    [0.5, 0.8, 2.1, 1.2, 0.6, 0, 0.3, 0.9, 1.0],
    [0.6, 0.9, 2.2, 1.0, 0.7, 0.3, 0, 0.8, 0.9],
    [1.2, 1.1, 2.5, 1.3, 1.0, 0.9, 0.8, 0, 0.6],
    [1.3, 1.2, 2.6, 1.4, 1.1, 1.0, 0.9, 0.6, 0],
]

TOTAL_DAYS = 3
SUN_HOURS = list(range(9, 18))  # 9 to 17 inclusive

# --- Evolutionary Algorithm ---
def create_individual() -> List[bool]:
    return [random.choice([True, False]) for _ in activities]

def evaluate(individual: List[bool]) -> int:
    daily_sun = [9] * TOTAL_DAYS
    daily_locations = [[] for _ in range(TOTAL_DAYS)]
    daily_time_used = [0.0] * TOTAL_DAYS

    for active, activity in zip(individual, activities):
        if active:
            day = activity.day
            daily_locations[day].append(activity.location)
            daily_time_used[day] += activity.duration

    for day, locs in enumerate(daily_locations):
        if not locs:
            continue
        route = ["Strandbad"] + locs + ["Strandbad"]
        travel_time = 0.0
        for i in range(len(route) - 1):
            from_idx = location_indices[route[i]]
            to_idx = location_indices[route[i + 1]]
            travel_time += travel_time_matrix[from_idx][to_idx]
        daily_time_used[day] += travel_time
        daily_sun[day] = max(0, 9 - daily_time_used[day])

    return int(sum(daily_sun))

def mutate(individual: List[bool]) -> None:
    i = random.randrange(len(individual))
    individual[i] = not individual[i]

def crossover(ind1: List[bool], ind2: List[bool]) -> Tuple[List[bool], List[bool]]:
    point = random.randint(1, len(ind1) - 1)
    return ind1[:point] + ind2[point:], ind2[:point] + ind1[point:]

def run_evolution(generations=100, population_size=50) -> Tuple[List[bool], int]:
    population = [create_individual() for _ in range(population_size)]
    best = max(population, key=evaluate)
    best_score = evaluate(best)

    for _ in range(generations):
        new_population = []
        while len(new_population) < population_size:
            parents = random.sample(population, 2)
            child1, child2 = crossover(parents[0], parents[1])
            if random.random() < 0.2:
                mutate(child1)
            if random.random() < 0.2:
                mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:population_size]
        current_best = max(population, key=evaluate)
        current_score = evaluate(current_best)
        if current_score > best_score:
            best = current_best
            best_score = current_score

    return best, best_score

# --- Output Plan ---
def print_plan(individual: List[bool]):
    print("Tim's Optimized Vacation Schedule with Travel Time Consideration:\n")
    for active, activity in zip(individual, activities):
        status = "✔" if active else "✘"
        print(f"{status} {activity.name} on day {activity.day} at {activity.start}:00")
    print("\nMaximum Sun Hours:", evaluate(individual))

if __name__ == "__main__":
    best_individual, max_score = run_evolution()
    print_plan(best_individual)