import random
from typing import List, Dict, Tuple

# --- Data Models ---
class Guest:
    def __init__(self, name: str, gender: str, traits: List[str]):
        self.name = name
        self.gender = gender  # 'male' or 'female'
        self.traits = traits  # e.g. ['loud_voice'], ['deaf_left'], etc.

    def __repr__(self):
        return self.name

class Table:
    def __init__(self, size: int):
        self.size = size
        self.seats: List[Guest] = []

    def is_full(self):
        return len(self.seats) >= self.size

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

table_sizes = [4, 4, 2]  # Total of 10 seats

# --- Constraint Evaluation ---
def evaluate_constraints(tables: List[Table]) -> int:
    violations = 0

    for table in tables:
        for i, guest in enumerate(table.seats):
            # 1. Aunt Zia should have someone loud to her left
            if "deaf_left" in guest.traits and i > 0:
                if "loud_voice" not in table.seats[i - 1].traits:
                    violations += 1

            # 2. Ex-Wife and Current-Wife shouldn't be at the same table
            names = [g.name for g in table.seats]
            if "Ex-Wife Linda" in names and "Current-Wife Karen" in names:
                violations += 2

            # 3. Tante Sara doesn't want to sit next to men
            if "no_men_nearby" in guest.traits:
                if (i > 0 and table.seats[i - 1].gender == "male") or \
                   (i < len(table.seats) - 1 and table.seats[i + 1].gender == "male"):
                    violations += 1

        # 4. Kids should sit together, but with at least one adult
        kids = [g for g in table.seats if "child" in g.traits]
        adults = [g for g in table.seats if "child" not in g.traits]
        if len(kids) > 1 and len(kids) == len(table.seats):
            violations += 2
        elif len(kids) > 1 and len(adults) < 1:
            violations += 1

    return violations

# --- Evolutionary Algorithm ---
def create_individual(guests: List[Guest], table_sizes: List[int]) -> List[Table]:
    shuffled = guests[:]
    random.shuffle(shuffled)
    tables = []
    index = 0
    for size in table_sizes:
        table = Table(size)
        for _ in range(size):
            table.seats.append(shuffled[index])
            index += 1
        tables.append(table)
    return tables

def mutate(individual: List[Table]):
    flat = [g for t in individual for g in t.seats]
    i, j = random.sample(range(len(flat)), 2)
    flat[i], flat[j] = flat[j], flat[i]

    # Reassign to tables
    index = 0
    for table in individual:
        table.seats = flat[index:index + table.size]
        index += table.size

# --- Main Evolution Loop ---
def evolutionary_seating(guests: List[Guest], table_sizes: List[int], generations: int = 500) -> Tuple[List[Table], int]:
    best = create_individual(guests, table_sizes)
    best_score = evaluate_constraints(best)

    for _ in range(generations):
        candidate = create_individual(guests, table_sizes)
        mutate(candidate)
        score = evaluate_constraints(candidate)
        if score < best_score:
            best = candidate
            best_score = score
            if best_score == 0:
                break
    return best, best_score

# --- Run and Output ---
if __name__ == "__main__":
    final_tables, score = evolutionary_seating(guests, table_sizes)
    print(f"Constraint violations: {score}\n")
    for i, table in enumerate(final_tables):
        print(f"Table {i + 1}: {table.seats}")
