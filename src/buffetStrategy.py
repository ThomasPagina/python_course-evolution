import random
from typing import List, Tuple
import matplotlib.pyplot as plt

# --- Buffet and Strategy Definitions ---
class Dish:
    def __init__(self, name: str, value: int, popularity: int):
        self.name = name
        self.value = value  # Calories or satisfaction per serving
        self.popularity = popularity  # How many people want it (affects crowding)

    def __repr__(self):
        return f"{self.name} (val={self.value}, pop={self.popularity})"

class Guest:
    def __init__(self, strategy: str):
        self.strategy = strategy  # "toby" or "sonja"
        self.total_value = 0

    def select_dishes(self, buffet: List[Dish]) -> List[Dish]:
        if self.strategy == "toby":
            sorted_dishes = sorted(buffet, key=lambda d: d.popularity)
        elif self.strategy == "sonja":
            sorted_dishes = sorted(buffet, key=lambda d: -d.value)
        else:
            sorted_dishes = buffet[:]
        return sorted_dishes

    def eat(self, buffet: List[Dish], other_guests: List['Guest']):
        chosen_dishes = self.select_dishes(buffet)
        time_budget = 10
        total_value = 0

        for dish in chosen_dishes:
            competitors = sum(1 for g in other_guests if dish in g.select_dishes(buffet)[:3])
            time_cost = 1 + 0.5 * (dish.popularity + competitors)
            if time_budget >= time_cost:
                time_budget -= time_cost
                total_value += dish.value

        self.total_value = total_value

# --- Simulation ---
def create_buffet() -> List[Dish]:
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

def simulate(buffet: List[Dish], num_toby: int, num_sonja: int) -> Tuple[float, float]:
    guests = [Guest("toby") for _ in range(num_toby)] + [Guest("sonja") for _ in range(num_sonja)]
    for guest in guests:
        guest.eat(buffet, [g for g in guests if g != guest])
    toby_scores = [g.total_value for g in guests if g.strategy == "toby"]
    sonja_scores = [g.total_value for g in guests if g.strategy == "sonja"]
    avg_toby = sum(toby_scores) / len(toby_scores) if toby_scores else 0
    avg_sonja = sum(sonja_scores) / len(sonja_scores) if sonja_scores else 0
    return avg_toby, avg_sonja

# --- Run Experiment ---
def run_experiment():
    buffet = create_buffet()
    print("Buffet Strategy Simulation (Toby vs. Sonja)")
    print("\n#Toby\t#Sonja\tToby avg\tSonja avg")
    x_vals = []
    toby_results = []
    sonja_results = []
    for ratio in range(0, 11):
        num_toby = ratio
        num_sonja = 10 - ratio
        avg_toby, avg_sonja = simulate(buffet, num_toby, num_sonja)
        print(f"{num_toby}\t{num_sonja}\t{avg_toby:.2f}\t{avg_sonja:.2f}")
        x_vals.append(num_toby)
        toby_results.append(avg_toby)
        sonja_results.append(avg_sonja)

    # Plotting results
    plt.plot(x_vals, toby_results, marker='o', label='Toby Avg Value')
    plt.plot(x_vals, sonja_results, marker='x', label='Sonja Avg Value')
    plt.xlabel("Number of Toby Guests (out of 10)")
    plt.ylabel("Average Satisfaction Value")
    plt.title("Buffet Strategy Comparison: Toby vs. Sonja")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
