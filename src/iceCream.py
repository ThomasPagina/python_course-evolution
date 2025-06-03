import random
from typing import List
import matplotlib.pyplot as plt

# --- Agent Definition ---
class Child:
    def __init__(self, name: str, strategy: str):
        self.name = name
        self.strategy = strategy  # 'patient' or 'sneaky'
        self.ice_cream_count = 0

    def decide(self, day: int, time: str) -> bool:
        if self.strategy == 'patient':
            return time == 'afternoon'  # waits until after nap
        elif self.strategy == 'sneaky':
            return True
        return False

# --- Simulation ---
class IceCreamSimulator:
    def __init__(self, children: List[Child], ice_per_day: int = 1):
        self.children = children
        self.ice_per_day = ice_per_day
        self.freezer_stock = 5 * ice_per_day * len(children)  # enough for the week
        self.log = []
        self.stock_over_time = []

    def simulate_week(self):
        for day in range(5):  # Monday to Friday
            daily_log = []
            ice_available = self.ice_per_day * len(self.children)
            for time in ['morning', 'afternoon']:
                random.shuffle(self.children)  # order matters if stock runs out
                for child in self.children:
                    if self.freezer_stock <= 0:
                        break
                    if child.decide(day, time):
                        if ice_available > 0:
                            child.ice_cream_count += 1
                            ice_available -= 1
                            self.freezer_stock -= 1
                            daily_log.append(f"{child.name} took an ice cream in the {time}.")
                        else:
                            daily_log.append(f"{child.name} wanted ice cream in the {time}, but it was all gone.")
                self.stock_over_time.append(self.freezer_stock)
            self.log.append((day, daily_log))

    def print_summary(self):
        print("\n--- Weekly Ice Cream Summary ---\n")
        for child in self.children:
            print(f"{child.name} ({child.strategy}): {child.ice_cream_count} ice creams")
        print("\n--- Daily Log ---")
        for day, entries in self.log:
            print(f"\nDay {day + 1}:")
            for entry in entries:
                print(entry)

    def plot_results(self):
        # Ice cream count per child
        names = [child.name for child in self.children]
        counts = [child.ice_cream_count for child in self.children]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(names, counts)
        plt.xticks(rotation=45)
        plt.title("Total Ice Creams per Child")
        plt.ylabel("Ice Creams")

        # Freezer stock over time
        plt.subplot(1, 2, 2)
        plt.plot(self.stock_over_time, marker='o')
        plt.title("Freezer Stock Over Time")
        plt.xlabel("Action Steps (Morning & Afternoon)")
        plt.ylabel("Remaining Ice Creams")
        plt.tight_layout()
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    names = ["Leo", "Emma", "Noah", "Mia", "Olivia", "Liam", "Ella", "Ben", "Sophie", "Max"]
    children = []
    for name in names[:5]:
        children.append(Child(name, 'patient'))
    for name in names[5:]:
        children.append(Child(name, 'sneaky'))

    sim = IceCreamSimulator(children)
    sim.simulate_week()
    sim.print_summary()
    sim.plot_results()
