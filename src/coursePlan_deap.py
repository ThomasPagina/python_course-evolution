import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# --- Data Model ---
class Course:
    def __init__(self, name: str, ects: int, predicted_grade: float, workload_hours: int):
        self.name = name
        self.ects = ects
        self.predicted_grade = predicted_grade
        self.workload_hours = workload_hours

    def __repr__(self):
        return f"{self.name} (ECTS: {self.ects}, Grade: {self.predicted_grade}, Work: {self.workload_hours}h)"

# --- Sample Courses ---
courses = [
    Course("Digital Text Analysis", 6, 1.7, 35),
    Course("Introduction to Python", 5, 1.3, 25),
    Course("Editorial Theory", 6, 1.7, 50),
    Course("Cultural Data Studies", 5, 2.3, 30),
    Course("Philosophy of Technology", 4, 1.3, 20),
    Course("UX Research", 5, 2.0, 32),
    Course("Media Ethics", 6, 1.3, 15),
    Course("Statistics for Humanities", 10, 2.7, 55),
    Course("Archives and Metadata", 5, 2.0, 28),
    Course("Machine Learning Basics", 6, 2.3, 40),
]

# --- DEAP Setup ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: random.randint(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(courses))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    selected = [courses[i] for i in range(len(courses)) if individual[i] == 1]
    if not selected:
        return 0, 5.0, 999
    ects = sum(c.ects for c in selected)
    avg_grade = sum(c.predicted_grade for c in selected) / len(selected)
    workload = sum(c.workload_hours for c in selected)
    return ects, avg_grade, workload

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

def main():
    pop_size = 50
    generations = 40

    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.3, ngen=generations, verbose=False)

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    print("Top 10 Plans on the Pareto Front:\n")
    for ind in pareto[:10]:
        selected = [courses[i] for i in range(len(courses)) if ind[i] == 1]
        ects, grade, workload = evaluate(ind)
        print(f"ECTS: {ects}, Avg Grade: {grade:.2f}, Workload: {workload}h")
        for c in selected:
            print(f"  - {c.name}")
        print()

    # --- Plot Pareto Front ---
    grades = [evaluate(ind)[1] for ind in pareto]
    workloads = [evaluate(ind)[2] for ind in pareto]
    labels = [f"{evaluate(ind)[0]} ECTS" for ind in pareto]

    fig, ax = plt.subplots()
    ax.scatter(workloads, grades, color='green')
    for i, label in enumerate(labels[:10]):
        ax.annotate(label, (workloads[i], grades[i]), fontsize=8)

    ax.set_xlabel("Total Workload (hours)")
    ax.set_ylabel("Average Grade")
    ax.set_title("Pareto Front: Grade vs. Workload")
    ax.invert_yaxis()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
