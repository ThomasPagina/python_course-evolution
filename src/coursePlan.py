import itertools
import matplotlib.pyplot as plt

# --- Data Model ---
class Course:
    def __init__(self, name: str, ects: int, predicted_grade: float, workload_hours: int):
        self.name = name
        self.ects = ects
        self.predicted_grade = predicted_grade  # 1.0 is best, 4.0 is lowest passing grade
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

# --- Multi-Objective Evaluation ---
def evaluate_plan(plan):
    total_ects = sum(course.ects for course in plan)
    average_grade = sum(course.predicted_grade for course in plan) / len(plan)
    total_work = sum(course.workload_hours for course in plan)
    return total_ects, average_grade, total_work

# --- Generate and Evaluate All Combinations ---
def find_best_plans(courses, max_courses=5):
    best_plans = []
    for r in range(1, max_courses + 1):
        for combination in itertools.combinations(courses, r):
            ects, grade, work = evaluate_plan(combination)
            best_plans.append((ects, grade, work, combination))
    return best_plans

# --- Plot Top Plans ---
def plot_top_plans(plans):
    top_plans = plans[:10]
    grades = [p[1] for p in top_plans]
    workloads = [p[2] for p in top_plans]
    labels = [f"{p[0]} ECTS" for p in top_plans]

    plt.figure(figsize=(10, 6))
    plt.scatter(workloads, grades, c='blue')
    for i, label in enumerate(labels):
        plt.annotate(label, (workloads[i], grades[i]), fontsize=8)

    plt.xlabel("Total Workload (hours)")
    plt.ylabel("Average Grade")
    plt.title("Top 10 Plans: Grade vs. Workload")
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    plans = find_best_plans(courses)

    # Sort by highest ECTS, then best grade, then lowest effort
    sorted_plans = sorted(plans, key=lambda x: (-x[0], x[1], x[2]))

    print("Top 10 optimized course plans:\n")
    for ects, grade, work, combo in sorted_plans[:10]:
        print(f"ECTS: {ects}, Avg Grade: {grade:.2f}, Workload: {work}h")
        for c in combo:
            print(f"  - {c.name}")
        print()

    # Plot only Top 10
    plot_top_plans(sorted_plans)
