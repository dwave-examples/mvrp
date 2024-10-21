import random
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Integer, Real

# Parameters
num_locations = 7
depot = (0, 0)

# Randomly generate initial N locations (x, y) in 2D space
locations = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(num_locations)]
print(f"Depot is located at: {depot}")
print(f"These are the current locations: {locations}")

# Using Hybrid CQM (Constrained Quadratic Model) Solver
def solve_with_cqm():
    print("Solving with Hybrid CQM Solver...")

    # Initialize the Constrained Quadratic Model
    cqm = ConstrainedQuadraticModel()

    # Variables for each location (x, y coordinates)
    location_vars = {}
    for i in range(num_locations):
        location_vars[i] = {
            'x': Real(f'x_{i}', lower_bound=-10, upper_bound=10),
            'y': Real(f'y_{i}', lower_bound=-10, upper_bound=10)
        }

    # Objective: Minimize the sum of Manhattan distances from each location to the depot
    objective = 0
    for i in range(num_locations):
        x_i, y_i = location_vars[i]['x'], location_vars[i]['y']

        # Add auxiliary variables for the absolute value |x_i - depot[0]|, |y_i - depot[1]|
        abs_x = Real(f'abs_x_{i}', lower_bound=0)
        abs_y = Real(f'abs_y_{i}', lower_bound=0)

        # Add constraints to ensure abs_x represents the absolute value using two inequalities
        cqm.add_constraint(abs_x - (x_i - depot[0]) == 0, label=f'abs_x_pos_{i}')
        cqm.add_constraint(abs_x + (x_i - depot[0]) == 0, label=f'abs_x_neg_{i}')

        # Add constraints to ensure abs_y represents the absolute value using two inequalities
        cqm.add_constraint(abs_y - (y_i - depot[1]) == 0, label=f'abs_y_pos_{i}')
        cqm.add_constraint(abs_y + (y_i - depot[1]) == 0, label=f'abs_y_neg_{i}')

        # Add the auxiliary variables to the objective (Manhattan distance)
        objective += abs_x + abs_y

    # Set the objective in the CQM
    cqm.set_objective(objective)

    # Solve with the Leap Hybrid CQM sampler
    sampler = LeapHybridCQMSampler()
    result = sampler.sample_cqm(cqm)

    # Get the best solution
    best_solution = result.first.sample

    print("\nBest solution with CQM:")
    for i in range(num_locations):
        x = best_solution[f'x_{i}']
        y = best_solution[f'y_{i}']
        abs_x = abs(depot[0] - x)
        abs_y = abs(depot[1] - y)
        print(f"Location {i}:")
        print(f"  Coordinates: (x_{i} = {x}, y_{i} = {y})")
        print(f"  Abs values: (|x_{i} - depot_x| = {abs_x}, |y_{i} - depot_y| = {abs_y})\n")

# Run the CQM case
solve_with_cqm()
