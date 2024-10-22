import random
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Integer, Real
import matplotlib.pyplot as plt

# Parameters
num_locations = 7
depot = (0, 0)

# Randomly generate initial N locations (x, y) in 2D space
locations = [(random.randint(10, 20), random.randint(-10, 10)) for _ in range(num_locations)]
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
            'x': Real(f'x_{i}', lower_bound=10, upper_bound=20),
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

    optimized_locations = [(best_solution[f'x_{i}'], best_solution[f'y_{i}']) for i in range(num_locations)]

    print("\nBest solution with CQM within the rectangular box:")
    for i, (x, y) in enumerate(optimized_locations):
        print(f"Location {i}: Coordinates: (x_{i} = {x}, y_{i} = {y})")

        # Plotting the results
    plot_solution(locations, optimized_locations, depot)

def plot_solution(initial_locations, optimized_locations, depot):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the initial locations
    x_initial, y_initial = zip(*initial_locations)
    ax.scatter(x_initial, y_initial, color='blue', label='Initial Locations', marker='o')

    # Plot the optimized locations
    x_optimized, y_optimized = zip(*optimized_locations)
    ax.scatter(x_optimized, y_optimized, color='red', label='Optimized Locations', marker='x')

    # Plot the depot
    ax.scatter(*depot, color='green', label='Depot', marker='s')

    # Plot the rectangular boundary box
    box_x = [10, 20, 20, 10, 10]
    box_y = [-10, -10, 10, 10, -10]
    ax.plot(box_x, box_y, color='black', linestyle='--', label='Rectangular Boundary Box')

    # Set axis limits
    ax.set_xlim(-1, 25)
    ax.set_ylim(-15, 15)

    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Optimization of Locations within a Rectangular Box')
    ax.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Run the CQM case
solve_with_cqm()
