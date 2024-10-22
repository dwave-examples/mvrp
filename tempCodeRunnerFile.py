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
    ax.set_xlim(0, 25)
    ax.set_ylim(-15, 15)

    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Optimization of Locations within a Rectangular Box')
    ax.legend()

    # Show the plot
    plt.grid(True)
    plt.show()