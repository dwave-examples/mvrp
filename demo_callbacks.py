# Copyright 2024 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from collections import defaultdict
from operator import itemgetter
from typing import NamedTuple, Union

import dash
from dash import MATCH, callback_context, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from demo_interface import create_table, generate_locations_layer, generate_solution_layers
from src.demo_enums import SolverType, VehicleType
from src.map import (
    generate_mapping_information,
    get_client_locations,
    get_position,
    get_solution_routes,
)
from src.solver import RoutingProblemParameters, Solver


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    Output({"type": "collapse-trigger", "index": MATCH}, "aria-expanded"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> tuple[str, str]:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger: The (total) number of times a collapse button has been clicked.
        to_collapse_class: Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        A tuple containing:

        - str: The new class name of the thing to collapse.
        - str: The aria-expanded value.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes), "true"
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed", "false"


@dash.callback(
    Output("locations-layer", "children"),
    Output("routes-layer", "children"),
    Output("solution-map", "viewport"),
    Input("num-clients-select", "value"),
)
def render_initial_map(num_clients: int) -> tuple[list, list, dict]:
    """Draws the depot and client locations on the map.

    Runs on page load and whenever the number of locations changes, which clears any
    previous solution. The viewport is only fitted on page load so that changing the number
    of locations keeps the user's current zoom and position.

    Args:
        num_clients: Number of locations.

    Returns:
        A tuple containing:

        - list: Depot and client location markers.
        - list: Solution route lines (always empty, clearing any previous solution).
        - dict: Map viewport fitted to the bounds of the street network on page load only.
    """
    map_network, depot_id, client_subset, map_bounds = generate_mapping_information(num_clients)
    locations = generate_locations_layer(
        get_position(map_network, depot_id),
        get_client_locations(map_network, depot_id, client_subset),
    )

    viewport = dash.no_update
    if ctx.triggered_id is None:  # Only on page load
        viewport = {"bounds": map_bounds, "transition": "fitBounds"}

    return locations, [], viewport


@dash.callback(
    Output("solution-cost-table", "children"),
    Output("solution-cost-table-classical", "children"),
    inputs=[
        Input("run-in-progress", "data"),
        State("stored-results", "data"),
        State("reset-results", "data"),
        State("sampler-type", "data"),
    ],
    prevent_initial_call=True,
)
def update_tables(run_in_progress, stored_results, reset_results, solver_type) -> tuple[list, list]:
    """Update the results tables each time a run is made.

    Args:
        run_in_progress: Whether or not the ``run_optimization`` callback is running.
        stored_results: The results tab from the latest run.
        reset_results: Whether or not to reset the results tables before applying the new one.
        solver_type: The sampler type used in the latest run (``"quantum"`` or ``"classical"``)

    Returns:
        A tuple containing:

        - list: Solution cost table children.
        - list: Classical solution cost table children.
    """
    empty_or_no_update = [] if reset_results else dash.no_update

    if run_in_progress is True:
        raise PreventUpdate

    if solver_type == "classical":
        return empty_or_no_update, stored_results

    return stored_results, empty_or_no_update


def calculate_cost_comparison(
    cost_comparison: dict,
    final_cost: int,
    solver_type: Union[SolverType, int],
    reset_results: bool,
) -> tuple[dict, str]:
    """Calculates cost improvement between DQM and KMEANS.

    Args:
        cost_comparison: Dictionary with solver keys and run cost values.
        final_cost: The total distance cost of the most recent run.
        solver_type: The sampler that was run. Either Quantum Hybrid (DQM)
            (``0`` or ``SolverType.DQM``), Quantum Hybrid (Stride) (``1`` or ``SolverType.STRIDE``), or
            Classical (K-Means) (``2`` or ``SolverType.KMEANS``).
        reset_results: Whether or not to reset wall clock times.

    Returns:
        A tuple containing:

        - dict: Updated dictionary with solver keys and run cost values.
        - str: String stating the quantum hybrid performance improvement.
    """

    # Dict keys must be strings because Dash stores data as JSON
    key = str(solver_type.value if solver_type is SolverType.KMEANS else SolverType.DQM.value)
    cost_comparison_ratio = 1

    if reset_results:
        cost_comparison = {key: final_cost}
    else:
        cost_comparison[key] = final_cost
        if len(cost_comparison) == 2:
            cost_quantum = cost_comparison[str(SolverType.DQM.value)]
            cost_kmeans = cost_comparison[str(SolverType.KMEANS.value)]
            if cost_kmeans:
                cost_comparison_ratio = cost_quantum / cost_kmeans

    performance_improvement_quantum = ""
    if cost_comparison_ratio < 1:
        performance_improvement_quantum = f"The total distance \
            travelled is {1 - cost_comparison_ratio:.2%} \
            less using the quantum hybrid solution."
    return cost_comparison, performance_improvement_quantum


def get_updated_wall_clock_times(
    wall_clock_time: float, solver_type: Union[SolverType, int], reset_results: bool
) -> tuple[str, str]:
    """Determine which wall clock times to update in the UI.

    Args:
        wall_clock_time: Total run time.
        solver_type: The sampler that was run. Either Either Quantum Hybrid (DQM)
            (``0`` or ``SolverType.DQM``), Quantum Hybrid (Stride) (``1`` or ``SolverType.STRIDE``), or
            Classical (K-Means) (``2`` or ``SolverType.KMEANS``).
        reset_results: Whether or not to reset wall clock times.

    Returns:
        A tuple containing:

        - str: Updated kmeans wall clock time.
        - str: Updated hybrid solver wall clock time.
    """
    wall_clock_time_kmeans = ""
    wall_clock_time_quantum = ""
    if solver_type is SolverType.KMEANS:
        wall_clock_time_kmeans = f"{wall_clock_time:.3f}s"
        if not reset_results:
            wall_clock_time_quantum = dash.no_update
    else:
        wall_clock_time_quantum = f"{wall_clock_time:.3f}s"
        if not reset_results:
            wall_clock_time_kmeans = dash.no_update
    return wall_clock_time_kmeans, wall_clock_time_quantum


class RunOptimizationReturn(NamedTuple):
    """Return type for the ``run_optimization`` callback function."""

    locations_layer: list
    routes_layer: list
    cost_table: tuple
    hybrid_table_label: str
    solver_type: str
    reset_results: bool
    parameter_hash: str
    performance_improvement_quantum: str
    cost_comparison: dict
    problem_size: int
    search_space: str
    wall_clock_time_classical: str
    wall_clock_time_quantum: str
    num_locations: int
    vehicles_deployed: int


@dash.callback(
    # update map and results
    Output("locations-layer", "children", allow_duplicate=True),
    Output("routes-layer", "children", allow_duplicate=True),
    Output("stored-results", "data"),
    Output("hybrid-table-label", "children"),
    # store the solver used, whether or not to reset results tabs and the
    # parameter hash value used to detect parameter changes
    Output("sampler-type", "data"),
    Output("reset-results", "data"),
    Output("parameter-hash", "data"),
    Output("performance-improvement-quantum", "children"),
    Output("cost-comparison", "data"),
    # updates problem details table
    Output("problem-size", "children"),
    Output("search-space", "children"),
    Output("wall-clock-time-classical", "children"),
    Output("wall-clock-time-quantum", "children"),
    Output("num-locations", "children"),
    Output("vehicles-deployed", "children"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("vehicle-type-select", "value"),
        State("solver-type-select", "value"),
        State("num-vehicles-select", "value"),
        State("solver-time-limit", "value"),
        State("num-clients-select", "value"),
        State("parameter-hash", "data"),
        State("cost-comparison", "data"),
    ],
    running=[
        # show cancel button and hide run button, and disable and animate results tab
        (Output("cancel-button", "style"), {}, {"display": "none"}),
        (Output("run-button", "style"), {"display": "none"}, {}),
        (Output("results-tab", "disabled"), True, False),
        (Output("results-tab", "children"), "Loading...", "Results"),
        # switch to map tab while running
        (Output("tabs", "value"), "map-tab", "map-tab"),
        # block certain callbacks from running until this is done
        (Output("run-in-progress", "data"), True, False),
        (Output("loading", "display"), "show", "auto"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    run_click: int,
    vehicle_type: str,
    solver_type: str,
    num_vehicles: int,
    time_limit: float,
    num_clients: int,
    previous_parameter_hash: str,
    cost_comparison: dict,
) -> RunOptimizationReturn:
    """Run the optimization and update map and results tables.

    This is the main optimization function which is called when the Run optimization button is
    clicked. It used all inputs from the drop-down lists, sliders and text entries and runs the
    optimization, updates the run/cancel buttons, animates (and deactivates) the results tab,
    moves focus to the map tab and updates all relevant HTML entries.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        vehicle_type: Either Trucks (``0`` or ``VehicleType.TRUCKS``) or
            Delivery Drones (``1`` or ``VehicleType.DELIVERY_DRONES``).
        solver_type: Either Quantum Hybrid (DQM) (``0`` or ``SolverType.DQM``),
            Quantum Hybrid (Stride) (``1`` or ``SolverType.STRIDE``), or Classical (K-Means)
            (``2`` or ``SolverType.KMEANS``).
        num_vehicles: The number of vehicles.
        time_limit: The solver time limit.
        num_clients: The number of locations.
        previous_parameter_hash: Previous hash string to detect changed parameters
        cost_comparison: Dictionary with solver keys and run cost values.

    Returns:
        A NamedTuple (RunOptimizationReturn) containing all outputs to be used when updating the
        HTML template (in ``demo_interface.py``). These are:

        - locations_layer: Depot and stop markers on the map, colored by vehicle.
        - routes_layer: Solution route lines on the map, one per vehicle.
        - cost_table: Stores the Solution cost table in the results tab.
        - hybrid_table_label: Label for the hybrid results table (either Stride or DQM).
        - solver_type: The sampler used (``"quantum"`` or ``"classical"``).
        - reset_results: Whether or not to reset the results tables before applying the new one.
        - parameter_hash: Hash string to detect changed parameters.
        - performance_improvement_quantum: Updates quantum performance improvement message.
        - cost_comparison: Keeps track of the difference between classical and hybrid run costs.
        - problem_size: Updates the problem-size entry in the problem details table.
        - search_space: Updates the search-space entry in the problem details table.
        - wall_clock_time_classical: Updates the wall clock time in the Classical table header.
        - wall_clock_time_quantum: Updates the wall clock time in the Hybrid Quantum table header.
        - num_locations: Updates the number of locations in the problem details table.
        - vehicles_deployed: Updates the vehicles-deployed entry in the problem details table.
    """
    vehicle_type = VehicleType(int(vehicle_type))
    solver_type = SolverType(int(solver_type))

    map_network, depot_id, client_subset, _ = generate_mapping_information(num_clients)

    routing_problem_parameters = RoutingProblemParameters(
        map_network=map_network,
        depot_id=depot_id,
        client_subset=client_subset,
        num_clients=num_clients,
        num_vehicles=num_vehicles,
        vehicle_type=vehicle_type,
        solver_type=solver_type,
        time_limit=time_limit,
    )
    routing_problem_solver = Solver(routing_problem_parameters)

    # run problem and generate solution (stored in Solver)
    wall_clock_time = routing_problem_solver.generate()

    routes, solution_cost = get_solution_routes(routing_problem_parameters, routing_problem_solver)
    locations_layer, routes_layer = generate_solution_layers(
        get_position(map_network, depot_id), routes
    )

    problem_size = num_vehicles * num_clients
    search_space = f"{num_vehicles**num_clients:.2e}"

    solution_cost = dict(sorted(solution_cost.items()))
    total_cost = defaultdict(int)
    for cost_info_dict in solution_cost.values():
        for key, value in cost_info_dict.items():
            total_cost[key] += value

    cost_table = create_table(solution_cost, list(total_cost.values()))

    parameter_hash = _get_parameter_hash(**callback_context.states)
    reset_results = parameter_hash != previous_parameter_hash

    cost_comparison, performance_improvement_quantum = calculate_cost_comparison(
        cost_comparison, total_cost["optimized_cost"], solver_type, reset_results
    )

    wall_clock_time_kmeans, wall_clock_time_quantum = get_updated_wall_clock_times(
        wall_clock_time, solver_type, reset_results
    )

    hybrid_table_label = dash.no_update if solver_type is SolverType.KMEANS else solver_type.label

    return RunOptimizationReturn(
        locations_layer=locations_layer,
        routes_layer=routes_layer,
        cost_table=cost_table,
        hybrid_table_label=hybrid_table_label,
        solver_type="classical" if solver_type is SolverType.KMEANS else "quantum",
        reset_results=reset_results,
        parameter_hash=str(parameter_hash),
        performance_improvement_quantum=performance_improvement_quantum,
        cost_comparison=cost_comparison,
        problem_size=problem_size,
        search_space=search_space,
        wall_clock_time_classical=wall_clock_time_kmeans,
        wall_clock_time_quantum=wall_clock_time_quantum,
        num_locations=num_clients,
        vehicles_deployed=num_vehicles,
    )


def _get_parameter_hash(**states) -> str:
    """Calculate a key string for the parameters which reset the results tables.

    The key is the JSON serialization of the parameter values rather than a ``hash()``: it is
    only ever compared for equality, and the built-in ``hash()`` of a string differs between
    processes, which would reset the tables on every run now that ``run_optimization`` runs
    in a background process.
    """
    # list of parameter values that will reset the results tables when changed in the app
    items = [
        "vehicle-type-select.value",
        "num-vehicles-select.value",
        "num-clients-select.value",
        "solver-time-limit.value",
    ]
    return json.dumps(itemgetter(*items)(states))
