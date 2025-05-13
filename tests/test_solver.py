# Copyright 2025 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from dimod import DiscreteQuadraticModel
import networkx as nx
import numpy as np
import pytest
import dwave.optimization

import solver
from solver.ckmeans import CKMeans
# from solver.cvrp import CapacitatedVehicleRoutingProblem
from solver.solver import Solver

# default values used in fixtures and for solutions
from tests.conftest import DEFAULT_COST, FEASIBLE_NL_SOLUTION, NUM_LOCATIONS, NUM_VEHICLES


class TestCVRP:
    """TODO"""

    def test_add_depots(self, cvrp):
        """TODO"""
        coordinates = {10: (12.3, 23.4), 12: (45.6, 56.7)}
        cvrp.add_depots(coordinates)

        assert cvrp.depots == [10, 12]
        assert cvrp.locations == coordinates

    def test_add_clients(self, cvrp):
        """TODO"""
        coordinates = {10: (12.3, 23.4), 12: (45.6, 56.7)}
        demand = {10: 13, 12: 24}
        cvrp.add_clients(coordinates, demand)

        assert cvrp.clients == [10, 12]
        assert cvrp.locations == coordinates
        assert cvrp.demand == demand

        # test adding more clients works
        more_coordinates = {4: (11.3, 23.6), 2: (25.6, 46.2)}
        more_demand = {4: 6, 2: 12}
        cvrp.add_clients(more_coordinates, more_demand)

        assert cvrp.clients == [10, 12, 4, 2]
        assert cvrp.locations == {**coordinates, **more_coordinates}
        assert cvrp.demand == {**demand, **more_demand}


    def test_add_vehicles(self, cvrp):
        """TODO"""
        capacity = {2: None, 3: None}
        vehicles = list(capacity.keys())
        cvrp.add_vehicles(capacity)

        assert cvrp.vehicles == vehicles

    def test_solve_hybrid_nl(self, cvrp, mock_sample_nl):
        """TODO"""
        cvrp.add_depots({10: (12.3, 23.4)})
        cvrp.add_clients(
            {i: (1, 2) for i in range(NUM_LOCATIONS)},
            {i: 1 for i in range(NUM_LOCATIONS)} | {10: 0}
        )
        cvrp.add_vehicles({i: 10 for i in range(NUM_VEHICLES)})
        cvrp.solve_hybrid_nl()

        assert cvrp.solution
        for i, path in cvrp.paths.items():
            # add in depot location at start and end for each path
            assert path == [10] + FEASIBLE_NL_SOLUTION[i] + [10]


    @pytest.mark.parametrize("capacity_penalty_strength", [0.5, 1.0])
    @pytest.mark.parametrize("cluster_func", ["cluster_dqm", "cluster_kmeans"])
    def test_clustering(self, capacity_penalty_strength, cluster_func, cvrp, mock_sample_dqm):
        """TODO"""
        cvrp.add_depots({10: (12.3, 23.4)})
        cvrp.add_clients(
            {i: (1, 2) for i in range(NUM_LOCATIONS)},
            {i: 1 for i in range(NUM_LOCATIONS)} | {10: 0}
        )
        cvrp.add_vehicles({i: 10 for i in range(NUM_VEHICLES)})

        getattr(cvrp, cluster_func)(capacity_penalty_strength)

        # check that assingments are set (not if they're reasonable)
        assert cvrp.assignments is not None

        # check that no solution is created
        assert not cvrp.solution

    def test_solve_tsp_heuristic(self, cvrp):
        """TODO"""
        cvrp.add_depots({10: (12.3, 23.4)})
        cvrp.add_clients(
            {i: (1, 2) for i in range(NUM_LOCATIONS)},
            {i: 1 for i in range(NUM_LOCATIONS)} | {10: 0}
        )
        cvrp.add_vehicles({i: 10 for i in range(NUM_VEHICLES)})

        # add random assignments to the vehicles
        cvrp._optimization["assignments"] = {0: [0], 1: [0], 2: [1], 3: [0]}

        cvrp.solve_tsp_heuristic()

        assert cvrp.solution
        assert cvrp.paths

    @pytest.mark.parametrize("capacity_penalty_strength", [0.5, 1.0])
    def test_construct_clustering_dqm(self, cvrp, capacity_penalty_strength):
        """TODO"""
        capacity_per_vehicle = 10
        cvrp.add_depots({10: (12.3, 23.4)})
        cvrp.add_clients(
            {i: (1, 2) for i in range(NUM_LOCATIONS)},
            {i: 1 for i in range(NUM_LOCATIONS)} | {10: 0}
        )
        cvrp.add_vehicles({i: capacity_per_vehicle for i in range(NUM_VEHICLES)})

        dqm, offset = cvrp.construct_clustering_dqm(capacity_penalty_strength)

        assert isinstance(dqm, DiscreteQuadraticModel)

        expected_offset = capacity_penalty_strength * capacity_per_vehicle ** 2 * NUM_VEHICLES
        assert offset == expected_offset

    def test_generate_nl_model(self, cvrp):
        """TODO"""
        cvrp.add_depots({10: (12.3, 23.4)})
        cvrp.add_clients(
            {i: (1, 2) for i in range(NUM_LOCATIONS)},
            {i: 1 for i in range(NUM_LOCATIONS)} | {10: 0}
        )
        cvrp.add_vehicles({i: 10 for i in range(NUM_VEHICLES)})
        nlm = cvrp.generate_nl_model()

        assert isinstance(nlm, dwave.optimization.Model)

    def test_parse_solution_nl(self, cvrp, monkeypatch):
        """TODO"""
        cvrp.add_depots({10: (12.3, 23.4)})
        cvrp.add_clients(
            {i: (1, 2) for i in range(NUM_LOCATIONS)},
            {i: 1 for i in range(NUM_LOCATIONS)} | {10: 0}
        )
        cvrp.add_vehicles({i: 10 for i in range(NUM_VEHICLES)})

        # high capacity to make it (very) feasible
        capacity_per_vehicle = 10

        # 0 for depot, 1 for the rest
        demand = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        distances = np.full((NUM_LOCATIONS + 1, NUM_LOCATIONS + 1), DEFAULT_COST)
        np.fill_diagonal(distances, 0)

        model = dwave.optimization.generators.capacitated_vehicle_routing(
            demand, NUM_VEHICLES, capacity_per_vehicle, distances
        )
        model.states.resize(1)
        for decision in model.iter_decisions():
            decision.set_state(0, FEASIBLE_NL_SOLUTION)

        cvrp._optimization["nl"] = model

        cvrp.parse_solution_nl()

        assert cvrp.solution
        for i, path in cvrp.paths.items():
            # add in depot location at start and end for each path
            assert path == [10] + FEASIBLE_NL_SOLUTION[i] + [10]

    def test_nl_feasibility(self, cvrp, monkeypatch):
        """TODO"""
        cvrp._optimization["nl"] = dwave.optimization.generators.capacitated_vehicle_routing(
            [0, 1, 1], 2, 5, [[0, 4, 4], [4, 0, 4], [4, 4, 0]]
        )

        # without sampling the model, no solutions will be found
        with pytest.raises(ValueError, match="No feasible solution found."):
            cvrp.parse_solution_nl()

class TestCKMeans:
    """TODO"""

    @pytest.mark.parametrize("k", [2, 7])
    @pytest.mark.parametrize("max_iterations", [200, 500])
    def test_predict_once(self, k, max_iterations):
        """TODO"""
        ckmeans = CKMeans(k, max_iterations)



class TestSolver:
    """TODO"""

    def test_solver(self, parameters_with_combos, mock_sample_dqm, mock_sample_nl):
        """TODO"""
        solver = Solver(parameters_with_combos)

        assert solver.map_network == parameters_with_combos.map_network
        assert solver.depot_id == parameters_with_combos.depot_id
        assert solver.client_subset == parameters_with_combos.client_subset
        assert solver.num_clients == parameters_with_combos.num_clients
        assert solver.num_vehicles == parameters_with_combos.num_vehicles

        assert solver.vehicle_type is parameters_with_combos.vehicle_type
        assert solver.sampler_type is parameters_with_combos.sampler_type

        assert solver.time_limit == parameters_with_combos.time_limit

    def test_generate(self, parameters_with_combos, mock_sample_dqm, mock_sample_nl):
        """TODO"""
        solver = Solver(parameters_with_combos)
        wall_clock_time = solver.generate()

        assert wall_clock_time > 0
        assert len(solver.solution) == parameters_with_combos.num_vehicles

        visited_locations = 0
        for solution in solver.solution.values():
            assert isinstance(solution, nx.Graph)

            # total number of visited locations (minus depot);
            # two vehicles shouldn't visit the same location
            visited_locations += solution.size() - 1

        assert visited_locations == NUM_LOCATIONS

    @pytest.mark.parametrize("imperial", [True, False])
    def test_cost_between_nodes_trucks(self, parameters_trucks, imperial, monkeypatch):
        """TODO"""
        # monkeypatch the `UNITS_IMPERIAL` setting inside `solver.py`
        monkeypatch.setattr(solver.solver, "UNITS_IMPERIAL", imperial)

        # pair of existing start/end nodes
        start, end = 8775189108, 60189310

        expected = dict(nx.all_pairs_dijkstra(parameters_trucks.map_network, weight="length"))[start][0][end]
        cost = Solver(parameters_trucks).cost_between_nodes(p1=(0, 0), p2=(0, 0), start=start, end=end)

        if imperial:
            assert expected / 1609.34 == cost
        else:
            assert expected == cost


    @pytest.mark.parametrize("imperial", [True, False])
    def test_cost_between_nodes_drones(self, parameters_drones, imperial, monkeypatch):
        """TODO"""
        # monkeypatch the `UNITS_IMPERIAL` setting inside `solver.py`
        monkeypatch.setattr(solver.solver, "UNITS_IMPERIAL", imperial)

        # pair of valid coordinates
        p1, p2, = (-23.385846, 150.495835), (-23.380160, 150.499207)
        lat1_rad, lat2_rad = np.deg2rad((p1[0], p2[0]))
        diff_lat_rad, diff_lon_rad = np.deg2rad((p2[0] - p1[0], p2[1] - p1[1]))

        t1 = np.sin(diff_lat_rad / 2) ** 2
        t2 = np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(diff_lon_rad / 2) ** 2

        if imperial:
            # use earth radius in miles (3,958.8)
            expected = 2 * 3_958.8 * np.arcsin(np.sqrt(t1 + t2))
        else:
            # use earth radius in meters (6,371,000)
            expected = 2 * 6_371_000 * np.arcsin(np.sqrt(t1 + t2))

        cost = Solver(parameters_drones).cost_between_nodes(p1=p1, p2=p2, start=None, end=None)

        assert cost == pytest.approx(expected)
