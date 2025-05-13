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

import warnings
from collections import defaultdict
from itertools import combinations
from typing import Hashable, Optional

import networkx as nx
import numpy as np
from dimod import DiscreteQuadraticModel
from dimod.variables import Variables
from dwave.optimization import Model
from dwave.optimization.generators import capacitated_vehicle_routing
from dwave.system import LeapHybridDQMSampler, LeapHybridNLSampler

from app_configs import DEPOT_LABEL
from solver.ckmeans import CKMeans


class CapacitatedVehicleRoutingProblem:
    """A class to handle data and operations related to Multi-vehicle routing problem.

    Args:
        cost_function: The cost function that takes two coordinates and two labels and
            computes the cost.
    """

    def __init__(self, cost_function) -> None:
        self._cost_callback = cost_function

        self._depots = Variables()
        self._clients = Variables()
        self._vehicles = Variables()

        self._coordinates = {}
        self._solution = {}

        self._vehicle_capacity = {}
        self._costs = {}
        self._demand = {}
        self._paths = {}

        self._optimization = {}

    @property
    def solution(self) -> dict[Hashable, nx.DiGraph]:
        """Solution for the problem."""
        return self._solution

    @property
    def paths(self) -> dict[int, list[int]]:
        """Solution paths for each of the vehicles."""
        return self._paths

    @property
    def vehicles(self) -> Variables:
        """Variables of vehicles by labels and capacities."""
        return self._vehicles

    @property
    def depots(self) -> Variables:
        """Variables of depot(s) by coordinates."""
        return self._depots

    @property
    def demand(self) -> dict[int, int]:
        """Dictionary of client labels and demands"""
        return self._demand

    @property
    def clients(self) -> int:
        """List of client labels."""
        return self._clients

    @property
    def locations(self) -> dict[int, tuple[float, float]]:
        """Dictionary of coordinates for each client location."""
        return self._coordinates

    @property
    def costs(self) -> dict[tuple[float, float], float]:
        """Dictionary of costs for each edge."""
        return self._costs

    def add_depots(self, coordinates: dict[int, tuple]) -> None:
        """Add depot(s) by coordinates.

        Args:
            coordinates: Coordinates for each added depot.
        """
        self._coordinates.update(coordinates)
        for label in coordinates:
            if label in self._clients:
                raise ValueError(f"{DEPOT_LABEL} cannot be in the same location as a client.")

            self._depots._append(label)

    def add_clients(self, coordinates: dict, demand: dict) -> None:
        """Add clients by coordinates and supply demand.

        Args:
            coordinates: A dictionary of label, coordinates.
            demand: A dictionary of client labels and demands.
        """
        for label, co_1 in coordinates.items():
            for key, co_2 in self._coordinates.items():
                if label in self._depots:
                    continue

                # add cost for each new added edge (directed)
                self._costs[label, key] = self._cost_callback(co_1, co_2, label, key)
                self._costs[key, label] = self._cost_callback(co_2, co_1, key, label)

            # add new coordinate to clients and add/update existing coordinates
            self._clients._append(label)
            self._coordinates[label] = co_1

        self._demand.update(demand)

    def add_vehicles(self, capacity: dict) -> None:
        """Add vehicles by labels and capacities.

        Args:
            capacity: A dictionary of vehicle labels and capacities.
        """
        for label in capacity:
            self._vehicles._append(label)
        self._vehicle_capacity.update(capacity)

    def _get_nl(self) -> None:
        """Get and set NL model."""
        self._optimization["nl"] = self.generate_nl_model()

    def solve_hybrid_nl(self, time_limit: Optional[float] = None) -> None:
        """Find vehicle routes using Hybrid NL Solver.

        Args:
            time_limit: Time limit for the NL solver.
        """
        if not self._clustering_feasible():
            raise ValueError("Clustering not feasible due to demand being higher than capacity.")

        sampler = LeapHybridNLSampler()

        # Get and set the NL model
        self._get_nl()

        sampler.sample(self._optimization["nl"], time_limit=time_limit, label="Example - MVRP")

        self.parse_solution_nl()

    def cluster_dqm(
        self, capacity_penalty_strength: float, time_limit: Optional[float] = None, **kwargs
    ) -> None:
        """Cluster the client locations using the DQM.

        Other keyword args are passed on to the LeapHybridDQMSampler.

        Args:
            capacity_penalty_strength (float): Dictates the penalty for violating vehicle capacity.
            time_limit: Time limit for the DQM sampler.
        """
        if not self._clustering_feasible():
            raise ValueError("Clustering not feasible due to demand being higher than capacity.")

        sampler = LeapHybridDQMSampler()

        # get and set the DQM model
        self._get_clustering_dqm(capacity_penalty_strength=capacity_penalty_strength)

        if sampler.min_time_limit(self._optimization["dqm"]) > (time_limit or 5):
            warnings.warn("Defaulting to minimum time limit for Leap Hybrid DQM Sampler.")

            # setting time_limit to None uses the minimum time limit
            time_limit = None

        res = sampler.sample_dqm(
            self._optimization["dqm"],
            time_limit=time_limit,
            label="Example - MVRP",
            **kwargs
        )
        res.resolve()

        sample = res.first.sample
        assignments = defaultdict(list)
        for v in self._clients:
            assignments[v].append(self._vehicles[int(sample[v])])

        capacity_violation = {}
        for k in self._vehicles:
            capacity_violation[k] = -self._vehicle_capacity[k]

        for v in self._clients:
            k = int(sample[v])
            capacity_violation[self._vehicles[k]] += self._demand[v]

        self._optimization["assignments"] = assignments
        self._optimization["capacity_violation"] = assignments

    def _get_clustering_dqm(self, capacity_penalty_strength) -> None:
        """Get and set DQM and offset."""
        self._optimization["dqm"], offset = self.construct_clustering_dqm(capacity_penalty_strength)
        self._optimization["dqm_offset"] = offset

    def cluster_kmeans(self, time_limit=None) -> None:
        """Cluster the client locations using the K-Means classical method.

        Args:
            time_limit: Time limit for the K-Means clusterer.
        """
        clusterer = CKMeans(k=len(self._vehicles))

        locations = [self.locations[k] for k in self._clients]
        demand = [self.demand[k] for k in self._clients]
        capacity = [self._vehicle_capacity[k] for k in self._vehicles]

        assignments = clusterer.predict(locations, demand, capacity, time_limit or 5)

        assignments = list(map(lambda x: [self._vehicles[int(x)]], assignments))
        assignments = dict(zip(self._clients, assignments))

        capacity_violation = {}
        for k in self._vehicles:
            capacity_violation[k] = -self._vehicle_capacity[k]

        self._optimization["assignments"] = assignments
        self._optimization["capacity_violation"] = assignments

    def solve_tsp_heuristic(self) -> None:
        """Solve the travelling salesman problem for each cluster."""
        clusters = {vehicle_id: list(self.depots) for vehicle_id, _ in enumerate(self._vehicles)}

        # invert self.assignments dictionary to dict[vehicle_id, location_id]
        for location_id, cluster in self.assignments.items():
            for vehicle_id in cluster:
                clusters[vehicle_id].append(location_id)

        for vehicle_id, cluster in clusters.items():
            idx = {id: i for i, id in enumerate(cluster)}

            G = nx.MultiDiGraph()
            G.add_nodes_from(range(len(cluster)))
            for coord in combinations(cluster, 2):
                coord_reverse = tuple(reversed(coord))
                G.add_edge(idx[coord[0]], idx[coord[1]], weight=self.costs[coord])
                G.add_edge(idx[coord[1]], idx[coord[0]], weight=self.costs[coord_reverse])

            greedy_tsp = nx.approximation.traveling_salesman.greedy_tsp
            path = nx.approximation.traveling_salesman_problem(G, cycle=False, method=greedy_tsp)

            path += [path[0]]
            cluster += [cluster[0]]
            edges = [(cluster[n], cluster[path[i + 1]]) for i, n in enumerate(path[:-1])]

            self._paths[vehicle_id] = dict(enumerate(path))
            self._solution[vehicle_id] = nx.DiGraph(edges)

    def _clustering_feasible(self) -> bool:
        """Whether clustering is feasible based on total capacity >= demand."""
        total_demand = sum(self._demand.values())
        total_capacity = sum(self._vehicle_capacity.values())
        return total_capacity >= total_demand

    @property
    def assignments(self) -> dict[int, list[int]]:
        """The assignment of locations to vehicles in the clustering step.

        Returns:
             dict: A dictionary with labels as keys, and a list of
             vehicle that the location is assigned to as values.

        """
        return self._optimization.get("assignments", {})

    def construct_clustering_dqm(
        self, capacity_penalty_strength
    ) -> tuple[DiscreteQuadraticModel, float]:
        """Construct the DQM used for clustering.

        Args:
            capacity_penalty_strength (float): Dictates the penalty for violating vehicle capacity.

        Returns:
            DiscreteQuadraticModel, float: The DQM and offset.
        """
        dqm = DiscreteQuadraticModel()
        num_vehicles = len(self._vehicle_capacity)
        for v in self.demand:
            dqm.add_variable(num_vehicles, v)

        max_capacity = max(self._vehicle_capacity.values())
        precision = 1 + int(np.ceil(np.log2(max_capacity)))

        slacks = {
            (k, i): "s_capacity_{}_{}".format(k, i)
            for k in self._vehicle_capacity
            for i in range(precision)
        }

        for s in slacks.values():
            dqm.add_variable(2, s)

        for u, v in combinations(self.demand, r=2):
            for idk, k in enumerate(self._vehicle_capacity):
                dqm.set_quadratic_case(u, idk, v, idk, self.costs[u, v] + self.costs[v, u])

        capacity_penalty = {k: capacity_penalty_strength for k in self._vehicle_capacity}

        offset = 0
        for idk, k in enumerate(self._vehicle_capacity):
            slack_terms = [(slacks[k, i], 1, 2**i) for i in range(precision)]
            dqm.add_linear_equality_constraint(
                [(v, idk, self.demand[v]) for v in self.demand] + slack_terms,
                constant=-self._vehicle_capacity[k],
                lagrange_multiplier=capacity_penalty[k],
            )

            offset += capacity_penalty[k] * self._vehicle_capacity[k] ** 2
        return dqm, offset

    def generate_nl_model(self) -> Model:
        """Follows the NL solver formulation of the CVRP.

        Returns:
            Model: The NL Model.
        """

        # Take maxium vehicle capacity. Vehicle capacity should be updated to only allow
        # one value for all vehicles or update NL solution to allow multiple capacities.
        max_capacity = max(self._vehicle_capacity.values())
        num_vehicles = len(self._vehicles)
        all_locations = [*self._depots, *self._clients]

        # Convert demand dictionary to array
        demand = np.zeros(len(all_locations))
        for index, client in enumerate(self.clients):
            demand[index + 1] = self._demand[client]  # add 1 to skip depot

        # Generate cost/distance matrices
        distances = np.zeros((len(all_locations), len(all_locations)))
        for i, location_i in enumerate(all_locations):
            for j, location_j in enumerate(all_locations):
                if i != j:
                    distances[i, j] = self._costs[location_i, location_j]

        model = capacitated_vehicle_routing(demand, num_vehicles, max_capacity, distances)

        return model

    def _recompute_objective(self, solution):
        """Compute the objective given a solution."""

        all_locations = [*self._depots, *self._clients]
        num_vehicles = len(self._vehicle_capacity)
        total_cost = 0

        assert len(solution) == num_vehicles

        # Compute total cost for the solution.
        for r in solution:
            if len(r) == 0:
                continue

            for index, location in enumerate([0, *r[:-1]]):
                total_cost += self._costs[all_locations[location], all_locations[r[index]]]

            total_cost += self._costs[
                all_locations[r[-1]], all_locations[0]  # Go back to depot
            ]

        return total_cost

    def _check_feasibility(self, solution):
        """Check whether the given solution is feasible"""

        # Take maxium vehicle capacity. Vehicle capacity should be updated to only allow
        # one value for all vehicles or update NL solution to allow multiple capacities.
        max_capacity = max(self._vehicle_capacity.values())
        num_vehicles = len(self._vehicle_capacity)

        # Convert demand dictionary to array
        demand = np.zeros((len(self._clients) + 1))
        for index, client in enumerate(self.clients):
            demand[index + 1] = self._demand[client]

        assert len(solution) == num_vehicles

        for r in solution:
            # If route has no locations the vehicle never left the depot or
            # if demand exceeds capacity.
            if len(r) == 0 or demand[r].sum() > max_capacity:
                return False

        return True

    def _get_solution(self, tolerance=1e-6):
        """Extract solution and check feasibility"""
        model = self._optimization["nl"]
        num_states = model.states.size()
        for i in range(num_states):
            # extract the solution
            decision = next(model.iter_decisions())
            solution_candidate = [[int(v) + 1 for v in route.state(i)] for route in decision.iter_successors()]
            if not solution_candidate:
                continue

            solver_objective = model.objective.state(i)
            assert abs(solver_objective - self._recompute_objective(solution=solution_candidate)) < tolerance

            solver_feasibility = True
            for c in model.iter_constraints():
                if c.state(i) < 0.5:
                    solver_feasibility = False

            # Check feasibility and if feasible, return
            assert solver_feasibility == self._check_feasibility(solution=solution_candidate)
            if solver_feasibility:
                return solution_candidate
            print(f"Sample {i} is infeasible")

        raise ValueError("No feasible solution found.")

    def parse_solution_nl(self) -> None:
        """Checks the solutions from the NL solver (attached to the model) and outputs the parsed ones.
        """

        all_locations = [*self._depots, *self._clients]

        solution = self._get_solution()

        for vehicle_id, destinations in enumerate(solution):
            # Add depot and convert to node IDs.
            route = (
                [all_locations[0]]
                + [all_locations[destination] for destination in destinations]
                + [all_locations[0]]
            )
            self._paths[vehicle_id] = route
            edges = [(n, route[i + 1]) for i, n in enumerate(route[:-1])]
            self._solution[vehicle_id] = nx.DiGraph(edges)
