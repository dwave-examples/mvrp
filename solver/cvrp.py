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

from collections import defaultdict
from itertools import combinations, permutations
from typing import Hashable, Optional

import networkx as nx
import numpy as np
from dimod import DiscreteQuadraticModel
from dimod.variables import Variables
from dwave.system import LeapHybridDQMSampler
from python_tsp.heuristics import solve_tsp_local_search

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
                raise ValueError("Depot cannot be in the same location as a client.")

            self._depots._append(label)

    def add_clients(self, coordinates: dict, demand: dict) -> None:
        """Add clients by coordinates and supply demand.

        Args:
            coordinates: A dictionary of force label, coordinates.
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

    def cluster_dqm(self, capacity: float, time_limit: Optional[float] = None, **kwargs) -> None:
        """Cluster the client locations using the DQM.

        Other keyword args are passed on to the LeapHybridDQMSampler.

        Args:
            capacity: A dictionary of vehicle labels and capacities.
            time_limit: Time limit for the DQM sampler.
        """
        if not self._clustering_feasible():
            raise ValueError("Clustering not feasible due to demand being higher than capacity.")

        sampler = LeapHybridDQMSampler()

        # get and set the DQM model
        self._get_clustering_dqm(capacity_penalty=capacity)

        res = sampler.sample_dqm(self._optimization["dqm"], time_limit=time_limit, **kwargs)
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

    def _get_clustering_dqm(self, capacity_penalty) -> None:
        """Get and set DQM and offset."""
        self._optimization["dqm"], offset = self.construct_clustering_dqm(capacity_penalty)
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

            weight_matrix = np.zeros((len(cluster), len(cluster)))
            for coord in combinations(cluster, 2):
                coord_reverse = tuple(reversed(coord))
                weight_matrix[idx[coord[0]], idx[coord[1]]] = self.costs[coord]
                weight_matrix[idx[coord[1]], idx[coord[0]]] = self.costs[coord_reverse]

            path, _ = solve_tsp_local_search(weight_matrix)

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
             dict: A dictionary with force labels as keys, and a list of
             vehicle that the location is assigned to as values.

        """
        return self._optimization.get("assignments", {})

    def construct_clustering_dqm(self, capacity) -> tuple[DiscreteQuadraticModel, float]:
        """Construct the DQM used for clustering.

        Args:
            capacity: The amount of supply that the vehicle can carry.

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

        capacity_penalty = {k: capacity for k in self._vehicle_capacity}

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
