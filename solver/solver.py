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

import time
from enum import Enum
from typing import Any, Hashable, NamedTuple

import networkx as nx
import numpy as np

from solver.cvrp import CapacitatedVehicleRoutingProblem


class VehicleType(Enum):
    TRUCKS = 0
    DELIVERY_DRONES = 1


class SamplerType(Enum):
    NL = 0
    DQM = 1
    KMEANS = 2


class RoutingProblemParameters(NamedTuple):
    """Structure to hold all provided problem parameters.

    Args:
        map_network: ``nx.MultiDiGraph`` where nodes and edges represent locations and routes.
        depot_id: Node ID of the depot location.
        client_subset: client_subset: List of client IDs in the map's graph.
        num_clients: Number of locations to be visited.
        num_vehicles: Number of vehicles to deploy on routes.
        sampler_type: Sampler type to use in solving CVRP.
        time_limit: Time limit in seconds to run optimization for.
    """

    map_network: nx.MultiDiGraph
    depot_id: int
    client_subset: list
    num_clients: int
    num_vehicles: int
    vehicle_type: VehicleType
    sampler_type: SamplerType
    time_limit: float


class Solver:
    """Solver class to run the routing problem and store the solution.

    Args:
        parameters: NamedTuple that specifies all problem details.
    """

    def __init__(self, parameters: RoutingProblemParameters) -> None:
        self._parameters = parameters
        self._solution = None

        if self.vehicle_type is VehicleType.TRUCKS:
            self._paths_and_lengths = dict(
                nx.all_pairs_dijkstra(parameters.map_network, weight="length")
            )
        else:  # if vehicle_type is DELIVERY_DRONES
            self._paths_and_lengths = None

    def __getattr__(self, name: str) -> Any:
        """Gate routing problem parameters."""
        # checks for attributes in 'self._parameters' if not found in class
        return object.__getattribute__(self._parameters, name)

    @property
    def paths_and_lengths(self) -> dict:
        """All paths between locations/nodes and distances/weights between them."""
        return self._paths_and_lengths

    @property
    def solution(self) -> dict[Hashable, nx.DiGraph]:
        """Solution for the problem."""
        return self._solution

    def cost_between_nodes(self, p1, p2, start, end) -> float:
        """Calculate the cost (e.g., length) between two locations.

        Args:
            p1: Coordinates for first location (used for drones).
            p2: Coordinates for second location (used for drones).
            start: Start node label (used for trucks).
            end: End node label (used for trucks).
        """
        if self.vehicle_type is VehicleType.TRUCKS:
            return self.paths_and_lengths[start][0][end]

        radius_earth = 6371000  # meters
        lat1_rad, lat2_rad = np.deg2rad((p1[0], p2[0]))
        diff_lat_rad, diff_lon_rad = np.deg2rad((p2[0] - p1[0], p2[1] - p1[1]))

        t1 = np.sin(diff_lat_rad / 2) ** 2
        t2 = np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(diff_lon_rad / 2) ** 2
        return 2 * radius_earth * np.arcsin(np.sqrt(t1 + t2))

    def generate(self) -> float:
        """Generate map with solution routes plotted, map centered on depot location, for drone routes.

        Returns:
            float: The wall clock time for finding the solution.
        """
        start_time = time.perf_counter()

        demand = nx.get_node_attributes(self.map_network, "demand")
        depot = {
            self.depot_id: (
                self.map_network.nodes[self.depot_id]["y"],
                self.map_network.nodes[self.depot_id]["x"],
            )
        }
        clients = {
            client_id: (
                self.map_network.nodes[client_id]["y"],
                self.map_network.nodes[client_id]["x"],
            )
            for client_id in self.client_subset
        }

        capacity = -(-sum(demand.values()) // self.num_vehicles)
        cvrp = CapacitatedVehicleRoutingProblem(cost_function=self.cost_between_nodes)
        cvrp.add_depots(depot)
        cvrp.add_clients(clients, demand)
        cvrp.add_vehicles(
            {k: capacity for k in range(self.num_vehicles)}
        )

        if self.sampler_type is SamplerType.NL:
            cvrp.solve_hybrid_nl(capacity=capacity, time_limit=self.time_limit)
        else:
            # DQM and K-Means require a two-step solution: clustering + tsp
            if self.sampler_type is SamplerType.DQM:
                cvrp.cluster_dqm(capacity=1.0, time_limit=self.time_limit)
            else:
                cvrp.cluster_kmeans(time_limit=self.time_limit)

            cvrp.solve_tsp_heuristic()

        wall_clock_time = time.perf_counter() - start_time
        self._solution = cvrp.solution

        return wall_clock_time
