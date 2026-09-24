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

"""Street network generation and map data for the dash-leaflet map.

This module returns plain data (positions, demands and route GeoJSON). The dash-leaflet
components that display it are built in ``demo_interface.py``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import networkx as nx
import numpy as np
import osmnx as ox
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from demo_configs import ADDRESS, DISTANCE, RESOURCES
from src.demo_enums import VehicleType

ox.settings.use_cache = True
ox.settings.overpass_rate_limit = False

# Colorblind palette from seaborn (10 colors) paired with the marker icon of the same color.
# Routes are assigned colors from the end of this list.
PALETTE = [
    ("location_blue", "#56b4e9"),
    ("location_yellow", "#ece133"),
    ("location_grey", "#949494"),
    ("location_pink", "#fbafe4"),
    ("location_beige", "#ca9161"),
    ("location_purple", "#cc78bc"),
    ("location_orange", "#d55e00"),
    ("location_green", "#029e73"),
    ("location_gold", "#de8f05"),
    ("location_navy", "#0173b2"),
]


class Location(NamedTuple):
    """A client location on the map.

    Args:
        node_id: Node ID of the location in the street network.
        position: ``(latitude, longitude)`` of the location.
        demand: Demand at the location for each resource in ``RESOURCES``.
    """

    node_id: int
    position: tuple[float, float]
    demand: list[int]


class Stop(NamedTuple):
    """A client location visited by a vehicle.

    Args:
        location: The client location.
        stop_number: The position of this stop along the vehicle's route (the depot is stop 0).
    """

    location: Location
    stop_number: int


class VehicleRoute(NamedTuple):
    """The route driven (or flown) by a single vehicle.

    Args:
        vehicle_id: One-based vehicle number.
        color: Hex color used to draw the route and matching the marker icon.
        icon_name: File stem of the marker icon in ``static/location_icons``.
        stops: Client locations visited, in route order (excludes the depot).
        path: GeoJSON ``FeatureCollection`` of ``LineString`` features tracing the route.
    """

    vehicle_id: int
    color: str
    icon_name: str
    stops: list[Stop]
    path: dict


def _get_coordinates(node_index_map: dict) -> NDArray:
    """Returns an array of coordinates for all nodes."""
    coordinates = np.zeros((len(node_index_map), 2))
    for node_index, node in node_index_map.items():
        coordinates[node_index][0] = node[1]["y"]
        coordinates[node_index][1] = node[1]["x"]

    return coordinates


def _find_node_index_central_to_network(node_index_map: dict) -> int:
    """Finds node index central to network."""
    coordinates = _get_coordinates(node_index_map)
    centroid = np.sum(coordinates, 0) / len(node_index_map)
    kd_tree = cKDTree(coordinates)
    return kd_tree.query(centroid)[1]


def generate_mapping_information(num_clients: int) -> tuple[nx.MultiDiGraph, int, list, list]:
    """Return ``nx.MultiDiGraph`` with client demand, depot id in graph, client ids in graph.

    Args:
        num_clients: Number of locations to be visited in total.

    Returns:
        map_network: ``nx.MultiDiGraph`` where nodes and edges represent locations and routes.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.
        map_bounds: List of lower and upper bound locations for map
    """
    # a private generator keeps the sample reproducible even when callbacks run concurrently
    rng = random.Random(num_clients)

    G = ox.graph_from_address(
        address=ADDRESS, dist=DISTANCE, network_type="drive", truncate_by_edge=True
    )
    map_network = ox.truncate.largest_component(G, strongly=True)

    node_index_map = dict(enumerate(map_network.nodes(data=True)))

    depot_id = node_index_map[_find_node_index_central_to_network(node_index_map)][0]

    graph_copy = map_network.copy()
    graph_copy.remove_node(depot_id)
    client_subset = rng.sample(list(graph_copy.nodes), num_clients)

    for node_id in client_subset:
        map_network.nodes[node_id]["demand"] = 0

        for i in range(len(RESOURCES)):
            map_network.nodes[node_id][f"resource_{i}"] = rng.choice([1, 2])
            map_network.nodes[node_id]["demand"] += map_network.nodes[node_id][f"resource_{i}"]

    # Get min and max coordinates to determine map bounds
    coordinates = _get_coordinates(node_index_map)
    map_bounds = [coordinates.min(0).tolist(), coordinates.max(0).tolist()]

    return map_network, depot_id, client_subset, map_bounds


def get_position(G: nx.Graph, node_id: int) -> tuple[float, float]:
    """Return the ``(latitude, longitude)`` of a node in the street network.

    Args:
        G: The street network graph.
        node_id: The ID of the node whose position is to be retrieved.

    Returns:
        A tuple containing the latitude and longitude of the node.
    """
    return G.nodes[node_id]["y"], G.nodes[node_id]["x"]


def get_location(G: nx.Graph, node_id: int) -> Location:
    """Return the position and resource demand of a client location.

    Args:
        G: The street network graph.
        node_id: The ID of the client node.

    Returns:
        A Location object containing the node ID, position, and resource demand.
    """
    demand = [G.nodes[node_id][f"resource_{i}"] * 100 for i in range(len(RESOURCES))]
    return Location(node_id, get_position(G, node_id), demand)


def get_client_locations(G: nx.Graph, depot_id: int, client_subset: list) -> list[Location]:
    """Return the positions and demands of all client locations (excluding the depot).

    Args:
        G: The street network graph.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.

    Returns:
        A list of Location objects for all client nodes, excluding the depot.
    """
    return [get_location(G, node_id) for node_id in client_subset if node_id != depot_id]


def _lines_to_geojson(lines: Iterable[Sequence[Sequence[float]]]) -> dict:
    """Build a GeoJSON ``FeatureCollection`` of ``LineString`` features.

    Args:
        lines: Sequences of ``(longitude, latitude)`` coordinates, one per line.

    Returns:
        A GeoJSON ``FeatureCollection`` dictionary representing the lines.
    """
    features = [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(lon, 6), round(lat, 6)] for lon, lat in line],
            },
        }
        for line in lines
    ]
    return {"type": "FeatureCollection", "features": features}


def _clients_in_visiting_order(visiting_order: list[int], depot_id: int) -> list[int]:
    """Return the client nodes of a vehicle's route in the order they are first visited.

    Args:
        visiting_order: The solver's path for the vehicle, starting and ending at the depot.
            A heuristic tour may pass through a client more than once.
        depot_id: Node ID of the depot, which is excluded from the stops.

    Returns:
        The client nodes in the order they are first visited, excluding the depot.
    """
    return list(dict.fromkeys(node for node in visiting_order if node != depot_id))


def get_solution_routes(routing_parameters, routing_solver) -> tuple[list[VehicleRoute], dict]:
    """Build the per-vehicle routes and cost summary from a solved routing problem.

    Args:
        routing_parameters: Routing problem parameters.
        routing_solver: Solver class containing the solution (if run).

    Returns:
        A tuple containing:

        - list[VehicleRoute]: The route, stops, and drawing information for each vehicle.
        - dict: Solution cost information keyed by vehicle ID.
    """
    G = routing_parameters.map_network
    visiting_orders = routing_solver.visiting_order
    cost = routing_solver.cost_between_nodes
    paths = routing_solver.paths_and_lengths

    # expand the palette if there are more vehicles than colors
    palette = PALETTE * (len(visiting_orders) // len(PALETTE) + 1)

    routes = []
    solution_cost_information = {}
    for index, visiting_order in visiting_orders.items():
        vehicle_id = index + 1
        icon_name, route_color = palette.pop()

        stops = []
        route_order = _clients_in_visiting_order(visiting_order, routing_parameters.depot_id)
        for stop_number, node in enumerate(route_order, start=1):
            stops.append(Stop(get_location(G, node), stop_number))

        cost_information = {"optimized_cost": 0, "serviced": len(stops)}
        for i in range(len(RESOURCES)):
            cost_information[f"resource_{i}"] = sum(stop.location.demand[i] for stop in stops)

        # Walk the tour leg by leg.
        lines = []
        for start, end in zip(visiting_order[:-1], visiting_order[1:]):
            start_position, end_position = get_position(G, start), get_position(G, end)
            cost_information["optimized_cost"] += cost(start_position, end_position, start, end)

            if routing_parameters.vehicle_type is VehicleType.TRUCKS:
                # follow the street network along the shortest path between stops
                shortest_path = paths[start][1][end]
                edges = ox.routing.route_to_gdf(G, shortest_path, weight="length")
                lines.extend(geometry.coords for geometry in edges.geometry)
            else:  # if vehicle_type is DELIVERY_DRONES, fly as the crow flies
                lines.append([start_position[::-1], end_position[::-1]])

        routes.append(
            VehicleRoute(vehicle_id, route_color, icon_name, stops, _lines_to_geojson(lines))
        )
        solution_cost_information[vehicle_id] = cost_information

    return routes, solution_cost_information
