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

import random
from pathlib import Path

import folium
import folium.plugins as plugins
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import osmnx as ox
from scipy.spatial import cKDTree

from app_configs import ADDRESS, DEPOT_LABEL, DISTANCE, RESOURCES, VEHICLE_LABEL
from solver.solver import VehicleType

ox.settings.use_cache = True
ox.settings.overpass_rate_limit = False

depot_icon_path = Path(__file__).parent / "assets/depot_location.png"

depot_icon = folium.CustomIcon(str(depot_icon_path), icon_size=(30, 48))


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
    random.seed(num_clients)

    G = ox.graph_from_address(
        address=ADDRESS, dist=DISTANCE, network_type="drive", truncate_by_edge=True
    )
    map_network = ox.utils_graph.get_largest_component(G, strongly=True)

    node_index_map = dict(enumerate(map_network.nodes(data=True)))

    depot_id = node_index_map[_find_node_index_central_to_network(node_index_map)][0]

    graph_copy = map_network.copy()
    graph_copy.remove_node(depot_id)
    client_subset = random.sample(list(graph_copy.nodes), num_clients)

    for node_id in client_subset:
        map_network.nodes[node_id]["demand"] = 0

        for i in range(len(RESOURCES)):
            map_network.nodes[node_id][f"resource_{i}"] = random.choice([1, 2])
            map_network.nodes[node_id]["demand"] += map_network.nodes[node_id][f"resource_{i}"]

    # Get min and max coordinates to determine map bounds
    coordinates = _get_coordinates(node_index_map)
    map_bounds = [coordinates.min(0).tolist(), coordinates.max(0).tolist()]

    return map_network, depot_id, client_subset, map_bounds


def _get_node_info(G: nx.Graph, force_id: int, icon_name: str) -> tuple[folium.CustomIcon, list[int]]:
    """Get node demand values and icons for each client location."""
    icon_path = Path(__file__).parent / f"assets/location_icons/{icon_name}.png"
    location_icon = folium.CustomIcon(str(icon_path), icon_size=(30, 48))
    sources = (f"resource_{i}" for i in range(len(RESOURCES)))
    return location_icon, [G.nodes[force_id][s] * 100 for s in sources]


def show_locations_on_initial_map(
    G: nx.MultiDiGraph, depot_id: int, client_subset: list, map_bounds: list[list]
) -> folium.Map:
    """Prepare map to be rendered initially on app screen.

    Args:
        G: ``nx.MultiDiGraph`` to build map from.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.

    Returns:
        folium.Map: Map with depot, client locations and tooltip popups.
    """
    # create folium map on which to plot depots
    tiles = "cartodb positron" # foilum map theme

    folium_map = ox.graph_to_gdfs(G, nodes=False, node_geometry=False).explore(
        style_kwds={"opacity": 0}, tiles=tiles  # Change opacity to 1 to see graph edges/roads in blue
    )

    folium_map.fit_bounds(map_bounds)

    # add marker to the depot location
    folium.Marker(
        location=(G.nodes[depot_id]["y"], G.nodes[depot_id]["x"]),
        tooltip=folium.map.Tooltip(text=DEPOT_LABEL, style="font-size: 1.4rem;"),
        icon=depot_icon,
    ).add_to(folium_map)

    # add markers to all the client locations
    for force_id in client_subset:
        if force_id == depot_id:
            continue

        location_icon, nodes = _get_node_info(G, force_id, "location_orange")

        folium.Marker(
            location=(G.nodes[force_id]["y"], G.nodes[force_id]["x"]),
            tooltip=folium.map.Tooltip(
                text=" <br> ".join(
                    [f"{resource}: {nodes[index]}" for index, resource in enumerate(RESOURCES)]
                ),
                style="font-size: 1.4rem;",
            ),
            icon=location_icon,
        ).add_to(folium_map)

    # add fullscreen button to map
    plugins.Fullscreen().add_to(folium_map)
    return folium_map


def plot_solution_routes_on_map(
    folium_map: folium.Map,
    routing_parameters,
    routing_solver,
) -> folium.folium.Map:
    """Generate interactive folium map for drone routes given solution dictionary.

    Args:
        folium_map: Initial folium map to plot solution on.
        routing_parameters: Routing problem parameters.
        routing_solver: Solver class containing the solution (if run).

    Returns:
        `folium.folium.Map` object,  dictionary with solution cost information.

    """
    solution_cost_information = {}
    G = routing_parameters.map_network

    solution = routing_solver.solution
    cost = routing_solver.cost_between_nodes
    paths = routing_solver.paths_and_lengths

    # get colourblind palette from seaborn (10 colours) and expand if more vehicles
    palette = [
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
    ] * (len(solution) // 10 + 1)

    locations = {}
    for index, route_network in solution.items():
        vehicle_id = index + 1
        icon_name, route_color = palette.pop()

        solution_cost_information[vehicle_id] = {
            "optimized_cost": 0,
            "forces_serviced": len(route_network.nodes) - 1,
        }
        for i in range(len(RESOURCES)):
            solution_cost_information[vehicle_id][f"resource_{i}"] = 0

        for stop_number, node in enumerate(route_network.nodes):
            locations.update({node: (G.nodes[node]["y"], G.nodes[node]["x"])})

            if node != routing_parameters.depot_id:
                location_icon, nodes = _get_node_info(G, node, icon_name)

                folium.Marker(
                    locations[node],
                    tooltip=folium.map.Tooltip(
                        text=" <br> ".join(
                                [f"{resource}: {G.nodes[node][f'resource_{i}']*100}" for i, resource in enumerate(RESOURCES)]
                            ) + f" <br> {VEHICLE_LABEL} ID: {vehicle_id} <br> Stop: #{stop_number} of {len(route_network.nodes)-1}",
                        style="font-size: 1.4rem;",
                    ),
                    icon=location_icon,
                ).add_to(folium_map)

                for i in range(len(RESOURCES)):
                    solution_cost_information[vehicle_id][f"resource_{i}"] += nodes[i]

        for start, end in route_network.edges:
            solution_cost_information[vehicle_id]["optimized_cost"] += cost(
                locations[start], locations[end], start, end
            )

            if routing_parameters.vehicle_type is VehicleType.TRUCKS:
                route = paths[start][1][end]
                folium_map = ox.graph_to_gdfs(G.subgraph(route), nodes=False).explore(
                    m=folium_map, color=route_color, style_kwds={"weight": 4}
                )
            else:  # if vehicle_type is DELIVERY_DRONES
                folium.PolyLine((locations[start], locations[end]), color=route_color).add_to(
                    folium_map
                )

    return folium_map, solution_cost_information
