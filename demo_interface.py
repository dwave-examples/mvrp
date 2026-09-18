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

"""This file stores the HTML layout for the app."""

from __future__ import annotations

import dash_leaflet as dl
import dash_mantine_components as dmc
from dash import dcc, html

from demo_configs import (
    COST_LABEL,
    DEPOT_LABEL,
    DESCRIPTION,
    LOCATIONS_LABEL,
    MAIN_HEADER,
    NUM_CLIENT_LOCATIONS,
    NUM_VEHICLES,
    RESOURCES,
    SHOW_COST_COMPARISON,
    SHOW_DQM,
    SOLVER_TIME,
    THUMBNAIL,
    UNITS_IMPERIAL,
)
from map import Location, VehicleRoute
from src.demo_enums import SolverType, VehicleType

THEME_COLOR = "#2d4376"

DEPOT_ICON = "static/depot_location.png"
LOCATION_ICON_DIR = "static/location_icons"
INITIAL_LOCATION_ICON = "location_orange"
MARKER_ICON_SIZE = [30, 48]  # icon files are 200 x 320 px
MARKER_ICON_ANCHOR = [15, 40]  # pin tip, relative to the top-left corner of the icon
MARKER_TOOLTIP_ANCHOR = [0, -36]

# OpenStreetMap standard tiles from the volunteer-run OSM tile servers, which require
# attribution and are subject to https://operations.osmfoundation.org/policies/tiles/
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
MAP_ATTRIBUTION = (
    '<a href="https://leafletjs.com" title="A JavaScript library for interactive maps">Leaflet</a>'
    ' | &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
ROUTE_STYLE = {"weight": 4, "opacity": 1}


def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dcc.Slider Dash docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbLabel=f"{label} slider",
                color=THEME_COLOR,
            ),
        ],
    )


def dropdown(label: str, id: str, options: list) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """
    # calculate drop-down options
    vehicle_options = [
        {"label": vehicle_type.label, "value": f"{vehicle_type.value}"}
        for vehicle_type in VehicleType
    ]

    solver_options = []
    for solver_type in SolverType:
        if solver_type is not SolverType.DQM or SHOW_DQM:
            solver_options.append({"label": solver_type.label, "value": f"{solver_type.value}"})

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "Vehicle Type",
                "vehicle-type-select",
                sorted(vehicle_options, key=lambda op: op["value"]),
            ),
            slider(
                "Vehicles to Deploy",
                "num-vehicles-select",
                NUM_VEHICLES,
            ),
            slider(
                LOCATIONS_LABEL,
                "num-clients-select",
                NUM_CLIENT_LOCATIONS,
            ),
            dropdown(
                "Solver",
                "solver-type-select",
                sorted(solver_options, key=lambda op: op["value"]),
            ),
            html.Label("Solver Time Limit (seconds)", htmlFor="solver-time-limit"),
            dmc.NumberInput(
                id="solver-time-limit",
                **SOLVER_TIME,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button(
                "Run Optimization", id="run-button", className="button", n_clicks=0, disabled=False
            ),
            html.Button(
                "Cancel Optimization",
                id="cancel-button",
                className="button",
                n_clicks=0,
                style={"display": "none"},
            ),
        ],
    )


def create_row_cells(values: list) -> list[html.Td]:
    """List required to execute loop, unpack after to maintain required structure."""
    return [html.Td(round(value, 3 if UNITS_IMPERIAL else 0)) for value in values]


def create_table(values_dicts: dict[int, dict], values_totals: list) -> html.Table:
    """Create a table dynamically.

    Args:
        values_dicts: Dictionary with vehicle id keys and results data as values.
        values_totals: List of total results data (sum of individual vehicle data).
    """

    headers = ["Vehicle ID", COST_LABEL, LOCATIONS_LABEL, *RESOURCES]

    table = html.Table(
        className="results result-table",
        children=[
            html.Thead([html.Tr([html.Th(header) for header in headers])]),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(vehicle),
                            *create_row_cells(
                                list(results.values())
                            ),  # Unpack list to maintain required structure
                        ]
                    )
                    for vehicle, results in values_dicts.items()
                ]
            ),
            html.Tfoot(
                [
                    html.Tr(
                        [
                            html.Td("Total"),
                            *create_row_cells(
                                values_totals
                            ),  # Unpack list to maintain required structure
                        ],
                        className="total-cost-row",
                    )
                ]
            ),
        ],
    )

    return table


def _marker_icon(icon_url: str) -> dict:
    """Leaflet icon options for a map marker pinned at the icon's tip.

    Args:
        icon_url: URL of the marker icon image.

    Returns:
        Leaflet icon options for the marker.
    """

    return {
        "iconUrl": icon_url,
        "iconSize": MARKER_ICON_SIZE,
        "iconAnchor": MARKER_ICON_ANCHOR,
        "tooltipAnchor": MARKER_TOOLTIP_ANCHOR,
    }


def generate_marker(
    position: tuple[float, float], icon_url: str, tooltip_lines: list[str], alt: str
) -> dl.Marker:
    """Generate a map marker with a multi-line hover tooltip.

    Args:
        position: ``(latitude, longitude)`` of the marker.
        icon_url: URL of the marker icon image.
        tooltip_lines: Lines of text shown when hovering over the marker.
        alt: Alternative text for the marker icon.

    Returns:
        The generated map marker with tooltip.
    """
    tooltip_children = []
    for line in tooltip_lines:
        tooltip_children.extend([line, html.Br()])

    return dl.Marker(
        position=position,
        icon=_marker_icon(icon_url),
        alt=alt,
        children=dl.Tooltip(tooltip_children[:-1], direction="top"),
    )


def _demand_lines(location: Location) -> list[str]:
    """Tooltip lines listing the demand for each resource at a location.

    Args:
        location: The location object containing demand information.

    Returns:
        Tooltip lines for the location's resource demands.
    """
    return [f"{resource}: {demand}" for resource, demand in zip(RESOURCES, location.demand)]


def generate_locations_layer(
    depot_position: tuple[float, float], locations: list[Location]
) -> list[dl.Marker]:
    """Generate the depot marker and a marker for every client location.

    Args:
        depot_position: ``(latitude, longitude)`` of the depot.
        locations: The client locations to mark.

    Returns:
        Markers for the ``locations-layer`` layer group.
    """
    markers = [generate_marker(depot_position, DEPOT_ICON, [DEPOT_LABEL], DEPOT_LABEL)]
    icon_url = f"{LOCATION_ICON_DIR}/{INITIAL_LOCATION_ICON}.png"
    markers.extend(
        generate_marker(location.position, icon_url, _demand_lines(location), LOCATIONS_LABEL)
        for location in locations
    )
    return markers


def generate_solution_layers(
    depot_position: tuple[float, float], routes: list[VehicleRoute]
) -> tuple[list[dl.Marker], list[dl.GeoJSON]]:
    """Generate the markers and route lines for a solved routing problem.

    Args:
        depot_position: ``(latitude, longitude)`` of the depot.
        routes: The route driven by each vehicle.

    Returns:
        A tuple containing:

        - list[dl.Marker]: Depot and stop markers, colored by vehicle, for ``locations-layer``.
        - list[dl.GeoJSON]: One route line per vehicle for ``routes-layer``.
    """
    markers = [generate_marker(depot_position, DEPOT_ICON, [DEPOT_LABEL], DEPOT_LABEL)]
    route_lines = []

    for route in routes:
        icon_url = f"{LOCATION_ICON_DIR}/{route.icon_name}.png"
        for stop in route.stops:
            tooltip_lines = _demand_lines(stop.location) + [
                f"Vehicle ID: {route.vehicle_id}",
                f"Stop: #{stop.stop_number} of {len(route.stops)}",
            ]
            markers.append(
                generate_marker(
                    stop.location.position,
                    icon_url,
                    tooltip_lines,
                    f"{LOCATIONS_LABEL}, vehicle {route.vehicle_id}",
                )
            )

        route_lines.append(dl.GeoJSON(data=route.path, style={**ROUTE_STYLE, "color": route.color}))

    return markers, route_lines


def generate_map() -> html.Div:
    """Generate the map with empty layer groups that callbacks fill in.

    The background is OpenStreetMap tiles (desaturated in ``demo.css``). The viewport is set
    by callback once the street network is loaded.
    """
    return html.Div(
        className="map-region",
        role="region",
        children=dl.MapContainer(
            id="solution-map",
            center=[0, 0],
            zoom=1,
            keyboard=True,
            attributionControl=False,  # replaced by the control below, which credits the data
            children=[
                dl.TileLayer(url=OSM_TILE_URL, maxZoom=19),
                dl.LayerGroup(id="routes-layer"),
                dl.LayerGroup(id="locations-layer"),
                dl.FullScreenControl(),
                dl.ScaleControl(position="bottomleft"),
                dl.AttributionControl(position="bottomright", prefix=MAP_ATTRIBUTION),
            ],
        ),
        **{"aria-label": "Map of locations"},
    )


def problem_details(index: int) -> html.Div:
    """Generate the problem details section.

    Args:
        index: Unique element id to differentiate matching elements.
            Must be different from left column collapse button.

    Returns:
        html.Div: Div containing a collapsable table.
    """
    return html.Div(
        id={"type": "to-collapse-class", "index": index},
        className="details-collapse-wrapper collapsed",
        children=[
            # Problem details collapsible button and header
            html.Button(
                id={"type": "collapse-trigger", "index": index},
                className="details-collapse",
                children=[
                    html.H5("Problem Details"),
                    html.Div(className="collapse-arrow"),
                ],
                **{"aria-expanded": "true"},
            ),
            html.Div(
                className="details-to-collapse",
                children=[
                    html.Table(
                        id="solution-stats-table",
                        className="problem-details-table",
                        children=[
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th(
                                                colSpan=2,
                                                children=["Problem Specifics"],
                                            ),
                                            html.Th(
                                                colSpan=2,
                                                children=["Wall Clock Time"],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            html.Tbody(
                                id="problem-details",
                                children=[
                                    html.Tr(
                                        [
                                            html.Td(LOCATIONS_LABEL),
                                            html.Td(id="num-locations"),
                                            html.Td("Quantum Hybrid"),
                                            html.Td(id="wall-clock-time-quantum"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Vehicles Deployed"),
                                            html.Td(id="vehicles-deployed"),
                                            html.Td("Classical"),
                                            html.Td(id="wall-clock-time-classical"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Problem Size"),
                                            html.Td(id="problem-size"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Search Space"),
                                            html.Td(id="search-space"),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_interface():
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
            ),
            # below are any temporary storage items, e.g., for sharing data between callbacks
            dcc.Store(id="stored-results"),  # temporarily stored results table
            dcc.Store(id="sampler-type"),  # solver type used for latest run
            dcc.Store(
                id="reset-results"
            ),  # whether to reset the results tables before displaying the latest run
            dcc.Store(
                id="run-in-progress", data=False
            ),  # callback blocker to signal that the run is complete
            dcc.Store(id="parameter-hash"),  # hash string to detect changed parameters
            dcc.Store(id="cost-comparison"),  # dictionary with solver keys and run values
            html.Main(
                className="columns-main",
                id="main-content",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value="map-tab",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab(
                                                                "Map",
                                                                value="map-tab",
                                                                id="map-tab",
                                                            ),
                                                            dmc.TabsTab(
                                                                "Results",
                                                                value="results-tab",
                                                                id="results-tab",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="map-tab",
                                        **{"aria-labelledby": "map-tab"},
                                        children=[
                                            dcc.Loading(
                                                id="loading",
                                                type="circle",
                                                color=THEME_COLOR,
                                                parent_className="map-wrapper",
                                                overlay_style={"visibility": "visible"},
                                                children=generate_map(),
                                            ),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="results-tab",
                                        **{"aria-labelledby": "results-tab"},
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                className="results-tables",
                                                                children=[
                                                                    html.Div(
                                                                        id="solution-cost-table-div",
                                                                        className="result-table-div",
                                                                        children=[
                                                                            html.H3(
                                                                                className="table-label",
                                                                                children=[
                                                                                    html.Span(
                                                                                        id="hybrid-table-label"
                                                                                    ),
                                                                                    " Results",
                                                                                ],
                                                                            ),
                                                                            html.Div(
                                                                                title="Quantum Hybrid",
                                                                                id="solution-cost-table",
                                                                                children=[],  # add children dynamically using 'create_table' below
                                                                            ),
                                                                        ],
                                                                    ),
                                                                    html.Div(
                                                                        id="solution-cost-table-classical-div",
                                                                        className="result-table-div",
                                                                        children=[
                                                                            html.H3(
                                                                                children=[
                                                                                    "Classical (K-Means) Results"
                                                                                ],
                                                                                className="table-label",
                                                                            ),
                                                                            html.Div(
                                                                                title="Classical (K-Means)",
                                                                                id="solution-cost-table-classical",
                                                                                children=[],  # add children dynamically using 'create_table' below
                                                                            ),
                                                                        ],
                                                                    ),
                                                                ],
                                                            ),
                                                            html.H4(
                                                                id="performance-improvement-quantum",
                                                                className=(
                                                                    ""
                                                                    if SHOW_COST_COMPARISON
                                                                    else "display-none"
                                                                ),
                                                            ),
                                                        ]
                                                    ),
                                                    # Problem details dropdown
                                                    html.Div([html.Hr(), problem_details(1)]),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
