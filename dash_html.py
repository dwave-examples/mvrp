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

"""This file stores the HTML layout for the app (see ``mvrp.css`` for CSS styling)."""
from __future__ import annotations

import html

from dash import dcc, html

from app_configs import (DESCRIPTION, MAIN_HEADER, NUM_CLIENT_LOCATIONS, NUM_VEHICLES, SOLVER_TIME,
                         THUMBNAIL)

map_width, map_height = 1000, 600

VEHICLE_TYPES = ["Trucks", "Delivery Drones"]
SAMPLER_TYPES = ["Quantum Hybrid", "Classical (K-Means)"]


def description_card():
    """A Div containing dashboard title & descriptions."""
    return html.Div(
        id="description-card",
        children=[html.H2(MAIN_HEADER), html.P(DESCRIPTION)],
    )


def generate_control_card() -> html.Div:
    """
    This function generates the control card for the dashboard, which
    contains the dropdowns for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the dropdowns for selecting the scenario,
        model, and solver.
    """
    vehicle_options = [{"label": vehicle, "value": i} for i, vehicle in enumerate(VEHICLE_TYPES)]
    sampler_options = [{"label": sampler, "value": i} for i, sampler in enumerate(SAMPLER_TYPES)]

    return html.Div(
        id="control-card",
        style={"padding-top": "20px", "padding-right": "20px"},
        children=[
            html.H4("Vehicle Type", className="control-p"),
            dcc.Dropdown(
                id="vehicle-type-select",
                options=vehicle_options,
                value=vehicle_options[0]["value"],
                clearable=False,
                searchable=False,
            ),
            html.H4("Sampler", className="control-p"),
            dcc.Dropdown(
                id="sampler-type-select",
                options=sampler_options,
                value=sampler_options[0]["value"],
                clearable=False,
                searchable=False,
            ),
            html.H4("Number of vehicles to deploy", className="control-p"),
            dcc.Slider(
                id="num-vehicles-select",
                **NUM_VEHICLES,
                marks={
                    NUM_VEHICLES["min"]: str(NUM_VEHICLES["min"]),
                    NUM_VEHICLES["max"]: str(NUM_VEHICLES["max"]),
                },
                tooltip={
                    "placement": "top",
                    "always_visible": True,
                    # "style": {"color": "LightSteelBlue", "fontSize": "20px"}
                },
            ),
            html.H4("Number of force locations", className="control-p"),
            dcc.Slider(
                id="num-clients-select",
                **NUM_CLIENT_LOCATIONS,
                marks={
                    NUM_CLIENT_LOCATIONS["min"]: str(NUM_CLIENT_LOCATIONS["min"]),
                    NUM_CLIENT_LOCATIONS["max"]: str(NUM_CLIENT_LOCATIONS["max"]),
                },
                tooltip={
                    "placement": "top",
                    "always_visible": True,
                },
            ),
            html.H4("Solver Time Limit", className="control-p"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                **SOLVER_TIME,
            ),
            html.Div(
                id="button-group",
                children=[
                    html.Button(
                        id="run-button", children="Run Optimization", n_clicks=0, disabled=False
                    ),
                    html.Button(
                        id="cancel-button",
                        children="Cancel Optimization",
                        n_clicks=0,
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def set_html(app):
    """Set the application HTML."""
    app.layout = html.Div(
        id="app-container",
        children=[
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
            # Banner
            html.Div(id="banner", children=[html.Img(src=THUMBNAIL)]),
            # Left column
            html.Div(
                id="left-column",
                className="four-columns",
                children=[
                    description_card(),
                    generate_control_card(),
                    html.Div(["initial child"], id="output-clientside", style={"display": "none"}),
                ],
            ),
            # Right column
            html.Div(
                id="right-column",
                style={"padding": "10px"},
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="map-tab",
                        children=[
                            dcc.Tab(
                                label="Map",
                                id="map-tab",
                                value="map-tab",  # used for switching to programatically
                                className="tab",
                                children=[html.Iframe(id="solution-map")],
                            ),
                            dcc.Tab(
                                label="Results",
                                id="results-tab",
                                className="tab",
                                disabled=True,
                                children=[
                                    html.H3("Solution stats"),
                                    html.Table(
                                        id="solution-stats-table",
                                        children=[
                                            html.Tr(
                                                [
                                                    html.Th(
                                                        "Problem size", className="stats-heading"
                                                    ),
                                                    html.Th(
                                                        "Search space", className="stats-heading"
                                                    ),
                                                    html.Th(
                                                        "Wall clock time [s]",
                                                        className="stats-heading",
                                                    ),
                                                    html.Th(
                                                        "Force elements", className="stats-heading"
                                                    ),
                                                    html.Th(
                                                        "Vehicles deployed",
                                                        className="stats-heading",
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td(
                                                        id="problem-size", className="stats-row"
                                                    ),
                                                    html.Td(
                                                        id="search-space", className="stats-row"
                                                    ),
                                                    html.Td(
                                                        id="wall-clock-time", className="stats-row"
                                                    ),
                                                    html.Td(
                                                        id="force-elements", className="stats-row"
                                                    ),
                                                    html.Td(
                                                        id="vehicles-deployed",
                                                        className="stats-row",
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    html.H3("Solution cost"),
                                    html.Div(
                                        id="solution-cost-table-div",
                                        className="result-table-div",
                                        children=[
                                            html.H4("Quantum Hybrid"),
                                            html.Table(
                                                title="Quantum Hybrid",
                                                className="result-table",
                                                id="solution-cost-table",
                                                children=[],  # add children dynamically using 'create_table_row' below
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="solution-cost-table-classical-div",
                                        className="result-table-div",
                                        children=[
                                            html.H4("Classical (K-Means)"),
                                            html.Table(
                                                title="Classical (K-Means)",
                                                className="result-table",
                                                id="solution-cost-table-classical",
                                                children=[],  # add children dynamically using 'create_table_row' below
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )


def create_table_row(
    num_vehicles: int, values_dicts: dict[int, dict], values_tot: list
) -> list[html.Tr]:
    """Create a row in the table dynamically.

    Args:
        num_vehicles: Number of vehicles (i.e., number of table rows).
        values_dicts: List of dictionaries with vehicle number as results data as values.
        values_tot: List of total results data (sum of individual vehicle data).
    """
    table_row = [
        html.Tr(
            [
                html.Th("Vehicle", className="cost-heading"),
                html.Th("Cost [m]", className="cost-heading"),
                html.Th("Forces", className="cost-heading"),
                html.Th("Water", className="cost-heading"),
                html.Th("Food", className="cost-heading"),
                html.Th("Other", className="cost-heading"),
            ]
        )
    ]
    for i in range(num_vehicles):
        values = list(values_dicts[i].values())

        row = html.Tr(
            [
                html.Td(i + 1, className="cost-row"),
                html.Td(round(values[0]), className="cost-row"),
                html.Td(values[1], className="cost-row"),
                html.Td(values[2], className="cost-row"),
                html.Td(values[3], className="cost-row"),
                html.Td(values[4], className="cost-row"),
            ]
        )

        table_row.append(row)

    table_row.append(
        html.Tr(
            [
                html.Td("Total", className="total-cost-row"),
                html.Td(round(values_tot[0]), className="total-cost-row"),
                html.Td(values_tot[1], className="total-cost-row"),
                html.Td(values_tot[2], className="total-cost-row"),
                html.Td(values_tot[3], className="total-cost-row"),
                html.Td(values_tot[4], className="total-cost-row"),
            ]
        )
    )
    return table_row
