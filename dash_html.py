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
        children=[html.H1(MAIN_HEADER), html.P(DESCRIPTION)],
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
        children=[
            html.Label("Vehicle Type"),
            dcc.Dropdown(
                id="vehicle-type-select",
                options=vehicle_options,
                value=vehicle_options[0]["value"],
                clearable=False,
                searchable=False,
            ),
            html.Label("Vehicles to Deploy"),
            dcc.Slider(
                id="num-vehicles-select",
                className="select",
                **NUM_VEHICLES,
                marks={
                    NUM_VEHICLES["min"]: str(NUM_VEHICLES["min"]),
                    NUM_VEHICLES["max"]: str(NUM_VEHICLES["max"]),
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
            ),
            html.Label("Locations"),
            dcc.Slider(
                id="num-clients-select",
                className="select",
                **NUM_CLIENT_LOCATIONS,
                marks={
                    NUM_CLIENT_LOCATIONS["min"]: str(NUM_CLIENT_LOCATIONS["min"]),
                    NUM_CLIENT_LOCATIONS["max"]: str(NUM_CLIENT_LOCATIONS["max"]),
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
            ),
            html.Label("Solver"),
            dcc.Dropdown(
                id="sampler-type-select",
                options=sampler_options,
                value=sampler_options[0]["value"],
                clearable=False,
                searchable=False,
            ),
            html.Label("Solver Time Limit (seconds)"),
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
            html.Div(
                id="columns",
                children=[
                    # Left column
                    html.Div(
                        id="left-column",
                        children=[
                            html.Div([
                                html.Div([
                                    description_card(),
                                    generate_control_card(),
                                    html.Div(["initial child"], id="output-clientside", style={"display": "none"}),
                                ])
                            ]),
                            html.Div(
                                html.Button(id="left-column-collapse", children=[html.Div()]),
                            )
                        ],
                    ),
                    # Right column
                    html.Div(
                        id="right-column",
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
                                        children=[
                                            dcc.Loading(
                                                id="loading",
                                                type="circle",
                                                color="#17BEBB",
                                                children=html.Iframe(id="solution-map")
                                            ),],
                                    ),
                                    dcc.Tab(
                                        label="Results",
                                        id="results-tab",
                                        className="tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                className="tab-content--results",
                                                children=[
                                                    html.H3("Problem Details"),
                                                    html.Table(
                                                        id="solution-stats-table",
                                                        className="result-table",
                                                        children=[
                                                            html.Thead(
                                                                [
                                                                    html.Tr(
                                                                        [
                                                                            html.Th("Problem Size"),
                                                                            html.Th("Search Space"),
                                                                            html.Th("Locations"),
                                                                            html.Th("Vehicles Deployed"),
                                                                        ]
                                                                    )
                                                                ]
                                                            ),
                                                            html.Tbody(
                                                                [
                                                                    html.Tr(
                                                                        [
                                                                            html.Td(id="problem-size"),
                                                                            html.Td(id="search-space"),
                                                                            html.Td(id="force-elements"),
                                                                            html.Td(id="vehicles-deployed"),
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ],
                                                    ),
                                                    html.H3("Solution Cost"),
                                                    html.Div(
                                                        id="solution-cost-table-div",
                                                        className="result-table-div",
                                                        children=[
                                                            html.H4(
                                                                children=[
                                                                    "Quantum Hybrid ",
                                                                    html.Span(id="wall-clock-time-quantum")
                                                                ],
                                                                className="table-label"
                                                            ),
                                                            html.Table(
                                                                title="Quantum Hybrid",
                                                                className="result-table",
                                                                id="solution-cost-table",
                                                                children=[],  # add children dynamically using 'create_table' below
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        id="solution-cost-table-classical-div",
                                                        className="result-table-div",
                                                        children=[
                                                            html.H4(
                                                                children=[
                                                                    "Classical (K-Means) ",
                                                                    html.Span(id="wall-clock-time-classical")
                                                                ],
                                                                className="table-label"
                                                            ),
                                                            html.Table(
                                                                title="Classical (K-Means)",
                                                                className="result-table",
                                                                id="solution-cost-table-classical",
                                                                children=[],  # add children dynamically using 'create_table' below
                                                            ),
                                                        ],
                                                    ),
                                                ]
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ]
            )
        ],
    )

def create_row_cells(values: list) -> list[html.Td]:
    """List required to execute loop, unpack after to maintain required structure."""
    return [html.Td(round(value)) for iteration, value in enumerate(values)]

def create_table(
    values_dicts: dict[int, dict], values_tot: list
) -> list:
    """Create a row in the table dynamically.

    Args:
        values_dicts: List of dictionaries with vehicle number as results data as values.
        values_tot: List of total results data (sum of individual vehicle data).
    """

    table = [
        html.Thead(
            [
                html.Tr(
                    [
                        html.Th("Vehicle"),
                        html.Th("Cost (m)"),
                        html.Th("Forces"),
                        html.Th("Water"),
                        html.Th("Food"),
                        html.Th("Other"),
                    ]
                )
            ]
        ),
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(index + 1),
                        *create_row_cells(list(vehicle.values())), # Unpack list to maintain required structure
                    ]
                ) for index, vehicle in enumerate(values_dicts)
            ]
        ),
        html.Tfoot(
            [
                html.Tr(
                    [
                        html.Td("Total"),
                        *create_row_cells(values_tot), # Unpack list to maintain required structure
                    ],
                    className="total-cost-row"
                )
            ]
        )
    ]

    return table
