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

import html

from dash import dcc, html

from app_configs import HTML_CONFIGS, NUM_CLIENTS, NUM_VEHICLES, SAMPLER_TYPES, VEHICLE_TYPES

map_width, map_height = 1000, 600


def description_card():
    """A Div containing dashboard title & descriptions."""
    return html.Div(
        id="description-card",
        children=[html.H2(HTML_CONFIGS["main_header"]), html.P(HTML_CONFIGS["description"])],
    )


def generate_control_card() -> html.Div:
    """
    This function generates the control card for the dashboard, which
    contains the dropdowns for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the dropdowns for selecting the scenario,
        model, and solver.
    """
    vehicle_options = [{"label": vehicle, "value": vehicle} for vehicle in VEHICLE_TYPES]
    sampler_options = [{"label": sampler, "value": sampler} for sampler in SAMPLER_TYPES]

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
                value=NUM_VEHICLES["min"] + 1,
                marks={
                    NUM_VEHICLES["min"]: str(NUM_VEHICLES["min"]),
                    NUM_VEHICLES["max"]: str(NUM_VEHICLES["max"]),
                },
                tooltip={
                    "placement": "top",
                    "always_visible": True,
                    # "style": {"color": "LightSteelBlue", "fontSize": "20px"}
                }
            ),
            html.H4("Number of force locations", className="control-p"),
            dcc.Slider(
                id="num-clients-select",
                **NUM_CLIENTS,
                value=NUM_CLIENTS["min"],
                marks={
                    NUM_CLIENTS["min"]: str(NUM_CLIENTS["min"]),
                    NUM_CLIENTS["max"]: str(NUM_CLIENTS["max"]),
                },
                tooltip={
                    "placement": "top",
                    "always_visible": True,
                }
            ),
            html.H4("Solver Time Limit", className="control-p"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                value=HTML_CONFIGS["solver_options"]["default_time_seconds"],
                min=HTML_CONFIGS["solver_options"]["min_time_seconds"],
                max=HTML_CONFIGS["solver_options"]["max_time_seconds"],
                step=HTML_CONFIGS["solver_options"]["time_step_seconds"],
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
                        style={"visibility": "hidden"},
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
            # Banner
            html.Div(id="banner", children=[html.Img(src="assets/dwave_logo.svg")]),
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
                                label=HTML_CONFIGS["tabs"]["map"]["name"],
                                id="map-tab",
                                value="map-tab",  # used for switching to programatically
                                className="tab",
                                children=[html.Iframe(id="solution-map")],
                            ),
                            dcc.Tab(
                                label=HTML_CONFIGS["tabs"]["result"]["name"],
                                id="results-tab",
                                className="tab",
                                disabled=True,
                                children=[
                                    html.H3("Solution stats"),
                                    html.Table(
                                        className="result-table",
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
                                    html.Table(
                                        className="result-table",
                                        id="solution-cost-table",
                                        children=[],  # add children dynamically using 'create_table_row' below
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )


def create_table_row(num_vehicles, values_dicts, values_tot):
    """TODO"""
    table_row = [
        html.Tr(
            [
                html.Th("Vehicle", className="cost-heading"),
                html.Th("Solution cost [m]", className="cost-heading"),
                html.Th("Forces serviced", className="cost-heading"),
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
