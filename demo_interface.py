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

import dash_mantine_components as dmc
from dash import dcc, html

from demo_configs import (
    COST_LABEL,
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
from src.demo_enums import SolverType, VehicleType

map_width, map_height = 1000, 600
THEME_COLOR = "#2d4376"


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
                tabIndex=1,
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
                                                            dmc.TabsTab("Map", value="map-tab"),
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
                                        tabIndex="12",
                                        children=[
                                            dcc.Loading(
                                                id="loading",
                                                type="circle",
                                                color=THEME_COLOR,
                                                parent_className="map-wrapper",
                                                overlay_style={"visibility": "visible"},
                                                children=html.Iframe(
                                                    id="solution-map", title="Map of locations"
                                                ),
                                            ),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="results-tab",
                                        tabIndex="13",
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
