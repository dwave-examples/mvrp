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

"""This file stores input parameters for the app."""

# Sets Dash debug which hides and shows Dash debug menu.
# Set to True if developing and False if demoing.
# App should be restarted to see change.
DEBUG = False

# Shows/hides Quantum Hybrid vs Classical cost comparison in the
# results tab when both are run with the same settings.
SHOW_COST_COMPARISON = False

# THEME_COLOR is used for the button, text, and banner and should be dark
# and pass accessibility checks with white: https://webaim.org/resources/contrastchecker/
# THEME_COLOR_SECONDARY can be light or dark and is used for sliders, loading icon, and tabs
THEME_COLOR = "#074C91"  # D-Wave dark blue default #074C91
THEME_COLOR_SECONDARY = "#2A7DE1"  # D-Wave blue default #2A7DE1

ADDRESS = "Cambridge Ln, Rockhampton QLD 4700, Australia"
DISTANCE = 1700  # bounding box distance (in meters) around address
THUMBNAIL = "assets/dwave_logo.svg"

APP_TITLE = "MVRP Demo"
MAIN_HEADER = "Multi Vehicle Routing Problem"
DESCRIPTION = """\
Run the Multi Vehicle Routing Problem (MVRP) problem for several different scenarios. Select
between delivery drones (flight path) and trucks (roads), the number of vehicles and client
locations.
"""

LOCATIONS_LABEL = "Locations"  # Either "Locations" or business specific location type

SHOW_DQM = False  # Show/hide DQM drop down option

#######################################
# Sliders, buttons and option entries #
#######################################

# number of vehicles slider (value means default)
NUM_VEHICLES = {
    "min": 1,
    "max": 10,
    "step": 1,
    "value": 4,
}

# number of client locations slider (value means default)
NUM_CLIENT_LOCATIONS = {
    "min": 10,
    "max": 100,
    "step": 1,
    "value": 60,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}
