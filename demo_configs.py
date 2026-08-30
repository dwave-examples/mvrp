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

# Shows/hides Quantum Hybrid vs Classical cost comparison in the
# results tab when both are run with the same settings.
SHOW_COST_COMPARISON = False

# Units will be in miles if true, meters if false
# If updated, make sure units match COST_LABEL below
UNITS_IMPERIAL = False
COST_LABEL = "Distance (m)"  # Either "Distance (m)" or specific distance cost description

ADDRESS = "1055 Canada Pl, Vancouver, BC V6C 0C3"
DISTANCE = 1700  # bounding box distance (in meters) around address
THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "MVRP Demo"
MAIN_HEADER = "Multi Vehicle Routing"
DESCRIPTION = """\
Multi Vehicle Routing consists of routing delivery vehicles to a set of locations, such
that each location receives its required resources and no vehicle exceeds its carrying capacity.
"""

DEPOT_LABEL = "Depot"  # Either "Depot" or specific start location
LOCATIONS_LABEL = "Locations"  # Either "Locations" or business specific location type
RESOURCES = ["Water Pallets", "Food Boxes", "Clothing Boxes"]  # Supports any number of resources

#######################################
# Sliders, buttons and option entries #
#######################################

# number of vehicles slider (value means default)
# Kept small because the DQM solver runs locally via dimod's ExactDQMSolver, whose
# runtime grows as num_vehicles ** num_clients. Max here paired with max clients below
# tops out at ~9.8M states (~6s solve time) which stays safely under the point where the
# solver runs out of memory (observed around ~15M+ states).
NUM_VEHICLES = {
    "min": 1,
    "max": 5,
    "step": 1,
    "value": 2,
}

# number of client locations slider (value means default)
# See NUM_VEHICLES above for why this is kept small.
NUM_CLIENT_LOCATIONS = {
    "min": 3,
    "max": 10,
    "step": 1,
    "value": 6,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}
