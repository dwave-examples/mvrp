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

#######################################
# Sliders, buttons and option entries #
#######################################

# number of vehicles slider (value means default)
NUM_VEHICLES = {
    "min": 1,
    "max": 10,
    "step": 1,
    "value": 6,
}

# number of client locations slider (value means default)
NUM_CLIENT_LOCATIONS = {
    "min": 10,
    "max": 100,
    "step": 1,
    "value": 70,
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 5,
    "max": 300,
    "step": 5,
    "value": 5,
}
