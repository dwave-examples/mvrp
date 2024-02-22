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

import json

HTML_CONFIGS = {
    "title": "MVRP Demo",
    "main_header": "Multi Vehicle Routing Problem",
    "description": """
        Run the Multi Vehicle Routing Problem (MVRP) problem for several different scenarios. Select
        between delivery drones (flight path) and trucks (roads), the number of vehicles and client
        locations.
        """,
    "solver_options": {
        "min_time_seconds": 5,
        "max_time_seconds": 300,
        "time_step_seconds": 5,
        "default_time_seconds": 5,
    },
    "tabs": {
        "map": {
            "name": "Map",
            "header": "Solution map",
        },
        "classical": {
            "name": "Classical",
            "header": "K-Means classical solution",
        },
        "result": {
            "name": "Results",
            "header": "D-Wave Hybrid Solver results",
        },
    },
}

VEHICLE_TYPES = ["Delivery Drones", "Trucks"]
SAMPLER_TYPES = ["Quantum Hybrid (DQM)", "Classical (K-Means)"]
NUM_VEHICLES = {"min": 1, "max": 10, "step": 1}
NUM_CLIENTS = {"min": 10, "max": 100, "step": 1}
