# Copyright 2025 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import dimod
import numpy as np
import pytest

from dwave.system import LeapHybridDQMSampler, LeapHybridNLSampler

from map import generate_mapping_information
from solver.cvrp import CapacitatedVehicleRoutingProblem
from solver.solver import RoutingProblemParameters, SamplerType, VehicleType


# parameters used for generating a known feasible solution
# for NL solver tests; used as default values for fixtures

NUM_VEHICLES, NUM_LOCATIONS, DEFAULT_COST = 2, 10, 42
FEASIBLE_NL_SOLUTION = [[0, 1, 2, 3, 5], [4, 6, 7, 8, 9]]

# EDGES = [
#     (60189396, 309827903),
#     (309827903, 60189310),
#     (60189310, 297411050),
#     (297411050, 309641293),
#     (309641293, 59844443),
#     (59844443, 60189396)
# ]
# PATHS = [60189396, 309827903, 60189310, 297411050, 309641293, 59844443, 60189396]


@pytest.fixture(
        scope="function",
        params=[
            (SamplerType.NL, VehicleType.TRUCKS),
            (SamplerType.NL, VehicleType.DELIVERY_DRONES),
            (SamplerType.KMEANS, VehicleType.TRUCKS),
            (SamplerType.KMEANS, VehicleType.DELIVERY_DRONES),
            (SamplerType.DQM, VehicleType.TRUCKS),
            (SamplerType.DQM, VehicleType.DELIVERY_DRONES),
        ]
    )
def parameters_with_combos(
    request,
    time_limit: int = 5
) -> RoutingProblemParameters:
    """Parametrized parameter fixture with all samplers and vehicle type combinations."""
    return _parameters(sampler_type=request.param[0], vehicle_type=request.param[1])


@pytest.fixture(scope="function")
def parameters() -> RoutingProblemParameters:
    """Default parameter fixture."""
    return _parameters()


@pytest.fixture(scope="function")
def parameters_trucks() -> RoutingProblemParameters:
    """Parameter fixture with trucks as vehicle type."""
    return _parameters(vehicle_type=VehicleType.TRUCKS)


@pytest.fixture(scope="function")
def parameters_drones() -> RoutingProblemParameters:
    """Parameter fixture with drones as vehicle type."""
    return _parameters(vehicle_type=VehicleType.DELIVERY_DRONES)


def _parameters(
    vehicle_type: VehicleType = VehicleType.TRUCKS,
    sampler_type: SamplerType = SamplerType.NL,
    time_limit: int = 5
) -> RoutingProblemParameters:
    """Helper function for parameter fixures above."""
    map_network, depot_id, client_subset, _ = generate_mapping_information(NUM_LOCATIONS)

    routing_problem_parameters = RoutingProblemParameters(
        map_network=map_network,
        depot_id=depot_id,
        client_subset=client_subset,
        num_clients=NUM_LOCATIONS,
        num_vehicles=NUM_VEHICLES,
        vehicle_type=vehicle_type,
        sampler_type=sampler_type,
        time_limit=time_limit,
    )

    return routing_problem_parameters


@pytest.fixture
def mock_sample_dqm(monkeypatch):
    """Mock ``LeapHybridDQMSampler.sample_dqm()`` for all tests."""

    def sample_dqm(self, dqm, *args, **kwargs):
        num_samples = 12    # min num of samples from dqm solver
        samples = np.empty((num_samples, dqm.num_variables()), dtype=int)

        for vi, v in enumerate(dqm.variables):
            samples[:, vi] = np.random.choice(dqm.num_cases(v), size=num_samples)

        return dimod.SampleSet.from_samples(
            samples_like=(samples, dqm.variables),
            vartype='DISCRETE',
            energy=0
        )

    monkeypatch.setattr(LeapHybridDQMSampler, "sample_dqm", sample_dqm)


@pytest.fixture
def mock_sample_nl(monkeypatch, parameters):
    """Mock ``LeapHybridNLSampler.sample()`` for all tests."""

    def sample(self, model, *args, **kwargs):
        model.states.resize(1)

        for decision in model.iter_decisions():
            decision.set_state(0, FEASIBLE_NL_SOLUTION)

    monkeypatch.setattr(LeapHybridNLSampler, "sample", sample)


@pytest.fixture
def cvrp():
    return CapacitatedVehicleRoutingProblem(lambda co_1, co_2, label, key: DEFAULT_COST)
