# Imports from the standard library
import pytest
import numpy as np
import sys
import yaml
import copy

# Third party imports
import xtrack as xt

# Import the module to test
from src.collider_check import ColliderCheck

# Do not use collider_check as a fixture since it's heavy to load
# @pytest.fixture
# def collider_check():
#     # Load collider as test data
#     path_collider = "test_data/collider.json"
#     collider = xt.Multiline.from_json(path_collider)

#     # Build collider_check object
#     collider.build_trackers()

#     return ColliderCheck(collider=collider)


# Load collider as test data
path_collider = "test_data/collider.json"
collider = xt.Multiline.from_json(path_collider)

# Build collider_check object
collider.build_trackers()
collider_check = ColliderCheck(collider=collider)

# # Load independently configuration (normally identical, except for casting and path to filling scheme)
path_data = "test_data/config.yaml"
with open(path_data, "r") as stream:
    initial_configuration = yaml.safe_load(stream)


def test_configuration():
    # Save the intial configuration
    configuration = copy.deepcopy(collider_check.configuration)

    # Test that the configuration is either None or a dictionnary
    collider_check.configuration = configuration
    assert collider_check.configuration is None or isinstance(collider_check.configuration, dict)

    # Test the configuration getter and setter
    collider_check.configuration = initial_configuration
    assert collider_check.configuration == initial_configuration

    # Test that the configuration is either None or a dictionnary
    collider_check.configuration = configuration
    assert collider_check.configuration is None or isinstance(collider_check.configuration, dict)


def test_nemitt_x_y():
    # Test type returned
    assert isinstance(collider_check.nemitt_x, float)
    assert isinstance(collider_check.nemitt_y, float)

    print(initial_configuration["config_collider"])

    # Test values
    assert np.allclose(
        collider_check.nemitt_x,
        initial_configuration["config_collider"]["config_beambeam"]["nemitt_x"],
    )
    assert np.allclose(
        collider_check.nemitt_y,
        initial_configuration["config_collider"]["config_beambeam"]["nemitt_y"],
    )


def test_return_number_of_collisions():
    # Test type returned
    assert isinstance(collider_check.return_number_of_collisions(), int)
    # Get the expected number of collisions from the filling scheme
    l_expected_number_of_collisions = collider_check.configuration.path_filling_schemes.split("/")[
        -1
    ].split("_")[2:5]
    assert l_expected_number_of_collisions[0] == collider_check.return_number_of_collisions(IP=1)
    assert l_expected_number_of_collisions[1] == collider_check.return_number_of_collisions(IP=2)
    assert l_expected_number_of_collisions[2] == collider_check.return_number_of_collisions(IP=8)


# def test_return_luminosity(collider_check):
#     # Test the type returned
#     assert isinstance(collider_check.return_luminosity(), float)


# def test_return_twiss_at_ip(collider_check):
#     # Test the return_twiss_at_ip method
#     assert isinstance(collider_check.return_twiss_at_ip(), dict)


# def test_return_tune_and_chromaticity(collider_check):
#     # Test the return_tune_and_chromaticity method
#     assert isinstance(collider_check.return_tune_and_chromaticity(), dict)


# def test_return_linear_coupling(collider_check):
#     # Test the return_linear_coupling method
#     assert isinstance(collider_check.return_linear_coupling(), float)


# def test_return_momentum_compaction_factor(collider_check):
#     # Test the return_momentum_compaction_factor method
#     assert isinstance(collider_check.return_momentum_compaction_factor(), float)


# def test_return_polarity_ip_2_8(collider_check):
#     # Test the return_polarity_ip_2_8 method
#     assert isinstance(collider_check.return_polarity_ip_2_8(), dict)


# def test_compute_separation_variables(collider_check):
#     # Test the compute_separation_variables method
#     assert isinstance(collider_check.compute_separation_variables(), dict)


# def test_return_dic_position_all_ips(collider_check):
#     # Test the return_dic_position_all_ips method
#     assert isinstance(collider_check.return_dic_position_all_ips(), dict)
