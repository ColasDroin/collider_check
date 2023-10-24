import pytest
import numpy as np
import sys

print(sys.path)
from src.collider_check import ColliderCheck
import xtrack as xt
import copy

# Load collider as test data
path_collider = "test_data/collider.json"
collider = xt.Multiline.from_json(path_collider)
collider.build_trackers()
collider_check = ColliderCheck(collider=collider)
configuration = copy.deepcopy(collider_check.configuration)
print(configuration)

# def test_configuration(collider_check):
#     # Test the configuration getter and setter
#     config_dict = {"key": "value"}
#     collider_check.configuration = config_dict
#     assert collider_check.configuration == config_dict

#     # Test that the configuration is either None or a dictionnary
#     collider_check.configuration = configuration
#     assert collider_check.configuration is None or isinstance(collider_check.configuration, dict)


# def test_nemitt_x(collider_check):
#     # Test type returned
#     assert isinstance(collider_check.nemitt_x, float)


# def test_nemitt_y(collider_check):
#     # Test type returned
#     assert isinstance(collider_check.nemitt_y, float)


# def test_return_number_of_collisions(collider_check):
#     # Test type returned
#     assert isinstance(collider_check.return_number_of_collisions(), int)
#     # Get the expected number of collisions from the filling scheme
#     l_expected_number_of_collisions = configuration.path_filling_schemes.split("/")[-1].split("_")[
#         2:5
#     ]
#     assert l_expected_number_of_collisions[0] == collider_check.return_number_of_collisions(IP=1)
#     assert l_expected_number_of_collisions[1] == collider_check.return_number_of_collisions(IP=2)
#     assert l_expected_number_of_collisions[2] == collider_check.return_number_of_collisions(IP=8)


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
