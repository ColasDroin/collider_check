"""A minimal set of unit tests to ensure no bug is raised when calling any (public) function of the
package. This is not meant to be an exhaustive test of the package (the behaviour of each function
is not tested under a large variety of inputs)."""

# ==================================================================================================
# --- Imports
# ==================================================================================================
# Imports from the standard library
import copy

import numpy as np
import pandas as pd
import pytest

# Third party imports
import xtrack as xt
import yaml

# Import the module to test
from collider_check import ColliderCheck

# ==================================================================================================
# --- Script to load (heavy) data
# ==================================================================================================

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

# ==================================================================================================
# --- Tests
# ==================================================================================================

# Do not use collider_check as a fixture since it's heavy to load
# @pytest.fixture
# def collider_check():
#     # Load collider as test data
#     path_collider = "test_data/collider.json"
#     collider = xt.Multiline.from_json(path_collider)

#     # Build collider_check object
#     collider.build_trackers()

#     return ColliderCheck(collider=collider)


def test_configuration():
    if collider_check.configuration is None:
        return

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
    l_n_col = []
    for IP in [1, 2, 8]:
        n_col = collider_check.return_number_of_collisions(IP=IP)
        # Test type returned
        assert isinstance(n_col, int)
        l_n_col.append(n_col)

    # Get the expected number of collisions from the filling scheme
    if hasattr(collider_check, "path_filling_scheme") and collider_check.path_filling_scheme:
        l_expected_number_of_collisions = [
            int(x) for x in collider_check.path_filling_scheme.split("/")[-1].split("_")[2:5]
        ]
        assert np.allclose(l_expected_number_of_collisions, l_n_col)


@pytest.mark.parametrize(
    "IP,lumi",
    [
        (
            IP,
            initial_configuration["config_collider"]["config_beambeam"][
                f"luminosity_ip{IP}_with_beam_beam"
            ],
        )
        for IP in [1, 2, 5, 8]
    ],
)
def test_return_luminosity(IP, lumi):
    # Test the type returned
    assert isinstance(lumi, float)

    # Test the value returned
    assert np.allclose(collider_check.return_luminosity(IP=IP), lumi)


@pytest.mark.parametrize("beam, IP", [(beam, IP) for beam in [1, 2] for IP in [1, 2, 5, 8]])
def test_return_twiss_at_ip(beam, IP):
    # Test the return_twiss_at_ip method, and check that it returns a pandas dataframe
    assert isinstance(collider_check.return_twiss_at_ip(beam=beam, ip=IP), pd.DataFrame)


@pytest.mark.parametrize("beam", [1, 2])
def test_return_tune_and_chromaticity(beam):
    # Test the return_tune_and_chromaticity method, and check that it returns a list of floats
    qx, dqx, qy, dqy = collider_check.return_tune_and_chromaticity(beam)
    assert isinstance(qx, float)
    assert isinstance(dqx, float)
    assert isinstance(qy, float)
    assert isinstance(dqy, float)

    # Can't compare the values with what it is in the config since they are not the same (before and
    # after setting beambeam)


def test_return_linear_coupling():
    # Test the return_linear_coupling method
    c_minus_b1, c_minus_b2 = collider_check.return_linear_coupling()

    # Check the value returned
    assert isinstance(c_minus_b1, float)
    assert isinstance(c_minus_b2, float)


def test_return_momentum_compaction_factor():
    # Test the return_momentum_compaction_factor method
    alpha_p_b1, alpha_p_b2 = collider_check.return_momentum_compaction_factor()

    # Check the value returned
    assert isinstance(alpha_p_b1, float)
    assert isinstance(alpha_p_b2, float)


def test_return_polarity_ip_2_8():
    # Test the return_polarity_ip_2_8 method
    pol_2, pol_8 = collider_check.return_polarity_ip_2_8()

    # Check the value returned
    assert pol_2 in [1, -1]
    assert pol_8 in [1, -1]


@pytest.mark.parametrize(
    "beam, IP", [(beam, IP) for beam in ["b1", "b2"] for IP in ["ip1", "ip2", "ip5", "ip8"]]
)
def test_compute_separation_variables(beam, IP):
    # Test the compute_separation_variables method
    assert isinstance(collider_check.compute_separation_variables(ip=IP, beam_weak=beam), dict)
    # ! A manual (visual) inspection is required in addition here


def test_return_dic_position_all_ips():
    # Test the return_dic_position_all_ips method
    assert isinstance(collider_check.return_dic_position_all_ips(), dict)


@pytest.mark.parametrize(
    "beam, ip", [(beam, ip) for beam in ["b1", "b2"] for ip in ["ip1", "ip2", "ip5", "ip8"]]
)
def test_plots(beam, ip):
    collider_check.plot_orbits(beam_weak=beam, ip=ip)
    collider_check.plot_separation(beam_weak=beam, ip=ip)


def test_output():
    # Test the output method
    assert isinstance(collider_check.output_check_as_str(), str)
