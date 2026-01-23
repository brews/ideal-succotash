"""
Integration tests showing that we can create betas from gammas in a way comparable to beta created by the "yellow purple" diagnostic scripts.

The expected results were created using https://gitlab.com/ClimateImpactLab/Impacts/post-projection-tools/-/blob/90be5cc240c80cb0de8cc663d4cb6b8a887f2ad9/response_function/yellow_purple_package.R on 2026-01-23.
"""

import numpy as np
import pytest
import xarray as xr

from ideal_succotash.mortality.projection import (
    mortality_fulladapt_impact_model,
    mortality_incadapt_impact_model,
    mortality_noadapt_impact_model,
)


@pytest.fixture(scope="function")
def input_dataset():
    """
    Input xr.Dataset to test the calculation of beta from gamma under different
    adaptation scenarios in a mortality climate impact projection.

    This data is for region "USA.14.608" in the replication data for
    Carleton et al. 2022 (https://doi.org/10.1093/qje/qjac020). This is in the
    data.zip file in https://doi.org/10.5281/zenodo.6416119. Unzipped, data is in file
    'data/2_projection/3_impacts/main_specification/raw/single/rcp85/CCSM4/low/SSP3/mortality-allpreds.csv'
    in columns where 'model' is "Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-oldest".
    This data was processed for use here on 2025-12-18.

    This fixture is explicitly function scoped because some test cases might modify this data in place.
    Any other scope might cause tests to contaminate each other.
    """
    out = xr.Dataset(
        {
            "climtas": (["year"], np.array([10.69912, 14.43568858])),
            "loggdppc": (["year"], np.array([10.83039802, 11.58756156])),
            "gamma": (
                ["degree", "covarname", "year"],
                np.array(
                    [
                        [
                            [6.39902756e00, 6.39902756e00],
                            [4.36967574e-02, 4.36967574e-02],
                            [-6.75184274e-01, -6.75184274e-01],
                        ],
                        [
                            [-3.22143419e-01, -3.22143419e-01],
                            [1.37269824e-03, 1.37269824e-03],
                            [2.95628065e-02, 2.95628065e-02],
                        ],
                        [
                            [-4.42993455e-03, -4.42993455e-03],
                            [-1.06788430e-04, -1.06788430e-04],
                            [5.08517405e-04, 5.08517405e-04],
                        ],
                        [
                            [2.88863191e-04, 2.88863191e-04],
                            [9.32783836e-07, 9.32783836e-07],
                            [-2.73410162e-05, -2.73410162e-05],
                        ],
                    ]
                ),
            ),
        },
        coords={
            "year": np.array([2015, 2090]),
            "degree": np.array([1, 2, 3, 4]),
            "covarname": np.array(["1", "climtas", "loggdppc"], dtype="<U8"),
            "tas_bin": np.linspace(
                -40, 40, 9, dtype="float32"
            ),  # Temperature range over which the dose-response function is evaluated. This was not in the Carleton 2022 data.
        },
    )
    return out


def test_fulladapt_beta(input_dataset):
    """
    Check that we can replicate 'yellow-purple' beta creation for a 'full adaptation' mortality projection.
    """
    actual = mortality_fulladapt_impact_model.preprocess(input_dataset)["beta"]

    expected = xr.DataArray(
        np.array(
            [
                [
                    53.2544362,
                    32.70857888,
                    18.8792945,
                    9.73895633,
                    3.91491418,
                    0.68949442,
                    0.0,
                    2.43871044,
                    9.2528818,
                ],
                [
                    66.83552875,
                    43.3406702,
                    18.73261398,
                    6.37031311,
                    1.23223575,
                    0.0,
                    0.0,
                    1.25937671,
                    1.25937671,
                ],
            ]
        ),
        coords=[np.array([2015, 2090]), np.linspace(-40, 40, 9, dtype="float32")],
        dims=["year", "tas_bin"],
        name="beta",
    )

    xr.testing.assert_allclose(actual, expected, atol=1e-8)


def test_incadapt_beta(input_dataset):
    """
    Check that we can replicate 'yellow-purple' beta creation for an 'income-only adaptation' mortality projection.
    """
    actual = mortality_incadapt_impact_model.preprocess(input_dataset)["beta"]

    expected = xr.DataArray(
        np.array(
            [
                [
                    53.2544362,
                    32.70857888,
                    18.8792945,
                    9.73895633,
                    3.91491418,
                    0.68949442,
                    0.0,
                    2.43871044,
                    9.2528818,
                ],
                [
                    33.3824762,
                    32.70857888,
                    18.8792945,
                    9.73895633,
                    3.91491418,
                    0.0,
                    0.0,
                    2.37793343,
                    2.37793343,
                ],
            ]
        ),
        coords=[np.array([2015, 2090]), np.linspace(-40, 40, 9, dtype="float32")],
        dims=["year", "tas_bin"],
        name="beta",
    )

    xr.testing.assert_allclose(actual, expected, atol=1e-8)


def test_noadapt_beta(input_dataset):
    """
    Check that we can replicate 'yellow-purple' beta creation for an 'no adaptation' mortality projection.
    """
    actual = mortality_noadapt_impact_model.preprocess(input_dataset)["beta"]

    expected = xr.DataArray(
        np.array(
            [
                [
                    53.2544362,
                    32.70857888,
                    18.8792945,
                    9.73895633,
                    3.91491418,
                    0.68949442,
                    0.0,
                    2.43871044,
                    9.2528818,
                ],
                [
                    53.2544362,
                    32.70857888,
                    18.8792945,
                    9.73895633,
                    3.91491418,
                    0.68949442,
                    0.0,
                    2.43871044,
                    9.2528818,
                ],
            ]
        ),
        coords=[np.array([2015, 2090]), np.linspace(-40, 40, 9, dtype="float32")],
        dims=["year", "tas_bin"],
        name="beta",
    )

    xr.testing.assert_allclose(actual, expected, atol=1e-8)
