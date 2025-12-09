from muuttaa import project
import numpy as np
import pytest
import xarray as xr

from ideal_succotash.mortality.projection import (
    mortality_impact_model,
    uclip,
    mortality_impact_model_gamma,
)


@pytest.fixture
def beta():
    b = np.arange(4).reshape((2, 1, 2))
    out = xr.Dataset(
        {"beta": (["age_cohort", "region", "tas_bin"], b)},
        coords={
            "region": np.array(["foobar"]),
            "tas_bin": np.array([20.5, 21.5]),
            "age_cohort": np.array(["age1", "age2"]),
        },
    )
    return out


@pytest.fixture
def gamma():
    g_m = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    g_sampled = np.arange(2 * 2 * 4 * 3).reshape(2, 2, 4, 3)
    out = xr.Dataset(
        {
            "gamma_mean": (["age_cohort", "degree", "covarname"], g_m),
            "gamma_sampled": (
                ["sample", "age_cohort", "degree", "covarname"],
                g_sampled,
            ),
        },
        coords={
            "covarname": ["1", "climtas", "loggdppc"],
            "degree": np.arange(4) + 1,
            "age_cohort": np.array(["age1", "age2"]),
            "sample": [0, 1],
        },
    )
    return out


@pytest.fixture
def ageshare():
    x = np.array([[0.25], [0.5]])
    out = xr.Dataset(
        {"share": (["age_cohort", "region"], x)},
        coords={
            "region": np.array(["foobar"]),
            "age_cohort": np.array(["age1", "age2"]),
        },
    )
    return out


@pytest.fixture
def loggdppc():
    out = xr.Dataset(
        {"loggdppc": (["region"], np.array([10]))},
        coords={
            "region": np.array(["foobar"]),
        },
    )
    return out


@pytest.fixture
def histogram_tas():
    x = np.arange(4).reshape((1, 2, 2))
    x *= 10
    out = xr.Dataset(
        {"histogram_tas": (["region", "year", "tas_bin"], x)},
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "tas_bin": np.array([20.5, 21.5]),
        },
    )
    return out


@pytest.fixture
def climtas():
    x = np.arange(2).reshape(
        (
            1,
            2,
        )
    )
    x *= 10
    out = xr.Dataset(
        {"climtas": (["region", "year"], x)},
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
        },
    )
    return out


def test_mortality_impact_model(beta, ageshare, histogram_tas):
    """
    Test that mortality_impact_model runs through muuttaa.project with generally correct output.
    """
    expected = xr.Dataset(
        {
            "impact": (["region", "age_cohort"], np.array([[5.0, 50.0]])),
            "_effect": (
                ["region", "year", "age_cohort"],
                np.array([[[2.5, 15.0], [7.5, 65.0]]]),
            ),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "age_cohort": np.array(["age1", "age2"]),
        },
    )

    actual = project(
        histogram_tas,  # Transformed input climate data.
        model=mortality_impact_model,
        parameters=xr.merge([ageshare, beta]),
    )

    xr.testing.assert_allclose(actual, expected)


def test_mortality_impact_model_gamma_mean(
    gamma, ageshare, loggdppc, histogram_tas, climtas
):
    """
    Test that mortality_impact_model_gamma runs through muuttaa.project.
     Checks for generally correct output using mean gamma as input.
    """
    # Build up what we expect output to be.
    ex_i = np.array([[5.11400338e7, 2.22185568e8]])
    ex_e = np.array([[[1.13169512e7, 4.79985275e7], [6.24569850e7, 2.70184095e8]]])
    expected = xr.Dataset(
        {
            "impact": (["region", "age_cohort"], ex_i),
            "_effect": (["region", "year", "age_cohort"], ex_e),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "age_cohort": np.array(["age1", "age2"]),
        },
    )

    # Combine data for input
    transformed_input = xr.merge([histogram_tas, climtas])
    # Using rename_vars because model expects "gamma" variable, not "gamma_mean".
    params = xr.merge(
        [
            ageshare,
            loggdppc,
            gamma[["gamma_mean"]].rename_vars({"gamma_mean": "gamma"}),
        ],
    )

    actual = project(
        transformed_input, model=mortality_impact_model_gamma, parameters=params
    )

    xr.testing.assert_allclose(actual, expected)


def test_mortality_impact_model_gamma_sampled(
    gamma, ageshare, loggdppc, histogram_tas, climtas
):
    """
    Test that mortality_impact_model_gamma runs through muuttaa.project.
    Checks for generally correct output using sampled gamma as input.
    """
    # Build up what we expect output to be.
    ex_i = np.array([[[5.11400338e7, 2.22185568e8], [1.71045534e8, 4.61996568e8]]])
    ex_e = np.array(
        [
            [
                [[1.13169512e7, 4.79985275e7], [3.66815762e7, 9.87277775e7]],
                [[6.24569850e7, 2.70184095e8], [2.07727110e8, 5.60724345e8]],
            ]
        ]
    )
    expected = xr.Dataset(
        {
            "impact": (["region", "sample", "age_cohort"], ex_i),
            "_effect": (["region", "year", "sample", "age_cohort"], ex_e),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "age_cohort": np.array(["age1", "age2"]),
            "sample": [0, 1],
        },
    )

    # Combine data for input
    transformed_input = xr.merge([histogram_tas, climtas])
    # Using rename_vars because model expects "gamma" variable, not "gamma_mean".
    params = xr.merge(
        [
            ageshare,
            loggdppc,
            gamma[["gamma_sampled"]].rename_vars({"gamma_sampled": "gamma"}),
        ],
    )

    actual = project(
        transformed_input, model=mortality_impact_model_gamma, parameters=params
    )

    xr.testing.assert_allclose(actual, expected)


def test_uclip():
    """
    Test uclip clips input data with multiple regions. That is, with an
    additional dimension on the input data. One region has "u" shaped data,
    another "rising" data, and finally "falling" input data.
    """
    dim0_name = "x"
    region_coord = ["u", "rising", "falling"]
    dim0_coord = [5, 10, 20, 30, 40]
    da_in = xr.DataArray(
        [[3, 2, 3, 5, 5], [0, 2, 2, 5, 5], [5, 5, 2, 2, 0]],
        coords=[region_coord, dim0_coord],
        dims=["region", dim0_name],
        name="foobar",
    )
    expected = xr.DataArray(
        [
            [1.0, 0.0, 1.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 3.0, 3.0],
            [3.0, 3.0, 0.0, 0.0, 0.0],
        ],
        coords=[region_coord, dim0_coord],
        dims=["region", dim0_name],
        name="foobar",
    )

    actual = uclip(da_in, dim=dim0_name)

    xr.testing.assert_equal(actual, expected)
