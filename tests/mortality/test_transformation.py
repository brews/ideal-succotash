from muuttaa import apply_transformations, SegmentWeights
import numpy as np
import pytest
import xarray as xr

from ideal_succotash.mortality.transformation import (
    _make_annual_tas,
    _make_30hbartlett_climtas,
    make_climtas,
    make_tas_20yrmean_annual_histogram,
)


@pytest.fixture
def basic_segment_weights():
    sw = SegmentWeights(
        weights=xr.Dataset(
            {
                "region": (["idx"], ["foobar"]),
                "weight": (["idx"], [1.0]),
                "lon": (["idx"], [1.0]),
                "lat": (["idx"], [0.0]),
            },
        )
    )
    return sw


def test__make_annual_tas():
    """
    Test that _make_annual_tas grabs "tas" variable from a Dataset and spits out
    a Dataset with time averaged in a new "year" dim.

    This covers the new "year" dim moving to the first dimension but I'm not sure that matters.
    """
    expected = xr.Dataset(
        {"tas": (["year", "lon", "lat"], [[[182.0]], [[365.0]]])},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "year": [2023, 2024],
        },
    )

    ds_in = xr.Dataset(
        {"tas": (["lon", "lat", "time"], np.arange(366).reshape((1, 1, 366)))},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range("2023-01-01", "2024-01-01", freq="1D"),
        },
    )

    actual = _make_annual_tas(ds_in)
    xr.testing.assert_allclose(actual, expected)


def test__make_30hbartlett_climtas():
    """
    Test _make_30hbartlett_climtas creates a 30 year half-Bartlett average
    returned as "climtas".
    """
    ex = np.empty((31, 1, 1), dtype=np.float32)
    ex[:] = np.nan
    ex[-2, ...] = 19.666666
    ex[-1, ...] = 20.666666

    expected = xr.Dataset(
        {"climtas": (["year", "lon", "lat"], ex.reshape(31, 1, 1))},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "year": np.arange(2000, 2031),
        },
    )

    ds_in = xr.Dataset(
        {"tas": (["year", "lon", "lat"], np.arange(31).reshape((31, 1, 1)))},
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "year": np.arange(2000, 2031),
        },
    )

    actual = _make_30hbartlett_climtas(ds_in)
    xr.testing.assert_allclose(actual, expected)


def test_make_climtas(basic_segment_weights):
    """
    Test that make_climtas transformation runs through apply_transformation using basic_segment_weights without error, spitting out smoothed annual "climtas" variables from input daily "tas".
    """
    ex = np.empty((1, 31), dtype=np.float32)
    ex[:] = np.nan
    ex[..., -2] = 7360.3335
    ex[..., -1] = 7725.3335
    expected = xr.Dataset(
        {"climtas": (["region", "year"], ex.reshape(1, 31))},
        coords={
            "region": ["foobar"],
            "year": np.arange(2000, 2031),
        },
    )

    ds_in = xr.Dataset(
        {
            "tas": (
                ["lon", "lat", "time"],
                np.arange(11315, dtype=np.float32).reshape(1, 1, 11315),
            )
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range(
                "2000-01-01", "2030-12-31", freq="1D", calendar="noleap"
            ),
        },
    )

    actual = apply_transformations(
        ds_in,
        strategies=[make_climtas],
        regionalize=basic_segment_weights,
    )
    xr.testing.assert_allclose(actual, expected)


def test_make_tas_20yrmean_annual_histogram(basic_segment_weights):
    """
    Test that make_tas_20yrmean_annual_histogram transformation runs through apply_transformation using basic_segment_weights without error.
    Does basic checks on output "histogram_tas" and "tas_bin" which are created from input daily "tas".
    """
    expected = xr.Dataset(
        {"histogram_tas": (["region", "year", "tas_bin"], np.zeros((1, 22, 110)))},
        coords={
            "region": np.array(["foobar"]),
            "year": np.arange(2000, 2022),
            "tas_bin": np.arange(230.5, 340.5),
        },
    )
    expected["histogram_tas"].loc[
        {"region": "foobar", "tas_bin": 274.5, "year": 2010}
    ] = 365.0
    expected["histogram_tas"].loc[
        {"region": "foobar", "tas_bin": 274.5, "year": 2011}
    ] = 365.0
    expected["histogram_tas"].loc[
        {"region": "foobar", "tas_bin": 274.5, "year": 2012}
    ] = 346.79998779

    ds_in = xr.Dataset(
        {
            "tas": (
                ["lon", "lat", "time"],
                np.ones((1, 1, 7666), dtype=np.float32) + 273.15,
            )
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range(
                "2000-01-01", "2021-01-01", freq="1D", calendar="noleap"
            ),
        },
    )

    actual = apply_transformations(
        ds_in,
        strategies=[make_tas_20yrmean_annual_histogram],
        regionalize=basic_segment_weights,
    )
    xr.testing.assert_allclose(actual, expected)


def test_transforms_run_together(basic_segment_weights):
    """
    Test that mortality's transformation can run together through apply_transformation using basic_segment_weights without error. Just checks expected output variables are present.
    """
    ds_in = xr.Dataset(
        {
            "tas": (
                ["lon", "lat", "time"],
                np.arange(11315, dtype=np.float32).reshape(1, 1, 11315),
            )
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range(
                "2000-01-01", "2030-12-31", freq="1D", calendar="noleap"
            ),
        },
    )

    transformed = apply_transformations(
        ds_in,
        strategies=[make_climtas, make_tas_20yrmean_annual_histogram],
        regionalize=basic_segment_weights,
    )

    assert "histogram_tas" in transformed
    assert "climtas" in transformed
