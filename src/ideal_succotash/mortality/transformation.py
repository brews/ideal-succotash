"""
Logic for mortality transformation and regionalization.
"""

from muuttaa import TransformationStrategy
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute annual average for 'tas'.
    """
    return ds[["tas"]].groupby("time.year").mean("time")


def _make_30hbartlett_climtas(ds: xr.Dataset) -> xr.Dataset:
    """
    From annaual 'tas' compute 30-year half-Bartlett kernel average.

    Output variable is "climtas". This assumes input's "tas" has "year"
    time dim.
    """
    kernel_length = 30
    w = np.arange(kernel_length)
    weight = xr.DataArray(w / w.sum(), dims=["window"])
    da = ds["tas"].rolling(year=30).construct(year="window").dot(weight)
    # TODO: What to do for NaNs? What happened in carb analysis for climtas? Check 'gs://rhg-data/climate/aggregated/NASA/NEX-GDDP-BCSD-reformatted/California_2019_census_tracts_weighted_by_population/{scenario}/{model}/tas-bartlett30/tas-bartlett30_BCSD_CA-censustract2019_{model}_{scenario}_{version}_{year}.zarr'
    return da.to_dataset(name="climtas").astype("float32")


make_climtas = TransformationStrategy(
    preprocess=_make_annual_tas,
    postprocess=_make_30hbartlett_climtas,
)


def _make_tas_20yrmean_annual_histogram(ds: xr.Dataset) -> xr.Dataset:
    bins = np.arange(230, 341)  # Range we get histogram count for. NOTE: in Kelvin!
    tas_annual_histogram = (
        ds["tas"].groupby("time.year").map(histogram, bins=[bins], dim=["time"])
    )

    tas_histogram_20yr = (
        tas_annual_histogram.rolling(year=20, center=True).mean().to_dataset()
    )
    return tas_histogram_20yr.astype("float32")


make_tas_20yrmean_annual_histogram = TransformationStrategy(
    preprocess=_make_tas_20yrmean_annual_histogram,
    postprocess=_no_processing,
)
