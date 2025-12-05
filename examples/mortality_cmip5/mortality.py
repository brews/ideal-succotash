%pip install muuttaa==0.1.1

import datetime
import os
import uuid

from dask_gateway import GatewayCluster
import muuttaa
import numpy as np
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()

print(
    f"""
        {JUPYTER_IMAGE=}
        {START_TIME=}
        {UID=}
    """
)

CMIP_URI = "gs://impactlab-data-scratch/brews/3e2f02fa-f607-4d14-ae53-12be6b3f1e10/clean_cmip5.zarr"
SOCIOECONOMICS_URI = "gs://impactlab-data/gcp/integration_replication_sync/inputs/econ/raw/integration-econ-bc39.zarr"
SEGMENT_WEIGHTS_URI = "gs://impactlab-data-scratch/brews/clean_segment_weights.zarr"
GAMMA_URI = "gs://impactlab-data-scratch/brews/clean_gamma.zarr"


# TRANSFORMATION


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


make_climtas = muuttaa.TransformationStrategy(
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


make_tas_20yrmean_annual_histogram = muuttaa.TransformationStrategy(
    preprocess=_make_tas_20yrmean_annual_histogram,
    postprocess=_no_processing,
)

# DEBUG
make_tas = muuttaa.TransformationStrategy(
    preprocess=lambda x: x,
    postprocess=lambda x: x,
)


# IMPACT PROJECTION

def _mortality_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins * census tract age-spec populations
    _effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin") * ds["share"]

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)

    return xr.Dataset({"impact": impact, "_effect": _effect})


# If you already have beta.
mortality_impact_model = muuttaa.Projector(
    preprocess=_no_processing,
    project=_mortality_impact_model,
    postprocess=_no_processing,
)

def uclip(da, dim, lmmt=10, ummt=30):
    """Performs U Clipping of an unclipped response function for all regions
    simultaneously, centered around each region's Minimum Mortality Temperature (MMT).

    Parameters
    ----------
    da : DataArray
        xarray.DataArray of unclipped response functions
    dim : str
        Dimension name along which clipping will be applied. Clipping will be applied
        along dimensions `dim` for all other dimensions independently (e.g. a unique
        minimum point will be found for each combination of any other dimensions in the
        data).
    lmmt : int
        Lower bound of the temperature range in which a minimum mortality temperature
        will be searched for. Default is 10.
    ummt : int
        Upper bound of the temperature range in which a minimum mortality temperature
        will be searched for. Default is 30.

    Returns
    -------
    clipped : xarray DataArray of the regions response functioned centered on its mmt
    """
    # identify mmt within defined range
    range_msk = (da[dim] >= lmmt) & (da[dim] <= ummt)
    min_idx = da.where(range_msk).idxmin(dim=dim)
    min_val = da.where(range_msk).min(dim=dim)

    # subtract mmt beta value
    diffed = da - min_val

    # mask data on each side of mmt and take cumulative max in each direction
    rhs = (
        diffed.where(diffed[dim] >= min_idx)
        .rolling({dim: len(diffed[dim])}, min_periods=1)
        .max()
    )
    lhs = (
        diffed.where(diffed[dim] <= min_idx)
        .sel({dim: slice(None, None, -1)})
        .rolling({dim: len(diffed[dim])}, min_periods=1)
        .max()
        .sel({dim: slice(None, None, -1)})
    )

    # combine the arrays where they've been masked
    clipped = rhs.fillna(lhs)
    return clipped


def _add_degree_coord(da: xr.DataArray, max_degrees: int | float) -> xr.DataArray:
    """
    Raises array to 1 ... max_degrees power, concatenating all together in a new "degree" coordinate.
    """
    if max_degrees < 2:
        # TODO: Test what actually happens for this edge case.
        # Raising an error because we're avoiding calculating da^1 because da is sometimes really big,
        # not sure the code handles this case very well and it's likely a mistake so just raising an
        # error for now.
        raise ValueError("'max_degree' arg must be >= 2")

    degree_idx = list(range(1, max_degrees + 1))
    out = xr.concat(
        [da]
        + [
            da**i for i in degree_idx[1:]
        ],  # Avoids computing ds^1 to not add tasks to dask graph when dask-backed data.
        pd.Index(degree_idx, name="degree"),
    )
    return out


def _beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates mortality impact polynomial model's beta coefficients from gamma coefficients.

    Returns a copy of `ds` with new "beta" variable.
    """
    # The ds["gamma"] has a "covarname" dimension with one element for each of the model's covariates.
    # This unpacks "covarname" so each covariate gamma coefficient is its own variable.
    # Doing this to try to make math easier to read.
    gamma_1 = ds["gamma"].sel(
        covarname="1"
    )  # coefficient for predictor tas (tas histogram bin labels).
    gamma_climtas = ds["gamma"].sel(
        covarname="climtas"
    )  # coefficient for climtas covariate.
    gamma_loggdppc = ds["gamma"].sel(
        covarname="loggdppc"
    )  # coefficient log of GDP per capita covariate.

    # Remember, annual histograms as input use histogram bin labels ("tas_bin") as "tas".
    # Creates a "degree" coordinate and populates it with tas^1, tas^2, tas^3, etc. equal to degrees in polynomial.
    tas = _add_degree_coord(ds["tas_bin"], max_degrees=gamma_1["degree"].size)
    # Do it this way so we don't need to repeat the same math for each degree of the polynomial below.

    #  γ_1 * tas + γ_climtas * climtas * tas + γ_loggdppc * loggdppc * tas
    # term for each of the polynomial degrees (∵ "degree" is a coordinate for variables that vary by degree).
    beta = (
        gamma_1 * tas
        + gamma_climtas * ds["climtas"] * tas
        + gamma_loggdppc * ds["loggdppc"] * tas
    ).sum("degree")  # Sum together terms for all degrees of polynomial.

    # Uclip beta across tas histogram's bin labels, so basically across range of daily temperature values.
    beta = uclip(beta, dim="tas_bin")
    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign({"beta": beta})


# If you have gamma and need to compute beta.
mortality_impact_model_gamma = muuttaa.Projector(
    preprocess=_beta_from_gamma,  # Not sure this should actually be a preprocess but I'm lazy.
    project=_mortality_impact_model,
    postprocess=_no_processing,
)
     

segment_weights = muuttaa.SegmentWeights(xr.open_zarr(SEGMENT_WEIGHTS_URI))
socioeconomics = xr.open_zarr(SOCIOECONOMICS_URI)
gammas = xr.open_zarr(GAMMA_URI)


# TODO: This doesn't work as we don't have all the socioeconomics we need in current project data.
impact_params = xr.Dataset(
    {
        "loggdppc": socioeconomics["loggdppc"],
        "gamma": gammas["gamma_mean"],
        #"gamma": gammas["gamma_sampled"],  # Run with MVN sampled gammas
        "share": socioeconomics["pop_share"]
    }
)

# Running a single test dataset.
# TODO: We really should rechunk this as part of cleaning.
cmip = xr.open_datatree(CMIP_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 20, "lat": 90, "lon": 90}
)

test_ds = cmip["rcp45/ACCESS1-0"].ds


# TODO: This scales too large for dask graph when on long timeseries and global hierid regions.
# Need dask cluster for these climate transformations.
with GatewayCluster(
        worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
) as cluster:
    client = cluster.get_client()
    print(client.dashboard_link)
    cluster.scale(50)

    transformed = muuttaa.apply_transformations(
        test_ds,
        regionalize=segment_weights,
        strategies=[
            make_climtas,
            make_tas_20yrmean_annual_histogram,
        ],
    )
    transformed = transformed.compute()

transformed = (
    transformed
    .assign_coords(tas_bin=(transformed["tas_bin"] - 273))
    .assign(climtas=(transformed["climtas"] - 273))
    .sel(year=slice(1990, 2098))  # Years outside this have NA due to climtas rolling operations.
)

mortality_impacts = muuttaa.project(
    transformed,
    model=mortality_impact_model_gamma,
    parameters=impact_params,
)
mortality_impacts = mortality_impacts.compute()

# Usually don't need dask cluster to value these damages.
mortality_damages = muuttaa.project(
    mortality_impacts,
    model=mortality_valuation_model,
    parameters=valuation_params,
)
mortality_damages = mortality_damages.compute()
print(mortality_damages)
