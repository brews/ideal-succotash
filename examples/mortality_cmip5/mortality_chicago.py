
%pip install muuttaa==0.1.2
%pip install git+https://github.com/brews/ideal-succotash@f9ae683092875e1ff5a463f280c227909490e6d5 --no-deps

import datetime
import os
import uuid

import muuttaa
from muuttaa import TransformationStrategy, Projector
import numpy as np
import numba
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram

import ideal_succotash.mortality.transformation as transformation
import ideal_succotash.mortality.projection as projection


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

CMIP_URI = "gs://impactlab-data-scratch/brews/90ddd653-8b41-43fe-9983-6fafde9f7925/clean_cmip5.zarr"
SOCIOECONOMICS_URI = "/gcs/impactlab-data-scratch/brews/SSP3.nc4"
SEGMENT_WEIGHTS_URI = "gs://impactlab-data-scratch/brews/clean_segment_weights.zarr"
GAMMA_URI = "gs://impactlab-data-scratch/brews/clean_gamma.zarr"


# TODO: This cleaning should be in the segment weights cleaning script.
segment_weights_raw = xr.open_zarr(SEGMENT_WEIGHTS_URI)[["lat", "lon", "weight", "region"]].load()
segment_weights_raw["region"] =segment_weights_raw["region"].astype('str')
target_region="USA.14.608"  # Chicago
segment_weights = muuttaa.SegmentWeights(
    segment_weights_raw.where(segment_weights_raw["region"] == target_region, drop=True)
)

socioeconomics = xr.open_dataset(SOCIOECONOMICS_URI)
pop = xr.concat(
    [socioeconomics["pop0to4"], socioeconomics["pop5to64"], socioeconomics["pop65plus"]],
    dim=xr.DataArray(["age1", "age2", "age3"], dims="age_cohort", name="age_cohort"),
)
pop_share = pop / socioeconomics["pop"]
gammas = xr.open_zarr(GAMMA_URI)
effect_params = xr.Dataset(
    {
        "loggdppc": np.log(socioeconomics["gdppc"]),
        "gamma": gammas["gamma_mean"],
        # "gamma": gammas["gamma_sampled"],  # Run with MVN sampled gammas
    }
).sel(year=slice(2010, 2100), model="low", region=target_region)

# Running a single test dataset.
# TODO: We really should rechunk this as part of cleaning.
cmip = xr.open_datatree(CMIP_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 50, "lat": 90, "lon": 90}
)
test_ds = cmip["rcp85/CCSM4"].ds



# Regionalization and transformation.
transformed = muuttaa.apply_transformations(
    test_ds,
    regionalize=segment_weights,
    strategies=[
        transformation.make_climtas,
        transformation.make_tas_annual_histogram,
    ],
)
# Some extra post-processing that was part of the carb projection.
# TODO: Unit conversion should happen in transformation.
transformed = (
    transformed
    .assign_coords(tas_bin=(transformed["tas_bin"] - 273.15))
    .assign(climtas=(transformed["climtas"] - 273.15))
    .sel(year=slice(2001, 2100))
)
transformed = transformed.compute()

# Project mortality effects.
projected = muuttaa.project(transformed, model=projection.mortality_impact_adaptations_model, parameters=effect_params)
# projected = muuttaa.project(
#     merged,
#     model=projection.mortality_effect_model_gamma,
#     parameters=xr.Dataset({}),
# )
# projected = projected.compute()


# # Doing a standard impact without population share adjustment
# impact = projected["effect"] - projected["effect"].sel(year=2015)
# impact.name = "impact"


# # Playing with things, adjusting with age's 'share' of total population.
# effect = projected["effect"] * pop_share.sel(model="low", region=target_region)
# impact = effect - effect.sel(year=2010)
# impact.name = "impact"
# impact.plot(x="year", hue="age_cohort")

projected["impact"].sel(year=slice(2010, 2099)).plot(hue="age_cohort", col="adaptation")

