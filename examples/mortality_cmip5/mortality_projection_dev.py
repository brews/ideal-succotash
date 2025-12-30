%pip install muuttaa==0.1.1
%pip install git+https://github.com/brews/ideal-succotash@f9ae683092875e1ff5a463f280c227909490e6d5 --no-deps

import datetime
import os
import uuid

from dask_gateway import GatewayCluster
import muuttaa
from muuttaa import TransformationStrategy, Projector
import numpy as np
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


cluster = GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)



segment_weights_raw = xr.open_zarr(SEGMENT_WEIGHTS_URI)[["lat", "lon", "weight", "region"]].load()
segment_weights_raw["region"] = segment_weights_raw["region"].astype('str')
segment_weights = muuttaa.SegmentWeights(segment_weights_raw)

socioeconomics = xr.open_dataset(SOCIOECONOMICS_URI)
gammas = xr.open_zarr(GAMMA_URI)
effect_params = xr.Dataset(
    {
        "loggdppc": np.log(socioeconomics["gdppc"]),
        "gamma": gammas["gamma_mean"],
        #"gamma": gammas["gamma_sampled"],  # Run with MVN sampled gammas
    }
).sel(year=slice(2010, 2100), model="low")

# Running a single test dataset.
# TODO: We really should rechunk this as part of cleaning.
cmip = xr.open_datatree(CMIP_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 50, "lat": 90, "lon": 90}
)
test_ds = cmip["rcp85/CCSM4"].ds
# test_ds = cmip["rcp85/CCSM4"].ds.drop_encoding().persist()


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

cluster.scale(0)
cluster.shutdown()


projection_input = xr.merge([transformed, effect_params])
projection_input = projection_input.chunk(
    {"region": 5000, "year": -1, "tas_bin": -1, "age_cohort": 1, "degree": -1}, #"model": 1},, "ssp": 1},
).unify_chunks()

# Project mortality impacts.
projected = muuttaa.project(
    projection_input,
    model=mortality_impact_adaptations_model,
    parameters=xr.Dataset({}),
)
projected = projected.chunk({"region": "auto", "year": -1, "age_cohort": 1})
projected = projected.persist()
projected

# Plot to sanity check.
projected["impact"].sel(year=slice(2010, 2099), region="USA.14.608").plot(hue="age_cohort", col="adaptation")

