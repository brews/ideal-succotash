%pip install muuttaa==0.1.1
%pip install git+https://github.com/brews/ideal-succotash@b080eb37e184f1e3ef8883ccbe24691ae015aee1 --no-deps

import datetime
import os
import uuid

from dask_gateway import GatewayCluster
import muuttaa
import numpy as np
import xarray as xr

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

CMIP_URI = "gs://impactlab-data-scratch/brews/3e2f02fa-f607-4d14-ae53-12be6b3f1e10/clean_cmip5.zarr"
SOCIOECONOMICS_URI = "gs://impactlab-data/gcp/integration_replication_sync/inputs/econ/raw/integration-econ-bc39.zarr"
SEGMENT_WEIGHTS_URI = "gs://impactlab-data-scratch/brews/clean_segment_weights.zarr"
GAMMA_URI = "gs://impactlab-data-scratch/brews/clean_gamma.zarr"


# TODO: This cleaning should be in the segment weights cleaning script.
segment_weights_raw = xr.open_zarr(SEGMENT_WEIGHTS_URI)[["lat", "lon", "weight", "region"]]
segment_weights_raw["region"] =segment_weights_raw["region"].astype('str')
segment_weights = muuttaa.SegmentWeights(segment_weights_raw.load())

socioeconomics = xr.open_zarr(SOCIOECONOMICS_URI)
gammas = xr.open_zarr(GAMMA_URI)


effect_params = xr.Dataset(
    {
        "loggdppc": np.log(socioeconomics["gdppc"]),
        "gamma": gammas["gamma_mean"],
        #"gamma": gammas["gamma_sampled"],  # Run with MVN sampled gammas
    }
).sel(year=slice(2010, 2099))

# Running a single test dataset.
# TODO: We really should rechunk this as part of cleaning.
cmip = xr.open_datatree(CMIP_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 50, "lat": 90, "lon": 90}
)
test_ds = cmip["rcp45/ACCESS1-0"].ds


# Need dask cluster for these climate transformations and projection.
cluster = GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)

# Regionalization and transformation.
transformed = muuttaa.apply_transformations(
    test_ds.drop_encoding(),
    regionalize=segment_weights,
    strategies=[
        transformation.make_climtas,
        transformation.make_tas_20yrmean_annual_histogram,
    ],
)

# Some extra post-processing that was part of the carb projection.
# TODO: Unit conversion should happen in transformation.
transformed = (
    transformed
    .assign_coords(tas_bin=(transformed["tas_bin"] - 273.15))
    .assign(climtas=(transformed["climtas"] - 273.15))
    .sel(year=slice(2010, 2099)) # Match socioecon data range.
)
transformed = transformed.compute()
out_path = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/transformed.zarr"
transformed.chunk({"region": "auto", "year": -1, "tas_bin": -1}).unify_chunks().drop_encoding().to_zarr(out_path, mode="w")
print(out_path)


transformed = xr.open_zarr(out_path)
# Merging and rechunking projection input  here explicitly because this can be sensitive.
projection_input = xr.merge([transformed, effect_params])
projection_input = projection_input.chunk(
    {"region": 5000, "year": -1, "tas_bin": -1, "age_cohort": -1, "degree": -1, "model": 1, "ssp": 1},
).unify_chunks().persist()

# Project mortality effects.
projected = muuttaa.project(
    projection_input,
    model=projection.mortality_effect_model_gamma,
    parameters=xr.Dataset({}),
)
projected = projected.chunk({"region": "auto", "year": -1, "age_cohort": -1})
projected.persist()

out_path = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/projected.zarr"
projected.drop_encoding().to_zarr(out_path, mode="w")
print(out_path)

cluster.scale(0)
cluster.shutdown()

