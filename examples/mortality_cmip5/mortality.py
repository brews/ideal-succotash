%pip install muuttaa==0.1.1
%pip install git+https://github.com/brews/ideal-succotash@d92c330985bb04cce0af2df11f67403b86da48b2 --no-deps

import datetime
import os
import uuid

from dask_gateway import GatewayCluster
import muuttaa
import numpy as np
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram

import ideal_succotash.mortality.transformation as transformation

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


# TODO: This scales too large for dask graph when on long timeseries and global hierid regions.
# Need dask cluster for these climate transformations.
with GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro") as cluster:
    client = cluster.get_client()
    print(client.dashboard_link)
    cluster.scale(50)

    transformed = muuttaa.apply_transformations(
        test_ds.drop_encoding(),
        regionalize=segment_weights,
        strategies=[
            transformation.make_climtas,
            transformation.make_tas_20yrmean_annual_histogram,
        ],
    )
    transformed = transformed.chunk({"region": "auto", "year": -1, "tas_bin": -1})

    transformed = (
        transformed
        .assign_coords(tas_bin=(transformed["tas_bin"] - 273.15))
        .assign(climtas=(transformed["climtas"] - 273.15))
        .sel(year=slice(1980, 2099)) # Years outside this have NA due to climtas rolling operations.
    )
    # TODO: Unit conversion should happen in transformation.

    transformed = transformed.compute()

