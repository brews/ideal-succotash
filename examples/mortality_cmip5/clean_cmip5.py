# Clean NASA NEX CMIP5 with surrogates

from dataclasses import dataclass
import datetime
from collections.abc import Iterable
import os
import uuid

from dask_gateway import GatewayCluster
import xarray as xr


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

OUT_CMIP5_ZARR = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/cmip5.zarr"
OUT_CMIP5_CONCAT_ZARR = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/clean_cmip5.zarr"

CMIP5_PATTERNMODELS_RCP45 = [
    f"pattern{i}" for i in [1, 2, 3, 5, 6, 27, 28, 29, 30, 31, 32]
]
CMIP5_PATTERNMODELS_RCP85 = [
    f"pattern{i}" for i in [1, 2, 3, 4, 5, 6, 28, 29, 30, 31, 32, 33]
]
CMIP5_MODELS = [
    "ACCESS1-0",
    "CNRM-CM5",
    "GFDL-ESM2G",
    "MIROC-ESM",
    "MPI-ESM-MR",
    "inmcm4",
    "BNU-ESM",
    "CSIRO-Mk3-6-0",
    "GFDL-ESM2M",
    "MIROC-ESM-CHEM",
    "MRI-CGCM3",
    "CCSM4",
    "CanESM2",
    "IPSL-CM5A-LR",
    "MIROC5",
    "NorESM1-M",
    "CESM1-BGC",
    "GFDL-CM3",
    "IPSL-CM5A-MR",
    "MPI-ESM-LR",
    "bcc-csm1-1",
]
CMIP5_RCP45_SMME_MODELS = CMIP5_MODELS + CMIP5_PATTERNMODELS_RCP45
CMIP5_RCP85_SMME_MODELS = CMIP5_MODELS + CMIP5_PATTERNMODELS_RCP85
YEAR_RANGE_HISTORICAL = range(1950, 2006)
YEAR_RANGE_SCENARIO = range(
    2006, 2100
)  # Not everything goes to 2100. E.g. MIROC5 is only 2099.

# This was originally in rhg-data GCS bucket in the original analysis but it appears to have moved to impactlab-data.
# FILE_PATTERN = (
#     "gs://impactlab-data/climate/source_data/NASA/NEX-GDDP/BCSD/reformatted/"
#     "{scenario_id}/{source_id}/{variable_id}/{variable_id}_day_BCSD_{scenario_id}_r1i1p1_{source_id}_{year}/{version}.nc4"
# )
FILE_PATTERN = (
    "/gcs/impactlab-data/climate/source_data/NASA/NEX-GDDP/BCSD/reformatted/"
    "{scenario_id}/{source_id}/{variable_id}/{variable_id}_day_BCSD_{scenario_id}_r1i1p1_{source_id}_{year}/{version}.nc4"
)


# Match numbered pattern to its source_id. To find matching historical scenario.
pattern2source_map = {
    "rcp45": {
        "pattern1": "MRI-CGCM3",
        "pattern2": "GFDL-ESM2G",
        "pattern3": "MRI-CGCM3",
        "pattern4": "GFDL-ESM2G",
        "pattern5": "MRI-CGCM3",
        "pattern6": "GFDL-ESM2G",
        "pattern27": "GFDL-CM3",
        "pattern28": "CanESM2",
        "pattern29": "GFDL-CM3",
        "pattern30": "CanESM2",
        "pattern31": "GFDL-CM3",
        "pattern32": "CanESM2",
    },
    "rcp85": {
        "pattern1": "MRI-CGCM3",
        "pattern2": "GFDL-ESM2G",
        "pattern3": "MRI-CGCM3",
        "pattern4": "GFDL-ESM2G",
        "pattern5": "MRI-CGCM3",
        "pattern6": "GFDL-ESM2G",
        "pattern28": "GFDL-CM3",
        "pattern29": "CanESM2",
        "pattern30": "GFDL-CM3",
        "pattern31": "CanESM2",
        "pattern32": "GFDL-CM3",
        "pattern33": "CanESM2",
    },
}

# Ug. This could be more effective, bit it's easy enough to read like this.

# Not sure I like this approach better but it makes it easy to track the multiple levels of metadata.
# Could be useful to build out this type system but not worth it if this is a rare one-off.


def _year_from_uri(uri: str) -> int:
    """
    Figure out what year this is supposed to be based on source file uri for CIL CMIP5 data

    Asuming fl_path's end is like ".../tasmax_day_BCSD_rcp45_r1i1p1_pattern30_2010/1.0.nc4"
    """
    # Working backwards from path because front changes based on bucket name or if accessed with GCSfuse.
    return int(uri.split("/")[-2].split("_")[-1])


def _preprocess_fn(ds: xr.Dataset, calendar: str = "noleap") -> xr.Dataset:
    """Required preprocessing for reading NetCDF files and merging with xarray"""
    # If this func gets too complex, break each task into its own function and apply them all here.

    # This is most likely a pattern model with "day" instead of "time" coord. It needs time based on a proper compatible calendar.
    if "day" in ds.coords:
        # Figure out what year this is supposed to be based on source file path.
        year = _year_from_uri(ds.encoding["source"])
        assert (year < 3000) and (year > 1000)  # Additional non essential sanity check.

        new_time = xr.cftime_range(
            start=f"{year}-01-01 12:00:00",
            end=f"{year}-12-31 12:00:00",
            calendar=calendar,
            freq="D",
        )
        # TODO It's a 365, or 366, or something calendar for rcp45/pattern1: ValueError: conflicting sizes for dimension 'day': length 366 on 'day' and length 365 on {'lat': 'lat', 'lon': 'lon', 'day': 'tas'}

        ds = (
            ds.assign_coords(day=new_time).swap_dims(day="time").rename_vars(day="time")
        )

        # In some pattern models, "time" is not included as an index for the dataset.
        if "time" not in ds.indexes:
            ds = ds.set_xindex("time")

    # Some files like impactlab-data/climate/source_data/NASA/NEX-GDDP/BCSD/reformatted/rcp45/pattern3/tasmin/tasmin_day_BCSD_rcp45_r1i1p1_pattern3_2098/1.0.nc4
    # appear to have nlat and nlon dims and lat and lon as data dims, but it's
    # inconsistent across files. Trying to fix it so it's more consistent:
    if ("lat" not in ds.coords) and ("nlat" in ds.dims):
        ds = ds.set_coords(["lat"]).swap_dims(nlat="lat")
    if ("lon" not in ds.coords) and ("nlon" in ds.dims):
        ds = ds.set_coords(["lon"]).swap_dims(nlon="lon")

    ds = ds.convert_calendar(calendar=calendar, dim="time", use_cftime=True)

    return ds


# If the ensemble is a tree, each member is a leaf.
@dataclass
class EnsembleLeaf:
    name: str
    source_id: str
    scenario_id: str
    variable_id: str
    source_uris: Iterable[str]

    def _ordered_uris(self) -> Iterable[str]:
        """Orders self.source_uris ascending by year by inferring year from uri string"""
        return sorted(self.source_uris, key=_year_from_uri)

    def to_dataset(self) -> xr.Dataset:
        """Merge self.source_urls into single xr.Dataset, by 'time'"""
        # xr magic can't merge pattern models without combine='nested' and assumption that input paths are ordered by year.
        ordered_uris = self._ordered_uris()
        ds = xr.open_mfdataset(
            ordered_uris,
            parallel=True,
            combine="nested",
            concat_dim="time",
            preprocess=_preprocess_fn,
        )
        ds.attrs["source_uris"] = str(self.source_uris)
        ds.attrs["source_id"] = self.source_id
        ds.attrs["scenario_id"] = self.scenario_id
        ds.attrs["name"] = self.name
        return ds


@dataclass
class VariableSpec:
    variable_id: str
    version: str


@dataclass
class ScenarioSpec:
    scenario_id: str
    years: Iterable[int]
    models: Iterable[str]


# Only need "tas" data for mortality projection.
variable_specs = [
    VariableSpec(variable_id="tas", version="1.1"),
    # VariableSpec(variable_id="tasmin", version="1.0"),
    # VariableSpec(variable_id="tasmax", version="1.0"),
]

scenario_specs = [
    ScenarioSpec(
        scenario_id="historical", years=YEAR_RANGE_HISTORICAL, models=CMIP5_MODELS
    ),
    ScenarioSpec(
        scenario_id="rcp45", years=YEAR_RANGE_SCENARIO, models=CMIP5_RCP45_SMME_MODELS
    ),
    ScenarioSpec(
        scenario_id="rcp85", years=YEAR_RANGE_SCENARIO, models=CMIP5_RCP85_SMME_MODELS
    ),
]

cluster = GatewayCluster(
    worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(100)

targets: dict[str, EnsembleLeaf] = {}
for variable_spec in variable_specs:
    for scenario_spec in scenario_specs:
        for model in scenario_spec.models:
            target_key = (
                f"/{scenario_spec.scenario_id}/{model}/{variable_spec.variable_id}"
            )

            # Future projections some models are just "pattern*", need to note the source_id.
            source_id = model
            if "pattern" in model:
                source_id = pattern2source_map[scenario_spec.scenario_id][model]

            # Ugg.
            uris: list[str] = [
                FILE_PATTERN.format(
                    scenario_id=scenario_spec.scenario_id,
                    variable_id=variable_spec.variable_id,
                    version=variable_spec.version,
                    source_id=model,
                    year=yr,
                )
                for yr in scenario_spec.years
            ]

            target = EnsembleLeaf(
                name=model,
                source_id=source_id,
                scenario_id=scenario_spec.scenario_id,
                variable_id=variable_spec.variable_id,
                source_uris=uris,
            )

            targets[target_key] = target

# Like this so it's easier to debug.
to_dt = {}
for k, v in targets.items():
    print(k)  # DEBUG
    to_dt[k] = v.to_dataset()
cmip5 = xr.DataTree.from_dict(to_dt, name="cmip5")
cmip5 = cmip5.chunk({"time": 365, "lat": 360, "lon": 360})
cmip5.to_zarr(OUT_CMIP5_ZARR, mode="w")
print(OUT_CMIP5_ZARR)

############ Try to concat/merge hist/future and variable projections #########
# NOTE: Graph is ~22 MiB so maybe break these down into a time concat and a variable merge step.
cluster.scale(100)
cmip5 = xr.open_datatree(OUT_CMIP5_ZARR, engine="zarr", chunks={})

# Parse the tree, merging historical and RCP scenarios. Need to handle pattern models appropriately.
merged = {}
for scenario_name, scenario_tree in cmip5.children.items():
    if scenario_name.lower() == "historical":
        continue

    for source_name, source_tree in scenario_tree.children.items():
        k = f"/{scenario_name}/{source_name}"
        print(k)

        # Trying to combine all variables and hist + future proj together.
        variable_dss = []
        for variable_name, variable_tree in source_tree.children.items():
            print(f"\t{variable_name}")

            # Account for pattern models having different name in future and historical projections.
            # Historical will use the pattern models original "source_id", this
            # should be in the pattern models attrs.
            source_id = variable_tree.attrs["source_id"]

            hist_ds = cmip5[f"historical/{source_id}/{variable_name}"].ds
            future_ds = variable_tree.ds

            concat_ds = xr.concat(
                [hist_ds, future_ds], dim="time", combine_attrs="drop_conflicts"
            )
            variable_dss.append(concat_ds)

        merged[k] = xr.merge(variable_dss, combine_attrs="drop_conflicts")

new_dt = xr.DataTree.from_dict(merged, name="cmip5")
# I think we need to rechunk to ensure everything is aligned.
new_dt = new_dt.chunk({"time": 365, "lat": 360, "lon": 360})

new_dt.to_zarr(OUT_CMIP5_CONCAT_ZARR, mode="w")
print(OUT_CMIP5_CONCAT_ZARR)

cluster.scale(0)
cluster.shutdown()
