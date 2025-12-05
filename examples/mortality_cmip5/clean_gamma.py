"""
Generates a Zarr Store with structured mean and random samples of mortality's gamma parameter. These are created from an input CSVV file.

This structured gamma data should be created from the CSVV before the projection system is run so the projection system can use these gammas as inputs.

These gammas are pre-created so the projection system itself can be deterministic. This also helps to ensure we can replicate outputs.
"""

%pip install metacsv==0.1.1

import csv
import datetime
import os
import re
import uuid

import fsspec
import metacsv
import numpy as np
import xarray as xr


UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(
    f"""
        {START_TIME=}
        {UID=}
    """
)

CSVV_URI = "gs://impactlab-data-scratch/brews/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv"  # Moved into scratch from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/csvvs/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv
OUT_ZARR = "gs://impactlab-data-scratch/brews/clean_gamma.zarr"
SEED = 42
N_SAMPLES = 15

# NOTE: If you change these, you will likely need to change the structure of the output Dataset.
N_AGE_COHORT = 3
N_POLYNOMIAL_DEGREES = 4
N_COVARNAMES = 3
GAMMA_SHAPE = [N_AGE_COHORT, N_POLYNOMIAL_DEGREES, N_COVARNAMES]


# Functions cut-n-pasted from CARB EJ code. Lightly modified to use fsspec.
def read_csvv(filename):
    """Interpret a CSVV file into a dictionary of the included information.

    Specific implementation is described in the two CSVV version
    readers, `read_girdin` and `csvvfile_legacy.read`.
    """
    with fsspec.open(filename, "r") as fp:
        attrs, coords, variables = metacsv.read_header(fp, parse_vars=True)

        # Clean up variables
        for variable in variables:
            vardef = variables[variable[0]]
            assert isinstance(vardef, dict), f"Variable definition {vardef} malformed."
            if "unit" in vardef:
                fullunit = vardef["unit"]
                if "]" in fullunit:
                    vardef["unit"] = fullunit[: fullunit.index("]")]
            else:
                print(f"WARNING: Missing unit for variable {variable}.")
                vardef["unit"] = None

        data = {"attrs": attrs, "variables": variables, "coords": coords}

        # `attrs` should have "csvv-version" otherwise should be read in with
        # `csvvfile_legacy.read` - but I'm not sure what this actually is.
        csvv_version = attrs["csvv-version"]
        if csvv_version == "girdin-2017-01-10":
            return _read_girdin(data, fp)
        else:
            raise ValueError("Unknown version " + csvv_version)


def _read_girdin(data, fp):
    """Interpret a Girdin version CSVV file into a dictionary of the
    included inforation.

    A Girdin CSVV has a lists of predictor and covariate names, which
    are matched up one-for-one.  This offered more flexibility and
    clarity than the previous version of CSVV files.

    Parameters
    ----------
    data : dict
        Meta-data from the MetaCSV description.
    fp : file pointer
        File pointer to the start of the file content.

    Returns
    -------
    dict
        Dictionary with MetaCSV information and the predictor and
    covariate information.
    """
    reader = csv.reader(fp)
    variable_reading = None

    for row in reader:
        if len(row) == 0 or (len(row) == 1 and len(row[0].strip()) == 0):
            continue
        row[0] = row[0].strip()

        if row[0] in [
            "observations",
            "prednames",
            "covarnames",
            "gamma",
            "gammavcv",
            "residvcv",
        ]:
            data[row[0]] = []
            variable_reading = row[0]
        else:
            if variable_reading is None:
                print("No variable queued.")
                print(row)
            assert variable_reading is not None
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) == 1:
                row = row[0].split("\t")
            if len(row) == 1:
                row = re.split(r"\s", row[0])
            data[variable_reading].append([x.strip() for x in row])

    data["observations"] = float(data["observations"][0][0])
    data["prednames"] = data["prednames"][0]
    data["covarnames"] = data["covarnames"][0]
    data["gamma"] = np.array(list(map(float, data["gamma"][0])))
    data["gammavcv"] = np.array([list(map(float, row)) for row in data["gammavcv"]])
    data["residvcv"] = np.array([list(map(float, row)) for row in data["residvcv"]])
    return data


def main():
    csvv = read_csvv(CSVV_URI)
    gamma_mean = csvv["gamma"].reshape(GAMMA_SHAPE)
    # This gets us the median.

    rng = np.random.default_rng(SEED)
    gamma_samples_raw = rng.multivariate_normal(csvv["gamma"], csvv["gammavcv"], N_SAMPLES)

    # Add additional dim for samples drawn, and reshape flat array to match structure.
    samples_shape = [N_SAMPLES] + GAMMA_SHAPE
    gamma_samples = gamma_samples_raw.reshape(samples_shape)

    # NOTE: This has some magic coordinates that need to change if the CSVV structure changes.
    g = xr.Dataset(
        {
            "gamma_mean": (["age_cohort", "degree", "covarname"], gamma_mean),
            "gamma_sampled": (
                ["sample", "age_cohort", "degree", "covarname"],
                gamma_samples,
            ),
        },
        coords={
            "age_cohort": (["age_cohort"], ["age1", "age2", "age3"]),
            "covarname": (["covarname"], ["1", "climtas", "loggdppc"]),
            "degree": (["degree"], np.arange(4) + 1),
            "sample": np.arange(N_SAMPLES),
        },
    )

    # TODO: Add metadata with source CSVV path ("history"?), etc...?
    g.attrs["uid"] = UID
    g.attrs["created_at"] = START_TIME 

    g = g.chunk({"sample": 1})
    g.to_zarr(OUT_ZARR, mode="w")
    print(f"Written to {OUT_ZARR}") 


if __name__ == "__main__":
    main()
