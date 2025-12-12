"""
Logic for mortality impact and damage projection.
"""

from muuttaa import Projector
import pandas as pd
import xarray as xr


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _mortality_effect_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins
    effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin")
    return xr.Dataset({"effect": effect})


# If you already have beta.
mortality_effect_model = Projector(
    preprocess=_no_processing,
    project=_mortality_effect_model,
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

    # Coefficient for predictor tas (tas histogram bin labels).
    gamma_1 = ds["gamma"].sel(covarname="1", drop=True)
    # coefficient for climtas covariate and coefficient log of GDP per capita covariate.
    gamma_covar = ds["gamma"].sel(covarname=["climtas", "loggdppc"])
    # The actual covariates to pair with the coefficients.
    covar = xr.concat(
        [ds["climtas"], ds["loggdppc"]],
        dim=xr.DataArray(["climtas", "loggdppc"], dims="covarname", name="covarname"),
    )

    # Remember, annual histograms as input use histogram bin labels ("tas_bin") as "tas".
    # Creates a "degree" coordinate and populates it with tas^1, tas^2, tas^3, etc. equal to degrees in polynomial.
    tas = _add_degree_coord(ds["tas_bin"], max_degrees=gamma_1["degree"].size)
    # Do it this way so we don't need to repeat the same math for each degree of the polynomial below.

    #  γ_1 * tas + γ_climtas * climtas * tas + γ_loggdppc * loggdppc * tas
    # term for each of the polynomial degrees (∵ "degree" is a coordinate for variables that vary by degree).
    beta0 = xr.dot(gamma_1, tas, dim=["degree"], optimize=True)
    beta1 = xr.dot(gamma_covar, covar, tas, dim=["covarname", "degree"], optimize=True)
    beta = beta0 + beta1

    # Uclip beta across tas histogram's bin labels, so basically across range of daily temperature values.
    beta = uclip(beta, dim="tas_bin")
    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign({"beta": beta})


# If you have gamma and need to compute beta.
mortality_effect_model_gamma = Projector(
    preprocess=_beta_from_gamma,  # Not sure this should actually be a preprocess but I'm lazy.
    project=_mortality_effect_model,
    postprocess=_no_processing,
)
