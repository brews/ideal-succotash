"""
Logic for mortality impact and damage projection.
"""

from muuttaa import Projector
import numba
import numpy as np
import pandas as pd
import xarray as xr


@numba.njit()
def _maximum_accumulate(x):
    rmax = x[0]
    y = np.empty_like(x)
    for i, val in enumerate(x):
        if val > rmax:
            rmax = val
        y[i] = rmax
    return y


@numba.guvectorize(["void(float64[:], int64, float64[:])"], "(n),()->(n)")
def _uclip_gufunc(x, idx_min, result):
    # Doing this if/else because can't return early in guvectorize funcs.
    if len(x) < 3:
        result[:] = x
    else:
        n = len(x)
        # WARNING: Throw error if min_idx_max is greater than n.
        # WARNING: Throw error if min_idx_min is greater than min_idx_max.

        # Get right side of minimum idx, ascending from minimum.
        rs = x[idx_min:n]
        # Left size, but reversed to still ascend from minimum.
        ls = x[0 : idx_min + 1][::-1]
        n_ls = len(ls)

        # Take accumulative maximum for each side and stick it in outgoing array,
        # with left hand side reversed back.
        result[0:idx_min] = _maximum_accumulate(ls)[1:n_ls][::-1]
        result[idx_min:n] = _maximum_accumulate(rs)


def uclip(x, dim, idx_min):
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
    idx_min : int
        Index of value to use as the middle "base" of the U. Usually the minimum mortality temperature.

    Returns
    -------
    clipped : xarray DataArray of the regions response functioned centered on idx_min.
    """
    # TODO: Check that `idx_min` can be found along the `dim` of `x`.
    return xr.apply_ufunc(
        _uclip_gufunc,
        x,
        idx_min,
        input_core_dims=[[dim], []],
        output_core_dims=[[dim]],
        output_dtypes=["float64"],
        dask="parallelized",
    )


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


def minimum_arg(x: xr.DataArray, *, dim="tas_bin", lmmt=10.0, ummt=30.0):
    """
    Get minimum and its associated dim label within an inclusive range along dim
    """
    # Find minimum within inclusive range, get the dim label for the minimum's position.
    min_arg = x.where((x[dim] >= lmmt) & (x[dim] <= ummt)).argmin(dim=dim)
    # Get the actual minima values.
    # Need to .compute() mmt_argmin because can't yet do vectorized indexing with dask arrays. See https://github.com/dask/dask/issues/8958
    min_value = x.isel({dim: min_arg.compute()})
    return min_value, min_arg


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

    # Find beta at minimum mortality temperature.
    mmt_beta, mmt = minimum_arg(
        beta,
        dim="tas_bin",
        lmmt=10.0,
        ummt=30.0,
    )
    # Shift so mmt beta is zero & Uclip beta across tas histogram's bin
    # labels, so basically across range of daily temperature values.
    beta = uclip(
        (beta - mmt_beta).chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt,
    )
    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign({"beta": beta})


# If you have gamma and need to compute beta.
mortality_effect_model_gamma = Projector(
    preprocess=_beta_from_gamma,  # Not sure this should actually be a preprocess but I'm lazy.
    project=_mortality_effect_model,
    postprocess=_no_processing,
)


def _beta_from_gamma_with_adaptations(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates mortality impact polynomial model's beta coefficients from gamma coefficients for adaptation scenarios.

    Returns a copy of `ds` with new "beta" variable.
    """
    # The ds["gamma"] has a "covarname" dimension with one element for each of the model's covariates.

    # This unpacks "covarname" so each covariate gamma coefficient is its own variable.
    # Doing this to try to make math easier to read.
    gamma_1 = ds["gamma"].sel(
        covarname="1",
        drop=True,
    )  # coefficient for predictor tas (tas histogram bin labels).
    gamma_climtas = ds["gamma"].sel(
        covarname="climtas",
        drop=True,
    )  # coefficient for climtas covariate.
    gamma_loggdppc = ds["gamma"].sel(
        covarname="loggdppc",
        drop=True,
    )  # coefficient log of GDP per capita covariate.

    # Remember, annual histograms as input use histogram bin labels ("tas_bin") as "tas".
    # Creates a "degree" coordinate and populates it with tas^1, tas^2, tas^3, etc. equal to degrees in polynomial.
    tas = _add_degree_coord(ds["tas_bin"], max_degrees=gamma_1["degree"].size)
    # Do it this way so we don't need to repeat the same math for each degree of the polynomial below.

    baseline_year = 2015
    climtas_baseline = ds["climtas"].sel(year=baseline_year)
    loggdppc_baseline = ds["loggdppc"].sel(year=baseline_year)
    #  γ_1 * tas + γ_climtas * climtas * tas + γ_loggdppc * loggdppc * tas
    # term for each of the polynomial degrees (∵ "degree" is a coordinate for variables that vary by degree).
    beta_1 = (gamma_1 * tas).sum("degree")
    beta_climtas = (gamma_climtas * ds["climtas"] * tas).sum("degree")
    beta_loggdppc = (gamma_loggdppc * ds["loggdppc"] * tas).sum("degree")

    beta_climtas_baseline = (gamma_climtas * climtas_baseline * tas).sum("degree")
    beta_loggdppc_baseline = (gamma_loggdppc * loggdppc_baseline * tas).sum("degree")

    # Increased money for adaptation can't cause maladaptation so use baseline
    # logGDPpc sensitivity when it reduces projected mortality rate.
    beta_loggdppc_goodmoney = beta_loggdppc.where(
        beta_loggdppc < beta_loggdppc_baseline,
        other=beta_loggdppc_baseline,
    )

    # Prevent maladaption from climtas adaptation. So use baseline
    # climtas sensitivity when it reduces projected mortality rate.
    beta_climtas_goodclimtas = beta_climtas.where(
        beta_climtas < beta_climtas_baseline,
        other=beta_climtas_baseline,
    )

    beta_noadapt_unshifted = beta_1 + beta_climtas_baseline + beta_loggdppc_baseline

    # Find idx with lowest beta & minimum mortality temperature (mmt) within degC range in the baseline period.
    mmt_beta, mmt = minimum_arg(
        beta_noadapt_unshifted,
        dim="tas_bin",
        lmmt=10.0,
        ummt=30.0,
    )

    beta_noadapt = beta_noadapt_unshifted - mmt_beta
    beta_incadapt = beta_1 + beta_loggdppc_goodmoney + beta_climtas_baseline - mmt_beta
    beta_fulladapt = (
        beta_1 + beta_loggdppc_goodmoney + beta_climtas_goodclimtas - mmt_beta
    )

    # u-clip. Makes the response function shaped like a big U, centered on the mmt.
    beta_noadapt_uclip = uclip(
        beta_noadapt.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt,
    )
    beta_incadapt_uclip = uclip(
        beta_incadapt.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt,
    )
    beta_fulladapt_uclip = uclip(
        beta_fulladapt.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt,
    )

    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign(
        {
            "beta_noadapt": beta_noadapt_uclip,
            "beta_incadapt": beta_incadapt_uclip,
            "beta_fulladapt": beta_fulladapt_uclip,
        },
    )


def _mortality_impact_adaptations_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins for each adaptation, with each adapation along a new dim.
    effect = xr.concat(
        [
            (ds["histogram_tas"] * ds["beta_noadapt"]).sum(dim="tas_bin"),
            (ds["histogram_tas"] * ds["beta_incadapt"]).sum(dim="tas_bin"),
            (ds["histogram_tas"] * ds["beta_fulladapt"]).sum(dim="tas_bin"),
        ],
        dim=xr.DataArray(
            ["noadapt", "incadapt", "fulladapt"], dims="adaptation", name="effect"
        ),
    )

    # Subtract baseline year so response effect become impact.
    baseline_year = 2015
    impact = effect - effect.sel(year=baseline_year)
    return xr.Dataset({"impact": impact})


mortality_impact_adaptations_model = Projector(
    preprocess=_beta_from_gamma_with_adaptations,  # not sure this should actually be a preprocess but i'm lazy.
    project=_mortality_impact_adaptations_model,
    postprocess=_no_processing,
)
