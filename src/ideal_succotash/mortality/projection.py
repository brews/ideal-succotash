"""
Logic for mortality impact and damage projection.
"""

from typing import Any, Sequence

from muuttaa import Projector
import numba
import numpy as np
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
        along dimensions `dim` for all other dimensions independently. This is usually
        the "dose" climate or weather variable, e.g., daily-average air 
        temperature ('tas' or 'tas_bin').
    idx_min : int
        Index of value to use as the middle "base" of the "U". Found along the 
        dimension 'dim' of `da`. In these mortality projections, this is often the 
        index of the Minimum Mortality Temperature (MMT).

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
    # Need to .compute() mmt_argmin because can't yet do vectorized indexing with dask arrays. See https://github.com/dask/dask/issues/8958
    min_arg = x.where((x[dim] >= lmmt) & (x[dim] <= ummt)).argmin(dim=dim).compute()
    # Get the actual minima values.
    min_value = x.isel({dim: min_arg})
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
        dim=xr.DataArray(degree_idx, dims="degree", name="degree"),
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


def _calculate_beta(ds: xr.Dataset) -> xr.DataArray:
    """
    Helper function to calculate beta from gamma and covariates using a scalable Einstein notation tensordot.
    """
    # The ds["gamma"] has a "covarname" dimension with one element for each of the model's covariates.
    # Coefficient for predictor tas (tas histogram bin labels).
    gamma_1 = ds["gamma"].sel(covarname="1", drop=True)
    # coefficient for climtas covariate and coefficient log of GDP per capita covariate.
    gamma_covar = ds["gamma"].sel(covarname=["climtas", "loggdppc"])
    # The covariates to pair with the coefficients, selecting for the baseline year.
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

    return beta0 + beta1


def _calculate_shifted_baseline_beta(
    ds: xr.Dataset,
    *,
    baseline_time: Any = 2015,
    time_dim: str = "year",
    tas_bin_dim: str = "tas_bin",
    lmmt: float = 10.0,
    ummt: float = 30.0,
    covars: Sequence[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Helper to calculate baseline beta and the index of the MMT from gamma and covariates.
    """
    # This all might be overkill but whatever.
    if covars is None:
        covars = ["climtas", "loggdppc"]
    else:
        covars = list(covars)

    # Start with no-adaptation beta calculation: the covariate variables are
    # subset to the baseline year because no adaptation is allowed through time.
    # Use `.assign()` for this so we don't disturb the original data in `ds`.
    beta = _calculate_beta(
        ds.assign({k: ds[k].sel({time_dim: baseline_time}, drop=True) for k in covars})
    )

    # Find idx with lowest beta & minimum mortality temperature (MMT) within °C
    # range in the baseline period.
    mmt_beta, mmt_idx = minimum_arg(
        beta,
        dim=tas_bin_dim,
        lmmt=lmmt,
        ummt=ummt,
    )
    # Shift beta so MMT is 0.
    beta -= mmt_beta

    return beta, mmt_idx


def _incadapt_beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates mortality impact polynomial model's beta coefficients from gamma coefficients for the income-adaptation scenario.

    Returns a copy of `ds` with new "beta" variable.
    """
    baseline_year = 2015

    # Start with no-adaptation beta calculation.
    beta, mmt_idx = _calculate_shifted_baseline_beta(ds, baseline_time=baseline_year)

    # Income adaptation freezes climtas at baseline year, let logGDPpc vary but
    # shift it to the Minimum-Mortality Temperature (MMT).
    beta_incadapt = _calculate_beta(
        ds.assign(climtas=ds["climtas"].sel(year=baseline_year, drop=True))
    )
    beta_incadapt -= beta_incadapt.isel(tas_bin=mmt_idx)

    # Only use incadapt if it reduces mortality rate relative to no adaptation.
    # This is key to this style of adaptation. Sometimes called "good money clipping".
    beta_incadapt = np.minimum(beta, beta_incadapt)

    # Negative values to zero. Sometimes called "level clipping".
    beta_incadapt = beta_incadapt.clip(min=0)

    # u-clip. Makes the response function shaped like a big U, centered on the mmt.
    beta_incadapt = uclip(
        beta_incadapt.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt_idx,
    )

    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign(beta=beta_incadapt)


def _mortality_rebased_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins for each adaptation, with each adapation along a new dim.
    effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin")

    # Subtract mean of baseline period so response effect becomes impact.
    impact = effect - effect.sel(year=slice(2001, 2010)).mean(dim="year")
    return xr.Dataset({"impact": impact})


mortality_incadapt_impact_model = Projector(
    preprocess=_incadapt_beta_from_gamma,
    project=_mortality_rebased_impact_model,
    postprocess=_no_processing,
)


def _fulladapt_beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates mortality impact polynomial model's beta coefficients from gamma coefficients for the full-adaptation scenario.

    Returns a copy of `ds` with new "beta" variable.
    """
    baseline_year = 2015

    # Start with no-adaptation beta calculation to estimate the Minimum-Mortality Temperature (MMT).
    _, mmt_idx = _calculate_shifted_baseline_beta(ds, baseline_time=baseline_year)

    # Full adaptation allows climtas and logGDPpc to vary over time.
    # Shift it to the beta over the MMT, so this beta is 0.
    beta_fulladapt = _calculate_beta(ds)
    beta_fulladapt = beta_fulladapt - beta_fulladapt.isel(tas_bin=mmt_idx)

    # Like above, but with "no-income" adaptation: freezes logGDPpc at baseline
    # year, let climtas vary.
    beta_noinc = _calculate_beta(
        ds.assign(loggdppc=ds["loggdppc"].sel(year=baseline_year, drop=True))
    )
    beta_noinc = beta_noinc - beta_noinc.isel(tas_bin=mmt_idx)

    # "Good money" clipping.
    # Money cannot cause maladaptation so use a no-income rate when it's better(lower) than a income-adaptation rate.
    beta_fulladapt = np.minimum(beta_noinc, beta_fulladapt)

    # Negative values to zero. Sometimes called "level clipping".
    beta_fulladapt = beta_fulladapt.clip(min=0)

    # u-clip. Makes the response function shaped like a big U, centered on the mmt.
    beta_fulladapt = uclip(
        beta_fulladapt.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt_idx,
    )
    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign(beta=beta_fulladapt)


mortality_fulladapt_impact_model = Projector(
    preprocess=_fulladapt_beta_from_gamma,
    project=_mortality_rebased_impact_model,
    postprocess=_no_processing,
)


def _noadapt_beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates mortality impact polynomial model's beta coefficients from gamma coefficients for the no-adaptation scenario.

    Returns a copy of `ds` with new "beta" variable.
    """
    # Subset all the covariate variables to the baseline year because no adaptation is allowed.
    beta, mmt_idx = _calculate_shifted_baseline_beta(ds)

    # Just in case, clip negative values to zero. Sometimes called "level clipping".
    beta = beta.clip(min=0)

    # u-clip. Makes the response function shaped like a big U, centered on the MMT.
    beta = uclip(
        beta.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
        dim="tas_bin",
        idx_min=mmt_idx,
    )

    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign(beta=beta)


mortality_noadapt_impact_model = Projector(
    preprocess=_noadapt_beta_from_gamma,
    project=_mortality_rebased_impact_model,
    postprocess=_no_processing,
)
