"""
Logic for mortality impact and damage projection.
"""

from muuttaa import Projector
import pandas as pd
import xarray as xr


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _mortality_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins * census tract age-spec populations
    _effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin") * ds["share"]

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)

    return xr.Dataset({"impact": impact, "_effect": _effect})


# If you already have beta.
mortality_impact_model = Projector(
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
mortality_impact_model_gamma = Projector(
    preprocess=_beta_from_gamma,  # Not sure this should actually be a preprocess but I'm lazy.
    project=_mortality_impact_model,
    postprocess=_no_processing,
)


def _mortality_valuation_model(ds: xr.Dataset) -> xr.Dataset:
    # Total damages are age-spec physical impacts (deaths/100k) * age-spec population * scale * vsl
    damages_total = ds["impact"] * ds["pop"] * ds["scale"] * ds["vsl"]

    # Damages per capita = total damages / population
    damages_pc = ds["impact"] * ds["scale"] * ds["vsl"]

    # Damages as share of average tract income = damages per capita / income per capita
    damages_pincome = damages_pc / ds["pci"]

    out = xr.Dataset(
        {
            "damages_total": damages_total,
            "damages_pc": damages_pc,
            "damages_pincome": damages_pincome,
        }
    )
    return out


mortality_valuation_model = Projector(
    preprocess=_no_processing,
    project=_mortality_valuation_model,
    postprocess=_no_processing,
)
