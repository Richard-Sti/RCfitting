# Copyright (C) 2023 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import matplotlib.pyplot as plt
import numpy as np
from jax import grad, hessian, jit
from jax import numpy as jnp
from jax.scipy import stats as jstats
from jax.scipy.special import gamma, gammaincc
from scipy.optimize import minimize
import numdifftools as nd

from warnings import warn

from tqdm import tqdm
from joblib import Parallel, delayed

GNEWTON = 4.30091727e-06    # km^2 kpc Msun^(-1) s^-2
RHO200C = 200 * 277.54      # Msun / kpc^3  (h = 1)


###############################################################################
#                                 RC model                                    #
###############################################################################


def parse_galaxy(name, fgals, Ups_bul_mean=0.7, Ups_disk_mean=0.5,
                 Ups_gas_mean=1.0, log_Ups_bul_std=0.1, log_Ups_disk_std=0.1,
                 log_Ups_gas_std=0.04, log_M200c_bounds=(5.0, 17.0),
                 log_conc_bounds=(-3, 3), einasto_alpha_bounds=(-5, 5),
                 log_Vflat_arctan_bounds=(-1, 3),
                 log_rturn_arctan_bounds=(-1, 3), log_a0_bounds=(-2, 2),
                 a0_mean=1.19, a0_std=0.034, prior_nstd=50,):
    """
    Parse the data for a single galaxy.

    Parameters
    ----------
    name : str
        Name of the galaxy. Must be a key in `fgals`.
    fgals : h5py.File
        HDF5 file containing the data for all galaxies.
    Ups_bul_mean : float, optional
        Mean of the log bulge mass-to-light ratio prior.
    Ups_disk_mean : float, optional
        Mean of the log disk mass-to-light ratio prior.
    Ups_gas_mean : float, optional
        Mean of the log gas mass-to-light ratio prior.
    log_Ups_bul_std : float, optional
        Standard deviation of the log bulge mass-to-light ratio prior.
    log_Ups_disk_std : float, optional
        Standard deviation of the log disk mass-to-light ratio prior.
    log_Ups_gas_std : float, optional
        Standard deviation of the log gas mass-to-light ratio prior.
    log_M200c_bounds : tuple, optional
        Bounds of log M200c prior in log10(Msun).
    log_conc_bounds : tuple, optional
        Bounds of log concentration prior.
    einasto_alpha_bounds : tuple, optional
        Bounds of Einasto index prior.
    log_Vflat_arctan_bounds : tuple, optional
        Bounds of log Vflat prior in km/s for the arctan profile.
    log_rturn_arctan_bounds : tuple, optional
        Bounds of log rturn prior in kpc for the arctan profile.
    log_a0_bounds : tuple, optional
        Bounds of log a0 prior in 1e-10 m / s^2 for the MOND RAR IF.
    a0_mean : float, optional
        Mean of the a_0 prior in 1e-10 m / s^2 for the MOND RAR IF.
    a0_std : float, optional
        Standard deviation of the a_0 prior in 1e-10 m / s^2 for the MOND
        RAR IF.
    prior_nstd : float, optional
        Number of standard deviations to use for the prior bounds.

    Returns
    -------
    data : dict
        Dictionary containing the parsed galaxy data.
    """
    if name not in fgals:
        raise ValueError(f"Galaxy {name} not found in `fgals`.")

    data = {}
    data["galaxy_name"] = {name: 0}
    for p in ("r", "Vbul", "Vgas", "Vdisk", "Vobs", "e_Vobs", "L36"):
        data[p] = fgals[name][p][:].astype(np.float64)

    for p in ["inc", "dist"]:
        data[p] = fgals[name][p][0]
        data[f"e_{p}"] = fgals[name][f"e_{p}"][0]

    data["inc"] = np.deg2rad(data["inc"])
    data["e_inc"] = np.deg2rad(data["e_inc"])

    data["Mstar"] = 1e9 * fgals[name]["L36"][0]
    data["e_Mstar"] = 1e9 * fgals[name]["e_L36"][0]
    data["MHI"] = 1e9 * fgals[name]["MHI"][0]

    # Precompute the products of the velocities
    for p in ("bul", "disk", "gas"):
        data[f"V{p}_product"] = data[f"V{p}"] * np.abs(data[f"V{p}"])

    data["e_Vobs_squared"] = data["e_Vobs"]**2

    data["log_Ups_bul_mean"] = np.log10(Ups_bul_mean)
    data["log_Ups_disk_mean"] = np.log10(Ups_disk_mean)
    data["log_Ups_gas_mean"] = np.log10(Ups_gas_mean)

    data["log_Ups_bul_std"] = log_Ups_bul_std
    data["log_Ups_disk_std"] = log_Ups_disk_std
    data["log_Ups_gas_std"] = log_Ups_gas_std

    # NFW halo
    data["NFW_log_M200c_bounds"] = log_M200c_bounds
    data["NFW_log_conc_bounds"] = log_conc_bounds
    data["NFW_params"] = {"log_Ups_bul": 0,
                          "log_Ups_disk": 1,
                          "log_Ups_gas": 2,
                          "inc": 3,
                          "dist": 4,
                          "log_M200c": 5,
                          "log_concentration": 6,
                          }

    # Isothermal sphere halo
    data["isothermal_log_M200c_bounds"] = log_M200c_bounds
    data["isothermal_log_conc_bounds"] = log_conc_bounds
    data["isothermal_params"] = {"log_Ups_bul": 0,
                                 "log_Ups_disk": 1,
                                 "log_Ups_gas": 2,
                                 "inc": 3,
                                 "dist": 4,
                                 "log_M200c": 5,
                                 "log_concentration": 6,
                                 }

    # Einasto halo
    data["Einasto_log_M200c_bounds"] = log_M200c_bounds
    data["Einasto_log_conc_bounds"] = log_conc_bounds
    data["Einasto_alpha_bounds"] = einasto_alpha_bounds
    data["Einasto_params"] = {"log_Ups_bul": 0,
                              "log_Ups_disk": 1,
                              "log_Ups_gas": 2,
                              "inc": 3,
                              "dist": 4,
                              "log_M200c": 5,
                              "log_concentration": 6,
                              "alpha": 7,
                              }

    # Arctan profile
    data["log_Vflat_arctan_bounds"] = log_Vflat_arctan_bounds
    data["log_rturn_arctan_bounds"] = log_rturn_arctan_bounds
    data["arctan_params"] = {"inc": 0,
                             "dist": 1,
                             "log_Vflat": 2,
                             "log_rturn": 3,
                             }

    # MOND RAR IF
    data["log_a0_bounds"] = log_a0_bounds
    data["a0_mean"] = a0_mean
    data["a0_std"] = a0_std
    data["RARIF_params"] = {"log_Ups_bul": 0,
                            "log_Ups_disk": 1,
                            "log_Ups_gas": 2,
                            "inc": 3,
                            "dist": 4,
                            "log_a0": 5,
                            }

    data["inc_bounds"] = [np.deg2rad(30), np.deg2rad(150)]
    data["prior_nstd"] = prior_nstd

    data["h"] = 0.7

    return data


def Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist, parsed_galaxy):
    """
    Calculate the squared baryonic circular velocity.

    Parameters
    ----------
    log_Ups_bul : float
        Log bulge mass-to-light ratio.
    log_Ups_disk : float
        Log disk mass-to-light ratio.
    log_Ups_gas : float
        Log gas mass-to-light ratio.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vbar2 : float
        Squared baryonic circular velocity in km^2/s^2.
    """
    Vbar2 = (10**log_Ups_bul * parsed_galaxy["Vbul_product"]
             + 10**log_Ups_disk * parsed_galaxy["Vdisk_product"]
             + 10**log_Ups_gas * parsed_galaxy["Vgas_product"])

    Vbar2 *= dist / parsed_galaxy["dist"]
    return Vbar2


def Vobs_scaled(inc, parsed_galaxy):
    Vobs_true = jnp.copy(parsed_galaxy["Vobs"])
    Vobs_true *= (jnp.sin(parsed_galaxy["inc"]) / jnp.sin(inc))

    e_Vobs = parsed_galaxy["e_Vobs"]

    return Vobs_true, e_Vobs


def initial_galaxy_params(parsed_galaxy, seed=None):
    gen = np.random.RandomState(seed)

    x0 = {}
    for p in ["bul", "disk", "gas"]:
        x0[f"log_Ups_{p}"] = gen.normal(
            loc=parsed_galaxy[f"log_Ups_{p}_mean"],
            scale=parsed_galaxy[f"log_Ups_{p}_std"])

    inc = 0
    nstd = parsed_galaxy["prior_nstd"]
    mu, std = parsed_galaxy["inc"], parsed_galaxy["e_inc"]
    inc_min = max(parsed_galaxy["inc_bounds"][0], mu - nstd * std)
    inc_max = min(parsed_galaxy["inc_bounds"][1], mu + nstd * std)
    while not (inc_min < inc < inc_max):
        inc = gen.normal(loc=mu, scale=std)
    x0["inc"] = inc

    dist = -1
    while not dist > 0:
        dist = gen.normal(loc=parsed_galaxy["dist"],
                          scale=parsed_galaxy["e_dist"])
    x0["dist"] = dist

    return x0


def galaxy_bounds(parsed_galaxy):
    nstd = parsed_galaxy["prior_nstd"]

    bounds = {}
    for p in ["bul", "disk", "gas"]:
        mu = parsed_galaxy[f"log_Ups_{p}_mean"]
        std = parsed_galaxy[f"log_Ups_{p}_std"]
        bounds["log_Ups_" + p] = (mu - nstd * std, mu + nstd * std)

    bounds["inc"] = parsed_galaxy["inc_bounds"]

    mu, std = parsed_galaxy["dist"], parsed_galaxy["e_dist"]
    dmin = max(0, mu - nstd * std)
    dmax = mu + nstd * std
    bounds["dist"] = (dmin, dmax)

    return bounds


def log_prior_galaxy(parsed_galaxy, log_Ups_bul=None, log_Ups_disk=None,
                     log_Ups_gas=None, inc=None, dist=None):
    log_prior = 0.0

    if log_Ups_bul is not None:
        log_prior += jstats.norm.logpdf(
            log_Ups_bul, loc=parsed_galaxy["log_Ups_bul_mean"],
            scale=parsed_galaxy["log_Ups_bul_std"])

    if log_Ups_disk is not None:
        log_prior += jstats.norm.logpdf(
            log_Ups_disk, loc=parsed_galaxy["log_Ups_disk_mean"],
            scale=parsed_galaxy["log_Ups_disk_std"])

    if log_Ups_gas is not None:
        log_prior += jstats.norm.logpdf(
            log_Ups_gas, loc=parsed_galaxy["log_Ups_gas_mean"],
            scale=parsed_galaxy["log_Ups_gas_std"])

    if inc is not None:
        log_prior += jstats.norm.logpdf(inc, loc=parsed_galaxy["inc"],
                                        scale=parsed_galaxy["e_inc"])

    if dist is not None:
        log_prior += jstats.norm.logpdf(dist, loc=parsed_galaxy["dist"],
                                        scale=parsed_galaxy["e_dist"])

    return log_prior


###############################################################################
#                                  Haloes                                     #
###############################################################################


def M200c2R200c(M200c, h=0.7):
    return (3 * M200c / (4 * np.pi * RHO200C * h**2))**(1./3)


###############################################################################
#                                NFW halo                                     #
###############################################################################


def log_M200c_to_mean_log_concentration_NFW(log_M200c, h=0.7):
    """
    Mean log concentration as a function of log M200c for a NFW halo from
    https://arxiv.org/abs/1402.7073,
    """
    return 0.905 - 0.101 * (log_M200c - 12. + jnp.log10(h))


def initial_params_NFW(parsed_galaxy, seed=None):
    gen = np.random.RandomState(seed)

    a0, a1 = parsed_galaxy["NFW_log_M200c_bounds"]
    log_M200c = gen.uniform(low=a0, high=a1)

    mu = log_M200c_to_mean_log_concentration_NFW(log_M200c,
                                                 h=parsed_galaxy["h"])
    log_concentration = gen.normal(mu, 0.11)

    return {"log_M200c": log_M200c,
            "log_concentration": log_concentration}


def param_bounds_NFW(parsed_galaxy):
    return {"log_M200c": parsed_galaxy["NFW_log_M200c_bounds"],
            "log_concentration": parsed_galaxy["NFW_log_conc_bounds"]}


def squared_circular_velocity_NFW(log_M200c, log_concentration, dist,
                                  parsed_galaxy):
    """
    Calculate the squared circular velocity for a NFW halo.

    Parameters
    ----------
    log_M200c : float
        Log M200c in Msun.
    log_concentration : float
        Log concentration.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vdm2 : float
        Squared circular velocity in km^2/s^2.
    """
    M200c, concentration = 10**log_M200c, 10**log_concentration

    R200c = M200c2R200c(M200c, h=parsed_galaxy["h"])
    Rs = R200c / concentration
    radius = parsed_galaxy["r"] * (dist / parsed_galaxy["dist"])

    return ((GNEWTON * M200c) / radius) * (jnp.log(1. + radius / Rs) - radius / (radius + Rs)) / (jnp.log(1. + concentration) - concentration / (1. + concentration))  # noqa


def Vobs_NFW(log_M200c, log_concentration, log_Ups_bul, log_Ups_disk,
             log_Ups_gas, dist, parsed_galaxy):
    """
    Model the observed circular velocity for a NFW halo.

    Parameters
    ----------
    log_M200c : float
        Log M200c in Msun.
    log_concentration : float
        Log concentration.
    log_Ups_bul : float
        Log bulge mass-to-light ratio.
    log_Ups_disk : float
        Log disk mass-to-light ratio.
    log_Ups_gas : float
        Log gas mass-to-light ratio.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vobs : float
        Observed circular velocity in km/s.
    """
    Vbar2 = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                         parsed_galaxy)
    Vdm2 = squared_circular_velocity_NFW(log_M200c, log_concentration, dist,
                                         parsed_galaxy)

    return jnp.sqrt(Vbar2 + Vdm2)


def unpack_NFW_params(params, parsed_galaxy):
    params_map = parsed_galaxy["NFW_params"]

    log_Ups_bul = params[params_map["log_Ups_bul"]]
    log_Ups_disk = params[params_map["log_Ups_disk"]]
    log_Ups_gas = params[params_map["log_Ups_gas"]]
    inc = params[params_map["inc"]]
    dist = params[params_map["dist"]]

    log_M200c = params[params_map["log_M200c"]]
    log_concentration = params[params_map["log_concentration"]]

    return (log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c,
            log_concentration)


def ll_NFW(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc = unpack_NFW_params(params, parsed_galaxy)  # noqa

    Vobs_true, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
    Vobs_pred = Vobs_NFW(log_M200c, log_conc, log_Ups_bul, log_Ups_disk,
                         log_Ups_gas, dist, parsed_galaxy)

    return -0.5 * jnp.sum(jnp.square((Vobs_pred - Vobs_true) / e_Vobs))


def loss_NFW(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc = unpack_NFW_params(params, parsed_galaxy)  # noqa

    lp = 0.
    # Galaxy parameters prior
    lp += log_prior_galaxy(parsed_galaxy, log_Ups_bul, log_Ups_disk,
                           log_Ups_gas, inc, dist)

    # NFW parameters prior
    b0, b1 = parsed_galaxy["NFW_log_M200c_bounds"]
    lp += jstats.uniform.logpdf(log_M200c, b0, b1 - b0)

    mu = log_M200c_to_mean_log_concentration_NFW(log_M200c,
                                                 h=parsed_galaxy["h"])
    lp += jstats.norm.logpdf(log_conc, loc=mu, scale=0.11)

    ll = ll_NFW(params, parsed_galaxy)
    return - (ll + lp)


###############################################################################
#                       Isothermal sphere halo                                #
###############################################################################


def initial_params_isothermal(parsed_galaxy, seed=None):
    gen = np.random.RandomState(seed)

    a0, a1 = parsed_galaxy["NFW_log_M200c_bounds"]
    log_M200c = gen.uniform(low=a0, high=a1)

    # TODO does this make sense?
    mu = log_M200c_to_mean_log_concentration_NFW(log_M200c,
                                                 h=parsed_galaxy["h"])
    log_concentration = gen.normal(mu, 0.11)

    return {"log_M200c": log_M200c,
            "log_concentration": log_concentration}


def param_bounds_isothermal(parsed_galaxy):
    return {"log_M200c": parsed_galaxy["isothermal_log_M200c_bounds"],
            "log_concentration": parsed_galaxy["isothermal_log_conc_bounds"]}


def squared_circular_velocity_isothermal(log_M200c, log_concentration, dist,
                                         parsed_galaxy):
    """
    Calculate the squared circular velocity for an isothermal sphere halo.

    Parameters
    ----------
    log_M200c : float
        Log M200c in Msun.
    log_concentration : float
        Log concentration.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vdm2 : float
        Squared circular velocity in km^2/s^2.
    """
    M200c, concentration = 10**log_M200c, 10**log_concentration

    R200c = M200c2R200c(M200c, h=parsed_galaxy["h"])
    radius = (dist / parsed_galaxy["dist"]) * parsed_galaxy["r"]

    cr = concentration * radius
    return (GNEWTON * M200c / R200c) * (1 - R200c / cr * jnp.arctan(cr / R200c)) / (1 - jnp.arctan(concentration) / concentration)  # noqa


def Vobs_isothermal(log_M200c, log_concentration, log_Ups_bul, log_Ups_disk,
                    log_Ups_gas, dist, parsed_galaxy):
    """
    Model the observed circular velocity for an isothermal sphere halo.

    Parameters
    ----------
    log_M200c : float
        Log M200c in Msun.
    log_concentration : float
        Log concentration.
    log_Ups_bul : float
        Log bulge mass-to-light ratio.
    log_Ups_disk : float
        Log disk mass-to-light ratio.
    log_Ups_gas : float
        Log gas mass-to-light ratio.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vobs : float
        Observed circular velocity in km/s.
    """
    Vbar2 = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                         parsed_galaxy)
    Vdm2 = squared_circular_velocity_isothermal(log_M200c, log_concentration,
                                                dist, parsed_galaxy)
    return jnp.sqrt(Vbar2 + Vdm2)


def unpack_isothermal_params(params, parsed_galaxy):
    params_map = parsed_galaxy["isothermal_params"]

    log_Ups_bul = params[params_map["log_Ups_bul"]]
    log_Ups_disk = params[params_map["log_Ups_disk"]]
    log_Ups_gas = params[params_map["log_Ups_gas"]]
    inc = params[params_map["inc"]]
    dist = params[params_map["dist"]]

    log_M200c = params[params_map["log_M200c"]]
    log_concentration = params[params_map["log_concentration"]]

    return (log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c,
            log_concentration)


def ll_isothermal(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc = unpack_isothermal_params(params, parsed_galaxy)  # noqa

    Vobs, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
    Vobs_pred = Vobs_isothermal(log_M200c, log_conc, log_Ups_bul, log_Ups_disk,
                                log_Ups_gas, dist, parsed_galaxy)
    return -0.5 * jnp.sum(jnp.square((Vobs_pred - Vobs) / e_Vobs))


def loss_isothermal(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc = unpack_isothermal_params(params, parsed_galaxy)  # noqa

    lp = 0.
    # Galaxy parameters prior
    lp += log_prior_galaxy(parsed_galaxy, log_Ups_bul, log_Ups_disk,
                           log_Ups_gas, inc, dist)

    # NFW parameters prior
    b0, b1 = parsed_galaxy["isothermal_log_M200c_bounds"]
    lp += jstats.uniform.logpdf(log_M200c, b0, b1 - b0)

    ll = ll_isothermal(params, parsed_galaxy)
    return - (ll + lp)


###############################################################################
#                            Einasto halo                                     #
###############################################################################


def initial_params_Einasto(parsed_galaxy, seed=None):
    gen = np.random.RandomState(seed)

    a0, a1 = parsed_galaxy["Einasto_log_M200c_bounds"]
    log_M200c = gen.uniform(low=a0, high=a1)

    # TODO does this make sense?
    mu = log_M200c_to_mean_log_concentration_NFW(log_M200c,
                                                 h=parsed_galaxy["h"])
    log_concentration = gen.normal(mu, 0.11)
    alpha = gen.normal(0.2, 0.1)

    return {"log_M200c": log_M200c,
            "log_concentration": log_concentration,
            "alpha": alpha}


def param_bounds_Einasto(parsed_galaxy):
    return {"log_M200c": parsed_galaxy["Einasto_log_M200c_bounds"],
            "log_concentration": parsed_galaxy["Einasto_log_conc_bounds"],
            "alpha": parsed_galaxy["Einasto_alpha_bounds"]}


def incomplete_gamma(a, z):
    return gamma(a) * gammaincc(a, z)


def squared_circular_velocity_Einasto(log_M200c, log_concentration, alpha,
                                      dist, parsed_galaxy):
    """
    Calculate the squared circular velocity for an Einasto halo.

    Parameters
    ----------
    log_M200c : float
        Log M200c in Msun.
    log_concentration : float
        Log concentration.
    alpha : float
        Einasto index.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vdm2 : float
        Squared circular velocity in km^2/s^2.
    """
    M200c, concentration = 10**log_M200c, 10**log_concentration

    R200c = M200c2R200c(M200c, h=parsed_galaxy["h"])
    radius = parsed_galaxy["r"] * (dist / parsed_galaxy["dist"])

    return (GNEWTON * M200c / radius) * (incomplete_gamma(3. / alpha, 0) - incomplete_gamma(3. / alpha, 2. / alpha * (concentration * radius / R200c)**alpha)) / (incomplete_gamma(3. / alpha, 0) - incomplete_gamma(3. / alpha, 2. / alpha * concentration**alpha))  # noqa


def Vobs_Einasto(log_M200c, log_concentration, alpha, log_Ups_bul,
                 log_Ups_disk, log_Ups_gas, dist, parsed_galaxy):
    """
    Model the observed circular velocity for a NFW halo.

    Parameters
    ----------
    log_M200c : float
        Log M200c in Msun.
    log_concentration : float
        Log concentration.
    alpha : float
        Einasto index.
    log_Ups_bul : float
        Log bulge mass-to-light ratio.
    log_Ups_disk : float
        Log disk mass-to-light ratio.
    log_Ups_gas : float
        Log gas mass-to-light ratio.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vobs : float
        Observed circular velocity in km/s.
    """
    Vbar2 = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                         parsed_galaxy)
    Vdm2 = squared_circular_velocity_Einasto(log_M200c, log_concentration,
                                             alpha, dist, parsed_galaxy)
    return jnp.sqrt(Vbar2 + Vdm2)


def unpack_Einasto_params(params, parsed_galaxy):
    params_map = parsed_galaxy["Einasto_params"]

    log_Ups_bul = params[params_map["log_Ups_bul"]]
    log_Ups_disk = params[params_map["log_Ups_disk"]]
    log_Ups_gas = params[params_map["log_Ups_gas"]]
    inc = params[params_map["inc"]]
    dist = params[params_map["dist"]]

    log_M200c = params[params_map["log_M200c"]]
    log_concentration = params[params_map["log_concentration"]]
    alpha = params[params_map["alpha"]]

    return (log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c,
            log_concentration, alpha)


def ll_Einasto(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc, alpha = unpack_Einasto_params(params, parsed_galaxy)  # noqa

    Vobs_true, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
    Vobs_pred = Vobs_Einasto(log_M200c, log_conc, alpha, log_Ups_bul,
                             log_Ups_disk, log_Ups_gas, dist, parsed_galaxy)

    return -0.5 * jnp.sum(jnp.square((Vobs_pred - Vobs_true) / e_Vobs))


def loss_Einasto(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc, alpha = unpack_Einasto_params(params, parsed_galaxy)  # noqa

    lp = 0.
    # Galaxy parameters prior
    lp += log_prior_galaxy(parsed_galaxy, log_Ups_bul, log_Ups_disk,
                           log_Ups_gas, inc, dist)

    b0, b1 = parsed_galaxy["Einasto_log_M200c_bounds"]
    lp += jstats.uniform.logpdf(log_M200c, b0, b1 - b0)

    b0, b1 = parsed_galaxy["Einasto_log_conc_bounds"]
    lp += jstats.uniform.logpdf(log_conc, b0, b1 - b0)

    b0, b1 = parsed_galaxy["Einasto_alpha_bounds"]
    lp += jstats.uniform.logpdf(alpha, b0, b1 - b0)

    ll = ll_Einasto(params, parsed_galaxy)
    return - (ll + lp)


###############################################################################
#                            Arctan profile                                   #
###############################################################################


def param_bounds_arctan(parsed_galaxy):
    return {"log_Vflat": parsed_galaxy["log_Vflat_arctan_bounds"],
            "log_rturn": parsed_galaxy["log_rturn_arctan_bounds"]}


def initial_params_arctan(parsed_galaxy, seed=None):
    gen = np.random.RandomState(seed)

    a0, a1 = parsed_galaxy["log_Vflat_arctan_bounds"]
    b0, b1 = parsed_galaxy["log_rturn_arctan_bounds"]

    return {"log_Vflat": gen.uniform(low=a0, high=a1),
            "log_rturn": gen.uniform(low=b0, high=b1)}


def unpack_params_arctan(params, parsed_galaxy):
    params_map = parsed_galaxy["arctan_params"]

    log_Vflat = params[params_map["log_Vflat"]]
    log_rturn = params[params_map["log_rturn"]]
    inc = params[params_map["inc"]]
    dist = params[params_map["dist"]]

    return inc, dist, log_Vflat, log_rturn


def Vobs_arctan(log_Vflat, log_rturn, dist, parsed_galaxy):
    """
    Model the observed circular velocity for an arctan profile.

    Parameters
    ----------
    log_Vflat : float
        Log Vflat in km/s.
    log_rturn : float
        Log rturn in kpc.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vobs : float
        Observed circular velocity in km/s.
    """
    Vflat = 10**log_Vflat
    rturn = 10**log_rturn

    radius = (dist / parsed_galaxy["dist"]) * parsed_galaxy["r"]

    return 2 / np.pi * Vflat * jnp.arctan(radius / rturn)


def ll_arctan(params, parsed_galaxy):
    inc, dist, log_Vflat, log_rturn = unpack_params_arctan(params, parsed_galaxy)  # noqa
    Vobs_true, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
    Vobs_pred = Vobs_arctan(log_Vflat, log_rturn, dist, parsed_galaxy)
    return -0.5 * jnp.sum(jnp.square((Vobs_pred - Vobs_true) / e_Vobs))


def loss_arctan(params, parsed_galaxy):
    inc, dist, log_Vflat, log_rturn = unpack_params_arctan(params, parsed_galaxy)  # noqa

    lp = log_prior_galaxy(parsed_galaxy, inc=inc, dist=dist)

    a0, a1 = parsed_galaxy["log_Vflat_arctan_bounds"]
    lp += jstats.uniform.logpdf(log_Vflat, a0, a1 - a0)

    b0, b1 = parsed_galaxy["log_rturn_arctan_bounds"]
    lp += jstats.uniform.logpdf(log_rturn, b0, b1 - b0)

    ll = ll_arctan(params, parsed_galaxy)
    return -(ll + lp)


###############################################################################
#                                  RAR IF                                     #
###############################################################################


def param_bounds_RARIF(parsed_galaxy):
    return {"log_a0": parsed_galaxy["log_a0_bounds"]}


def initial_params_RARIF(parsed_galaxy, seed=None):
    gen = np.random.RandomState(seed)
    mu, std = parsed_galaxy["a0_mean"], parsed_galaxy["a0_std"]
    return {"log_a0": np.log10(gen.normal(loc=mu, scale=std))}


def unpack_params_RARIF(params, parsed_galaxy):
    params_map = parsed_galaxy["RARIF_params"]

    log_Ups_bul = params[params_map["log_Ups_bul"]]
    log_Ups_disk = params[params_map["log_Ups_disk"]]
    log_Ups_gas = params[params_map["log_Ups_gas"]]
    inc = params[params_map["inc"]]
    dist = params[params_map["dist"]]

    log_a0 = params[params_map["log_a0"]]

    return log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_a0


def Vobs_RARIF(log_a0, log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
               parsed_galaxy):
    """
    Model the observed circular velocity in MOND using the RAR IF.

    Parameters
    ----------
    log_a0 : float
        Log a0 in 1e-10 m / s^2.
    log_Ups_bul : float
        Log bulge mass-to-light ratio.
    log_Ups_disk : float
        Log disk mass-to-light ratio.
    log_Ups_gas : float
        Log gas mass-to-light ratio.
    dist : float
        Distance to the galaxy in Mpc.
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    Vobs : float
        Observed circular velocity in km/s.
    """
    # gbar is distance independent, so we pass the SPARC distance
    Vbar2 = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas,
                         parsed_galaxy["dist"], parsed_galaxy)
    # Calculate gbar in 1e-10 m / s^2
    gbar = Vbar2 / parsed_galaxy["r"] * 0.00032407792894443654

    # Calculate gobs in 1e-10 m / s^2
    gobs = gbar / (1 - jnp.exp(-jnp.sqrt(gbar / 10**log_a0)))

    rdist = parsed_galaxy["r"] * (dist / parsed_galaxy["dist"])
    return jnp.sqrt(gobs * rdist) * 55.54887561  # Convert to km/s


def ll_RARIF(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_a0 = unpack_params_RARIF(params, parsed_galaxy)  # noqa

    Vobs_true, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
    Vobs = Vobs_RARIF(log_a0, log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                      parsed_galaxy)
    return -0.5 * jnp.sum(jnp.square((Vobs - Vobs_true) / e_Vobs))


def loss_RARIF(params, parsed_galaxy):
    log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_a0 = unpack_params_RARIF(params, parsed_galaxy)  # noqa

    lp = 0.
    lp += log_prior_galaxy(parsed_galaxy, log_Ups_bul, log_Ups_disk,
                           log_Ups_gas, inc, dist)

    b0, b1 = parsed_galaxy["log_a0_bounds"]
    lp += jstats.uniform.logpdf(log_a0, b0, b1 - b0)
    # lp += jstats.norm.logpdf(10**log_a0, loc=parsed_galaxy["a0_mean"],
    #                          scale=parsed_galaxy["a0_std"])

    ll = ll_RARIF(params, parsed_galaxy)
    return - (ll + lp)


###############################################################################
#                              Optimizer                                      #
###############################################################################


def initial_params_generator(kind, data, seed=None):
    """
    Generate initial parameters for the optimizer.

    Parameters
    ----------
    kind : str
        Kind of halo profile/dynamics to use. Must be one of "NFW",
        "isothermal" or "arctan".
    data : dict
        Dictionary containing the parsed galaxy data.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    x0 : 1-dimensional array
    """
    if kind == "NFW":
        params_map = data["NFW_params"]
        x0_ = {**initial_params_NFW(data, seed=seed),
               **initial_galaxy_params(data, seed=seed)}
    elif kind == "isothermal":
        params_map = data["isothermal_params"]
        x0_ = {**initial_params_isothermal(data, seed=seed),
               **initial_galaxy_params(data, seed=seed)}
    elif kind == "Einasto":
        params_map = data["Einasto_params"]
        x0_ = {**initial_params_Einasto(data, seed=seed),
                **initial_galaxy_params(data, seed=seed)}
    elif kind == "arctan":
        params_map = data["arctan_params"]
        x0_ = {**initial_params_arctan(data, seed=seed),
               **initial_galaxy_params(data, seed=seed)}
    elif kind == "RARIF":
        params_map = data["RARIF_params"]
        x0_ = {**initial_params_RARIF(data, seed=seed),
               **initial_galaxy_params(data, seed=seed)}
    else:
        raise ValueError(f"Unknown kind `{kind}`.")

    x0 = np.empty(len(params_map))
    for key, i in params_map.items():
        x0[i] = x0_[key]

    return x0


def bounds_generator(kind, data):
    """
    Generate bounds for the optimizer.

    Parameters
    ----------
    kind : str
        Kind of halo profile/dynamics to use. Must be one of "NFW",
        "isothermal" or "arctan".
    data : dict
        Dictionary containing the parsed galaxy data.

    Returns
    -------
    bounds : list of tuples
    """
    if kind == "NFW":
        params_map = data["NFW_params"]
        bounds_ = {**param_bounds_NFW(data),
                   **galaxy_bounds(data)}
    elif kind == "isothermal":
        params_map = data["isothermal_params"]
        bounds_ = {**param_bounds_isothermal(data),
                   **galaxy_bounds(data)}
    elif kind == "Einasto":
        params_map = data["Einasto_params"]
        bounds_ = {**param_bounds_Einasto(data),
                   **galaxy_bounds(data)}
    elif kind == "arctan":
        params_map = data["arctan_params"]
        bounds_ = {**param_bounds_arctan(data),
                   **galaxy_bounds(data)}
    elif kind == "RARIF":
        params_map = data["RARIF_params"]
        bounds_ = {**param_bounds_RARIF(data),
                   **galaxy_bounds(data)}
    else:
        raise ValueError(f"Unknown kind {kind}.")

    bounds = [(None, None)] * len(params_map)

    for key, i in params_map.items():
        bounds[i] = bounds_[key]

    return bounds


def loss_negll(kind):
    if kind == "NFW":
        return loss_NFW, ll_NFW
    elif kind == "isothermal":
        return loss_isothermal, ll_isothermal
    elif kind == "Einasto":
        return loss_Einasto, ll_Einasto
    elif kind == "arctan":
        return loss_arctan, ll_arctan
    elif kind == "RARIF":
        return loss_RARIF, ll_RARIF
    else:
        raise ValueError(f"Unknown kind {kind}.")


def minimize_single(kind, parsed_galaxy, method="L-BFGS-B", nconv=10,
                    nrepeat=100, tol=1e-12, conv_rtol=1e-5,
                    options={'maxiter': 50000}, seed=None):
    """
    Minimize the loss function for a single galaxy.

    Parameters
    ----------
    kind : str
        Kind of halo profile/dynamics to use. Must be one of "NFW",
        "isothermal" or "arctan".
    parsed_galaxy : dict
        Dictionary containing the parsed galaxy data.
    method : str, optional
        Optimization method to use. See `scipy.optimize.minimize` for more
        info.
    nconv : int, optional
        Number of consecutive iterations with the same loss value to consider
        a fit converged.
    nrepeat : int, optional
        Number of times to repeat the optimization.
    tol : float, optional
        Tolerance for the optimization.
    conv_rtol : float, optional
        Relative tolerance for the convergence criterion.
    options : dict, optional
        Options to pass to `scipy.optimize.minimize`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    res : dict
        Dictionary containing the results of the optimization.
    """
    loss, ll = loss_negll(kind)

    bounds = bounds_generator(kind, parsed_galaxy)

    jit_loss = jit(loss)
    grad_loss = jit(grad(loss))

    seed = np.random.randint(0, 2**32 - 1) if seed is None else seed

    nparams = len(bounds)
    fval = np.full(nrepeat, np.nan)
    x_min, dx_min = np.full(nparams, np.nan), np.full(nparams, np.nan)
    fval_min, success = np.inf, False
    ind2pname = {i: pname
                 for pname, i in parsed_galaxy[f"{kind}_params"].items()}

    for n in range(nrepeat):
        x0 = initial_params_generator(kind, parsed_galaxy, seed=seed + n)
        res = minimize(jit_loss, x0=x0, method=method, args=(parsed_galaxy,),
                       tol=tol, jac=grad_loss, bounds=bounds, options=options)

        # Check if the optimizer is hitting bounds
        for i in range(nparams):
            x = res.x[i]
            b0, b1 = bounds[i]
            if np.isclose(x, b0) or np.isclose(x, b1):
                pname = ind2pname[i]
                warn(f"Optimizer hit bound for parameter {pname} = {x}.",
                     RuntimeWarning)
                res.success = False
                break

        if not res.success:
            continue

        fval[n] = res.fun

        if fval[n] < fval_min:
            fval_min = res.fun
            x_min = res.x

        if nconv <= np.sum(np.isclose(fval_min, fval, rtol=conv_rtol)):
            success = True
            break

    # Calculate error from the Hessian
    if success:
        if kind != "Einasto":
            dx_min = np.diag(
                np.linalg.inv(hessian(loss)(x_min, parsed_galaxy)))
        else:
            dx_min = np.diag(
                np.linalg.inv(nd.Hessian(loss)(x_min, parsed_galaxy)))

        dx_min.flags.writeable = True
        ll_min = float(ll(x_min, parsed_galaxy))

        if np.any(dx_min < 0):
            dx_min *= np.nan
        else:
            dx_min = np.sqrt(dx_min)
    else:
        dx_min = np.full(nparams, np.nan)
        ll_min = np.nan

    chi2 = -2 * ll_min
    nobservations = parsed_galaxy["r"].size

    if nobservations > nparams:
        reduced_chi2 = chi2 / (nobservations - nparams)
    else:
        reduced_chi2 = np.nan

    return {
        "x_min": x_min,
        "dx_min": dx_min,
        "loss_min": fval_min,
        "ll_min": ll_min,
        "BIC": nparams * np.log(nobservations) + chi2,
        "AIC": 2 * nparams + chi2,
        "reduced_chi2": reduced_chi2,
        "success": success,
        "nparams": nparams,
        "nobservations": nobservations,
        "nrepeat": n + 1
        }


def minimize_many(kind, parsed_galaxies, n_jobs=-1, method="L-BFGS-B",
                  nconv=10, nrepeat=100, tol=1e-12, conv_rtol=1e-5,
                  options={'maxiter': 50000}, seed=None):
    """
    Minimize the loss function for a list of galaxies.

    Parameters
    ----------
    kind : str
        Kind of halo profile/dynamics to use. Must be one of "NFW",
        "isothermal" or "arctan".
    parsed_galaxies : dict
        List of dictionaries containing the parsed galaxy data.
    method : str, optional
        Optimization method to use. See `scipy.optimize.minimize` for more
        info.
    nconv : int, optional
        Number of consecutive iterations with the same loss value to consider
        a fit converged.
    nrepeat : int, optional
        Number of times to repeat the optimization.
    tol : float, optional
        Tolerance for the optimization.
    conv_rtol : float, optional
        Relative tolerance for the convergence criterion.
    options : dict, optional
        Options to pass to `scipy.optimize.minimize`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    res : dict
        List of dictionaries containing the results of the optimization from
        `minimize_single`.
    """
    return Parallel(n_jobs=n_jobs)(delayed(minimize_single)(
        kind, parsed_galaxy, method=method, nconv=nconv, nrepeat=nrepeat,
        tol=tol, conv_rtol=conv_rtol, options=options, seed=seed)
                              for parsed_galaxy in tqdm(parsed_galaxies,
                                                        desc="Fitting RCs"))


###############################################################################
#                             Plotting                                        #
###############################################################################


def plot_fit(res, kind, parsed_galaxy, fpath=None):
    r0 = parsed_galaxy["r"]

    plt.figure()
    plt.title(f"{list(parsed_galaxy['galaxy_name'].keys())[0]} ({kind} fit)")
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if kind == "NFW":
        log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc = unpack_NFW_params(res["x_min"], parsed_galaxy)  # noqa

        rnew = r0 * (dist / parsed_galaxy["dist"])
        Vobs, e_Vobs = Vobs_scaled(inc, parsed_galaxy)

        plt.errorbar(rnew, Vobs, yerr=e_Vobs, capsize=3,
                     label=r"$V_{\rm obs}$", color=cols[0])

        pred_Vobs = Vobs_NFW(log_M200c, log_conc, log_Ups_bul, log_Ups_disk,
                             log_Ups_gas, dist, parsed_galaxy)
        plt.plot(rnew, pred_Vobs, label=r"$V_{\rm pred}$", color=cols[1])

        Vbar = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                            parsed_galaxy)**0.5
        plt.plot(rnew, Vbar, label=r"$V_{\rm bar}$", color=cols[2])

        # Plot the individual components
        eta = (dist / parsed_galaxy["dist"])**0.5
        plt.plot(rnew, 10**(0.5 * log_Ups_bul) * parsed_galaxy["Vbul"] * eta,
                 label=r"$V_{\rm bulge}$", c=cols[3])
        plt.plot(rnew, 0.7**0.5 * parsed_galaxy["Vbul"], ls="--", c=cols[3])

        plt.plot(rnew, 10**(0.5 * log_Ups_disk) * parsed_galaxy["Vdisk"] * eta,
                 label=r"$V_{\rm disk}$", c=cols[4])
        plt.plot(rnew, 0.5**0.5 * parsed_galaxy["Vdisk"], ls="--", c=cols[4])

        plt.plot(rnew, 10**(0.5 * log_Ups_gas) * parsed_galaxy["Vgas"] * eta,
                 c=cols[5], label=r"$V_{\rm gas}$")
        plt.plot(rnew, parsed_galaxy["Vgas"], ls="--", c=cols[5])

    elif kind == "Einasto":
        log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc, alpha = unpack_Einasto_params(res["x_min"], parsed_galaxy)  # noqa

        rnew = r0 * (dist / parsed_galaxy["dist"])
        Vobs, e_Vobs = Vobs_scaled(inc, parsed_galaxy)

        plt.errorbar(rnew, Vobs, yerr=e_Vobs, capsize=3,
                     label=r"$V_{\rm obs}$", color=cols[0])

        pred_Vobs = Vobs_Einasto(log_M200c, log_conc, alpha, log_Ups_bul,
                                 log_Ups_disk, log_Ups_gas, dist,
                                 parsed_galaxy)
        plt.plot(rnew, pred_Vobs, label=r"$V_{\rm pred}$", color=cols[1])

        Vbar = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                            parsed_galaxy)**0.5
        plt.plot(rnew, Vbar, label=r"$V_{\rm bar}$", color=cols[2])

        # Plot the individual components
        eta = (dist / parsed_galaxy["dist"])**0.5
        plt.plot(rnew, 10**(0.5 * log_Ups_bul) * parsed_galaxy["Vbul"] * eta,
                 label=r"$V_{\rm bulge}$", c=cols[3])
        plt.plot(rnew, 0.7**0.5 * parsed_galaxy["Vbul"], ls="--", c=cols[3])

        plt.plot(rnew, 10**(0.5 * log_Ups_disk) * parsed_galaxy["Vdisk"] * eta,
                 label=r"$V_{\rm disk}$", c=cols[4])
        plt.plot(rnew, 0.5**0.5 * parsed_galaxy["Vdisk"], ls="--", c=cols[4])

        plt.plot(rnew, 10**(0.5 * log_Ups_gas) * parsed_galaxy["Vgas"] * eta,
                 c=cols[5], label=r"$V_{\rm gas}$")
        plt.plot(rnew, parsed_galaxy["Vgas"], ls="--", c=cols[5])

    elif kind == "isothermal":
        log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_M200c, log_conc = unpack_isothermal_params(res["x_min"], parsed_galaxy)  # noqa

        rnew = r0 * (dist / parsed_galaxy["dist"])
        Vobs, e_Vobs = Vobs_scaled(inc, parsed_galaxy)

        plt.errorbar(rnew, Vobs, yerr=e_Vobs, capsize=3,
                     label=r"$V_{\rm obs}$")

        pred_Vobs = Vobs_isothermal(log_M200c, log_conc, log_Ups_bul,
                                    log_Ups_disk, log_Ups_gas, dist,
                                    parsed_galaxy)
        plt.plot(rnew, pred_Vobs, label=r"$V_{\rm pred}$")

        Vbar = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                            parsed_galaxy)**0.5
        plt.plot(rnew, Vbar, label=r"$V_{\rm bar}$")

        # Plot the individual components
        eta = (dist / parsed_galaxy["dist"])**0.5
        plt.plot(rnew, 10**(0.5 * log_Ups_bul) * parsed_galaxy["Vbul"] * eta,
                 label=r"$V_{\rm bulge}$", c=cols[3])
        plt.plot(rnew, 0.7**0.5 * parsed_galaxy["Vbul"], ls="--", c=cols[3])

        plt.plot(rnew, 10**(0.5 * log_Ups_disk) * parsed_galaxy["Vdisk"] * eta,
                 label=r"$V_{\rm disk}$", c=cols[4])
        plt.plot(rnew, 0.5**0.5 * parsed_galaxy["Vdisk"], ls="--", c=cols[4])

        plt.plot(rnew, 10**(0.5 * log_Ups_gas) * parsed_galaxy["Vgas"] * eta,
                 label=r"$V_{\rm gas}$", c=cols[5])
        plt.plot(rnew, parsed_galaxy["Vgas"], ls="--", c=cols[5])

    elif kind == "arctan":
        inc, dist, log_Vflat, log_rturn = unpack_params_arctan(res["x_min"], parsed_galaxy)  # noqa

        Vobs, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
        rnew = r0 * (dist / parsed_galaxy["dist"])
        plt.errorbar(rnew, Vobs, yerr=e_Vobs, capsize=3,
                     label=r"$V_{\rm obs}$")

        pred_Vobs = Vobs_arctan(log_Vflat, log_rturn, dist, parsed_galaxy)

        plt.plot(rnew, pred_Vobs, label=r"$V_{\rm pred}$")
    elif kind == "RARIF":
        log_Ups_bul, log_Ups_disk, log_Ups_gas, inc, dist, log_a0 = unpack_params_RARIF(res["x_min"], parsed_galaxy)  # noqa

        Vobs, e_Vobs = Vobs_scaled(inc, parsed_galaxy)
        rnew = r0 * (dist / parsed_galaxy["dist"])

        plt.errorbar(rnew, Vobs, yerr=e_Vobs, capsize=3,
                     label=r"$V_{\rm obs}$")

        pred_Vobs = Vobs_RARIF(log_a0, log_Ups_bul, log_Ups_disk, log_Ups_gas,
                               dist, parsed_galaxy)
        plt.plot(rnew, pred_Vobs, label=r"$V_{\rm pred}$")

        Vbar = Vbar_squared(log_Ups_bul, log_Ups_disk, log_Ups_gas, dist,
                            parsed_galaxy)**0.5
        plt.plot(rnew, Vbar, label=r"$V_{\rm bar}$")

        # Plot the individual components
        eta = (dist / parsed_galaxy["dist"])**0.5
        plt.plot(rnew, 10**(0.5 * log_Ups_bul) * parsed_galaxy["Vbul"] * eta,
                 label=r"$V_{\rm bulge}$", c=cols[3])
        plt.plot(rnew, 0.7**0.5 * parsed_galaxy["Vbul"], ls="--", c=cols[3])

        plt.plot(rnew, 10**(0.5 * log_Ups_disk) * parsed_galaxy["Vdisk"] * eta,
                 label=r"$V_{\rm disk}$", c=cols[4])
        plt.plot(rnew, 0.5**0.5 * parsed_galaxy["Vdisk"], ls="--", c=cols[4])

        plt.plot(rnew, 10**(0.5 * log_Ups_gas) * parsed_galaxy["Vgas"] * eta,
                 label=r"$V_{\rm gas}$", c=cols[5])
        plt.plot(rnew, parsed_galaxy["Vgas"], ls="--", c=cols[5])
    else:
        raise ValueError(f"Unknown kind {kind}.")

    plt.legend()
    plt.xlabel(r"$r ~ [\mathrm{kpc}]$")
    plt.ylabel(r"$V_{\rm circ} ~ [\mathrm{km} / \mathrm{s}]$")

    if fpath is not None:
        plt.savefig(fpath, bbox_inches="tight", dpi=450)

    plt.show()
