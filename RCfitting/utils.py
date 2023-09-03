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

import numpy as np


def extract_results(key, results, gals, kind):
    """
    Extract optimization results from the list of results from `minimize_many`.

    Parameters
    ----------
    key : str
        Key to extract from the results. Can be either one of the inferred
        parameters (or its uncertainty) or one of the optimization statistics.
    results : list of dict
        List of results from `minimize_many`.
    gals : list of dict
        List of galaxies.
    kind : str
        Dynamics model.

    Returns
    -------
    1-dimensional array
    """
    params_map = gals[0][f"{kind}_params"]
    error_keys = ["e_" + par for par in params_map.keys()]

    if key not in ["x_min", "dx_min"] and key in results[0].keys():
        return np.asanyarray([results[i][key] for i in range(len(results))])
    elif key in params_map:
        k = params_map[key]
        return np.asanyarray([results[i]["x_min"][k]
                              for i in range(len(results))])
    elif key in error_keys:
        k = params_map[key[2:]]
        return np.asanyarray([results[i]["dx_min"][k]
                              for i in range(len(results))])
    else:
        options = list(results[0].keys())
        options.remove("x_min")
        options.remove("dx_min")
        options += list(params_map.keys())
        options += error_keys

        raise KeyError(f"Invalid key `{key}`. Options are: `{options}`")


def extract_gals(key, gals):
    """
    Extract information about the galaxies from the list of galaxies.

    Parameters
    ----------
    key : str
        Key to extract from the galaxies.
    gals : list of dict
        List of galaxies.

    Returns
    -------
    n-dimensional array
    """
    if key not in gals[0].keys():
        raise KeyError(f"Invalid key `{key}`. "
                       f"Options are: `{list(gals[0].keys())}`")

    out = [gal[key] for gal in gals]

    if isinstance(out[0], np.ndarray) and out[0].size == 1:
        out = [x[0] for x in out]

    if not isinstance(out[0], np.ndarray):
        out = np.array(out)

    return out
