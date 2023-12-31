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

from .main import (Vbar_squared, parse_galaxy,                                                  # noqa
                   squared_circular_velocity_NFW, Vobs_NFW,                                     # noqa
                   squared_circular_velocity_isothermal, Vobs_isothermal,                       # noqa
                   squared_circular_velocity_Einasto, Vobs_Einasto,                             # noqa
                   Vobs_arctan,                                                                 # noqa
                   Vobs_RARIF,                                                                  # noqa
                   minimize_single, minimize_many, initial_params_generator, bounds_generator,  # noqa
                   plot_fit                                                                     # noqa
                   )

from .utils import extract_results, extract_gals                                                # noqa

from jax import config
config.update("jax_enable_x64", True)
