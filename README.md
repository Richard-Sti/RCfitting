# Galaxy Rotation Curve Fitting Library

This library provides a suite of tools for fitting rotation curves to galaxy data (tested on SPARC) using various halo profile/dynamics:
- NFW dark matter profile
- Isothermal dark matter profile
- Arctan model of rotational velocity
- MOND RAR IF model

## Features

- Support for multiple halo profiles/dynamics.
- Modular design for easy customization and expansion.
- Optimization accelerated with Just-In-Time (JIT) compilation.
- Parallel processing for batch optimization.

## Installation

```bash
git clone https://github.com/Richard-Sti/RCfitting.git
cd RCfitting
pip install -e .
```

The above commands will clone the repository and install the library in editable mode. This will allow you to make changes to the library code and use the modified version without having to reinstall it. If you don't want to make any changes to the library code, you can simply install it using `pip install .`.

Additionally, installation in a clean virtual environment is recommended. You can create a virtual environment using `python -m venv venv` and activate it using `source venv/bin/activate`.


## Usage

### Optimizing a Single Galaxy

```python
from RCfitting import minimize_single

# Prepare your galaxy data as per the RCfitting.parse_galaxy(...)
parsed_galaxy = {...}

result = minimize_single("NFW", parsed_galaxy)
print(result)
```

### Optimizing Multiple Galaxies

```python
from RCfitting import minimize_many

# Prepare your galaxy data list as per the mentioned format
parsed_galaxies = [{...}, {...}, ...]

results = minimize_many("NFW", parsed_galaxies)
for res in results:
    print(res)
```

## Contributing

We welcome contributions to this project. If you have a feature request or find a bug, please open an issue. If you'd like to contribute code, please fork the repository, make your changes, and open a pull request.

## License

This project is licensed under the GNU General Public License. See the LICENSE file for details. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.
