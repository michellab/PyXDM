# PyXDM

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![PyXDM Logo](dev/pyxdm_logo.png)

PyXDM is a Python package for calculating XDM (Exchange-hole Dipole Moment) moments using atoms-in-molecules (AIM) partitioning schemes.

## Features

- **XDM multipole moments**: dipole, quadrupole, octupole.
- **AIM schemes**: Becke, Hirshfeld, Hirshfeld-I, Iterative Stockholder, MBIS.


## Installation

Create a conda environment with all the required dependencies:

```
conda env create -f environment.yaml
conda activate pyxdm
```

Install `pyxdm` in interactive mode within the activated environment:

```bash
pip install -e .
```

For developers:

```bash
pip install -e .[dev]
# Or install development dependencies separately:
pip install -r dev/requirements-dev.txt
```

## Usage

### Command Line Interface

After installation, use the `pyxdm` command:

```bash
pyxdm <wfn_file> [optional]
```

#### Arguments

##### Positional
- `<wfn_file>`: Path to the wavefunction file (Molden, WFN, etc.)

##### Optional
- `--mesh <file>`: Optional postg mesh file for integration grid
- `--scheme <schemes>`: Partitioning scheme(s) to use, separated by comma. If not specified, all available schemes are calculated. Available schemes: `becke`, `hirshfeld`, `hirshfeld_i`, `is`, `mbis`
- `--proatomdb <path>`: Path to proatom database file (required for Hirshfeld schemes)
- `--grid <quality>`: Grid quality for numerical integration. Options: `coarse`, `medium`, `fine`, `veryfine`, `ultrafine`, `insane` (default: `fine`)
- `--xdm-moments <orders>`: Order(s) of multipole moments to calculate (default: `1 2 3`)
- `--radial-moments <orders>`: Order(s) of radial moments to calculate (default: none)
- `--aniso`: Calculate anisotropic multipole moments (default: False)
- `--output <file>`, `-o <file>`: Output HDF5 file to save calculated data (default: `<input_basename>.h5`)

#### Examples

Basic usage with MBIS scheme:
```bash
pyxdm orca.molden.input --scheme mbis
```

Calculate multiple schemes with custom grid granularity:
```bash
pyxdm orca.molden.input --scheme mbis,becke --grid ultrafine
```

### Python API

You can also use PyXDM as a library in your own Python scripts:

```python
from pyxdm.core import XDMSession

session = XDMSession('examples/water/orca.molden.input')
session.load_molecule()
session.setup_grid()
session.setup_calculator()
session.setup_partition_schemes(['mbis'])
xdm_results = session.calculator.calculate_xdm_moments(
    partition_obj=session.partitions['mbis'],
    grid=session.grid,
    order=[1, 2, 3],
    anisotropic=False,
)
```

## Acknowledgments

Some implementations in this package, such as the Newton-Raphson algorithm for the Becke-Roussel (BR) exchange model, are based on [postg](https://github.com/aoterodelaroza/postg). As a state-of-the-art package for XDM calculations, postg serves as an essential reference for accuracy and methodology. The aim of this package is mainly to provide easy-to-use access to a broader range of partitioning schemes beyond Hirshfeld.
