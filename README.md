# collider-check Package

<p align="center">
    <img src="https://img.shields.io/pypi/v/collider-check" alt="PyPI - Version">
    <img src="https://img.shields.io/github/commits-since/ColasDroin/collider_check/latest" alt="GitHub commits since latest release">
    <img src="https://img.shields.io/github/actions/workflow/status/ColasDroin/collider_check/tests.yaml?label=Tests" alt="GitHub Actions Workflow Status">
    <img src="https://img.shields.io/github/license/ColasDroin/collider_check" alt="GitHub License">    
    <img src="https://img.shields.io/pypi/pyversions/collider-check" alt="PyPI - Python Version">
    <img src="https://img.shields.io/badge/Built%20with%20uv-8A2BE2">
</p>

## Overview

A small package that provides functions to check the observables in a collider, using a collider built with [Xsuite](https://github.com/xsuite) as input (must be a json file). Very useful for debugging purposes before/after tracking, and used as a backend of the [simulation dashboard](https://github.com/ColasDroin/simulation-dashboard).

## Installation

The package is available from PyPI, so you can install it with pip:

```bash
pip install collider-check
```

## Usage

You can import the package in your python script and load a collider with:

```python
import collider_check
check = collider_check.from_json("path/to/collider.json")
```

To get a quick summary of the collider observables, use:

```python
check.output_check_as_str('output.txt')
```

## License

This package is distributed under the MIT License. See [LICENSE](LICENSE) for more information.
