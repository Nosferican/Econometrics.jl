# Econometrics.jl

| **Documentation** | **Continous Integration** | **Metadata** | **Reference**      |
|:-----------------:|:-------------------------:|:------------:|:------------------:|
| [![][dsi]][dsu]   | [![][bsi]][bsu]           | [![][li]][lu]| [![][pubi]][pubu]  |
| [![][ddi]][ddu]   | [![][cci]][ccu]           | ![csi]       | [![][doii]][doiu]  |

[bsi]: https://github.com/Nosferican/Econometrics.jl/workflows/CI/badge.svg
[bsu]: https://github.com/Nosferican/Econometrics.jl/actions?workflow=CI
[cci]: https://codecov.io/gh/Nosferican/Econometrics.jl/branch/master/graph/badge.svg
[ccu]: https://codecov.io/gh/Nosferican/Econometrics.jl
[dsi]: https://img.shields.io/badge/docs-stable-blue?style=plastic
[dsu]: https://Nosferican.github.io/Econometrics.jl/stable/
[ddi]: https://img.shields.io/badge/docs-dev-blue?style=plastic
[ddu]: https://Nosferican.github.io/Econometrics.jl/dev/
[li]: https://img.shields.io/github/license/Nosferican/Econometrics.jl?style=plastic
[lu]: https://tldrlegal.com/license/-isc-license
[pubu]: https://doi.org/10.21105/jcon.00038
[pubi]: https://proceedings.juliacon.org/papers/10.21105/jcon.00038/status.svg
[doiu]: https://doi.org/10.5281/zenodo.3379185
[doii]: https://zenodo.org/badge/DOI/10.5281/zenodo.3379185.svg
[csi]: https://img.shields.io/github/commits-since/Nosferican/Econometrics.jl/v0.2.6

This package provides the functionality to estimate the following regression models:

- Continuous Response Models
  - Ordinary Least Squares (Stata's `reg`/`ivregress 2sls`)
  - Longitudinal estimators
    - Random effects model à la [Swamy Arora](https://dx.doi.org/10.2307/1909405) (Stata's `xtreg`/`xtivreg`)
    - Between estimator (Stata's `xtreg, be`)
- Nominal Response Model
  - Multinomial logistic (softmax) regression (Stata's `mlogit`)
- Ordinal Response Model
  - Proportional Odds Logistic Regression (Stata's `ologit`)

In addition, models incorporate the following features:
  - Implements the StatsBase.jl `StatisticalModel`/`RegressionModel` [API](http://juliastats.github.io/StatsBase.jl/latest/statmodels/)
  - Support for frequency weights
  - Robust Variance Covariance Estimators (e.g., heteroscedasticity consistent)
  - Instrumental Variables Model through Two-Stage Least Squares (2SLS)
  - Feature absorption for estimating a subset of parameters with high dimensional fixed effects as controls efficiently
