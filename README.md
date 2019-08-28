# Econometrics.jl

[![License: ISC - Permissive License](https://img.shields.io/badge/License-ISC-green.svg)](https://img.shields.io/github/license/Nosferican/Econometrics.jl)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![JuliaCon](https://submissions.juliacon.org/papers/446fde271579d85e0d4c691d54093dbb/status.svg)](https://submissions.juliacon.org/papers/446fde271579d85e0d4c691d54093dbb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3379185.svg)](https://doi.org/10.5281/zenodo.3379185)

[![Documentation: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nosferican.github.io/Econometrics.jl/stable)
[![Documentation: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nosferican.github.io/Econometrics.jl/dev)

[![Build Status](https://travis-ci.com/Nosferican/Econometrics.jl.svg?branch=master)](https://travis-ci.com/Nosferican/Econometrics.jl)
[![Code Coverage](https://codecov.io/gh/Nosferican/Econometrics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Nosferican/Econometrics.jl)

![GitHub commits since latest release](https://img.shields.io/github/commits-since/Nosferican/Econometrics.jl/v0.2.2)


This package uses continuous integration on Linux, OSX, and Windows (x64/x86).

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
