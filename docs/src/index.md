# Econometrics.jl

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

**Citation**

Please use the following reference when citing this package:

[![Publication](https://proceedings.juliacon.org/papers/10.21105/jcon.00038/status.svg)](https://doi.org/10.21105/jcon.00038)

```
@article{SantiagoCalderón2020,
  author = {Santiago Calderón, José Bayoán},
  title = {Econometrics.jl},
  journal = {Proceedings of the JuliaCon Conferences},
  publisher = {The Open Journal},
  year = {2020},
  volume = {1},
  number = {1},
  pages = {38},
  doi = {10.21105/jcon.00038}
}
```
