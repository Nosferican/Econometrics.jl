# Interacting with a model

The API allows for:

- build/estimate models
- query models for statistics (e.g., goodness-of-fit statistics), components (parameter estimates)
- export model estimates

## Core Functionality

```@docs
Econometrics
EconometricModel
```

## Statistical/Regression Model Abstraction

```@docs
aic
aicc
bic
r2
adjr2
mss
rss
deviance
nulldeviance
loglikelihood
nullloglikelihood
coef
dof
dof_residual
coefnames(::EconometricModel)
coeftable(::EconometricModel)
islinear
informationmatrix
vcov(::EconometricModel)
stderror(::EconometricModel, ::Econometrics.VCE)
confint(::EconometricModel)
hasintercept
isfitted
fit(::EconometricModel)
fit!(::EconometricModel)
response
meanresponse
fitted
predict
modelmatrix
residuals
leverage
nobs
weights(::EconometricModel)
```

## Estimators

```@docs
BetweenEstimator
RandomEffectsEstimator
Econometrics.VCE
```

## Formula Components
```@docs
absorb
```
