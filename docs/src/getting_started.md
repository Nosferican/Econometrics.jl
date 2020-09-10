# Getting Started

## Installation

Econometrics.jl is registered with the Julia's General registry.

For installing the package, users can use Julia's package manager.

```
using Pkg; Pkg.add("Econometrics") # one alternative
]add Econometrics # same as the line above
```

The package manager will automatically resolve which release version for Econometrics.jl is compatible with the environment in use.

The package manager allows for other experiences such as using the development version of the package through

```
]add Econometrics#master
```

## Data

Econometrics.jl relies on users being able to pass in data for estimating the various models. This package allows for data to be passed through any tabular representation that implements the [Tables.jl](https://github.com/JuliaData/Tables.jl) API. For example, a `DataFrame` from [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) is a common solution for in-memory tabular data that implements the Tables.jl API.

The second assumption Econometrics.jl makes about the data is that categorical data uses the [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl) representations (i.e., `AbstractCategoricalVector`, `CategoricalValue`).

Lastly, the package currently assumes in-memory data.

## Specifying a Model

Statistical models use a formula with formulae syntax to specify the relation among the various features. For example, Econometrics.jl uses the following syntax

```
modelformula = @formula(response ~ exogenous + (endogenous ~ instruments) + absorb(high_dimensional_controls))
```

where the right-hand side is the response and the left-hand side has

- exogenous variables
- potentially endogenous features and additional instruments
- high dimensional controls

Formulas also allow for passing special terms such as interactions and special terms (e.g., `lag`, `lead`, `poly`). These have limited support at the moment.

For more information on the formulae implementation see [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl).

## Statistical Model API

Econometrics.jl implements a subset of the abstraction for statistical models provided by [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl). This API allows for a consistent API across the statitics ecosystem. It is common practice for various packages to implement different models/estimators. However, these can interacted uniformly with similar syntax.

### Estimating a model

```
model = fit(ModelType, # package specific type
            formula, # @formula(lhs ~ rhs)
            data, # a table
            args..., # model or fitting specific arguments
            kwargs... # model or fitting specific keyword arguments
            )
```

or

```
model = ModelType(formula, data, args...; kwargs...)
fit!(model, args..., kwargs...)
coeftable(model, args..., kwargs...)
```
