# Econometrics

## Setup

```@example Main
using Econometrics, CSV, RDatasets
```

## Continuous Response Models

```@example Main
data = RDatasets.dataset("Ecdat", "Crime") |>
  (data -> select(data, [:County, :Year, :CRMRTE, :PrbConv, :AvgSen, :PrbPris]))
first(data, 6)
```

### Pooling
```@example Main
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data)
```

You can also request heteroscedasticity consistent estimators for linear models.

```@example Main
vcov(model, HC0)
```

```@example Main
vcov(model, HC1)
```

```@example Main
vcov(model, HC2)
```

```@example Main
vcov(model, HC3)
```

```@example Main
vcov(model, HC4)
```

### Between
```@example Main
model = fit(BetweenEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County)
```

### Within
```@example Main
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
            data)
```

### Within (two-ways)
```@example Main
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County + Year)),
            data)
```

### Random
```@example Main
model = fit(RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County,
            time = :Year)
```

### Random with Instrumental Variables
```@example Main
model = fit(RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
            data,
            panel = :County,
            time = :Year)
```

## Nominal Response Model

```@example Main
data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
   CSV.read |>
   (data -> select(data, [:insure, :age, :male, :nonwhite, :site])) |>
   dropmissing |>
   (data -> categorical!(data, [:insure, :site]))
first(data, 6)
```

```@example Main
model = fit(EconometricModel,
            @formula(insure ~ age + male + nonwhite + site),
            data,
            contrasts = Dict(:insure => DummyCoding(base = "Uninsure")))
```

## Ordinal Response Model

```@example Main
data = RDatasets.dataset("Ecdat", "Kakadu") |>
       (data -> select(data, [:RecParks, :Sex, :Age, :Schooling]))
data.RecParks = convert(Vector{Int}, data.RecParks)
data.RecParks = levels!(categorical(data.RecParks, ordered = true), collect(1:5))
first(data, 6)
```

```@example Main
model = fit(EconometricModel,
            @formula(RecParks ~ Age + Sex + Schooling),
            data)
```
