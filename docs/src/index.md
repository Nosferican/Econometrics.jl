# Econometrics

## Setup

```@example Main
using Econometrics, CSV, RDatasets
```

## Continuous Response Models

```@example Main
data = RDatasets.dataset("Ecdat", "Crime")
```

### Pooling
```@example Main
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data)
```

### Between
```@example Main
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + between(County)),
            data)
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
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + PID(County) + TID(Year)),
            data)
```

### Random with Instrumental Variables
```@example Main
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + PID(County) + TID(Year)),
            data)
```

## Nominal Response Model

```@example Main
data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
   CSV.read |>
   (data -> data[[:insure, :age, :male, :nonwhite, :site]]) |>
   (data -> dropmissing!(data, disallowmissing = true)) |>
   (data -> categorical!(data, [:insure, :site]))
model = fit(EconometricModel,
            @formula(insure ~ 1 + age + male + nonwhite + site),
            data,
            contrasts = Dict(:insure => DummyCoding(base = "Uninsure")))
```

## Ordinal Response Model

```@example Main
data = RDatasets.dataset("Ecdat", "Kakadu")[[:RecParks, :Sex, :Age, :Schooling]]
data.RecParks = convert(Vector{Int}, data.RecParks)
data.RecParks = levels!(categorical(data.RecParks, ordered = true), collect(1:5))
model = fit(EconometricModel,
            @formula(RecParks ~ Age + Sex + Schooling),
            data,
            contrasts = Dict(:RecParks => DummyCoding(levels = collect(1:5)))
            )
```
