# Formula displays
@testset "Formula Displays" begin
    data = dataset("Ecdat", "Crime")
    # Continuous Response
    f = @formula(CRMRTE ~ 1 + PrbConv)
    model = fit(EconometricModel, f, data)
    @test Econometrics.clean_fm(model) == "Formula: CRMRTE ~ 1 + PrbConv"
    f = @formula(CRMRTE ~ 0 + PrbConv)
    model = fit(EconometricModel, f, data)
    @test Econometrics.clean_fm(model) == "Formula: CRMRTE ~ 0 + PrbConv"
    f = @formula(CRMRTE ~ -1 + PrbConv)
    model = fit(EconometricModel, f, data)
    @test Econometrics.clean_fm(model) == "Formula: CRMRTE ~ -1 + PrbConv"
    f = @formula(CRMRTE ~ -1 + PrbConv + PrbConv & PrbPris^2 + absorb(County + Year))
    model = fit(EconometricModel, f, data)
    @test Econometrics.clean_fm(model) ==
          "Formula: CRMRTE ~ -1 + PrbConv + absorb(County + Year) + PrbConv & (PrbPris ^ 2)"
    f = @formula(log(CRMRTE) ~ 1)
    model = fit(EconometricModel, f, data)
    @test Econometrics.clean_fm(model) == "Formula: log(CRMRTE) ~ 1"
    # Random Effects Model
    f = @formula(CRMRTE ~ 1 + PrbConv)
    model = fit(RandomEffectsEstimator, f, data, panel = :County, time = :Year)
    @test Econometrics.clean_fm(model) == "Formula: CRMRTE ~ PrbConv"
    # Nominal Response Model
    data =
        joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |> CSV.File |> DataFrame |>
        (data -> select(data, [:insure, :age, :male, :nonwhite, :site])) |> dropmissing |>
        (data -> categorical!(data, [:insure, :site]))
    f = @formula(insure ~ 0 + age^2 + male + nonwhite + site)
    model = fit(
        EconometricModel,
        f,
        data,
        contrasts = Dict(:insure => DummyCoding(base = "Uninsure")),
    )
    @test Econometrics.clean_fm(model) ==
          "Formula: insure ~ 0 + (age ^ 2) + male + nonwhite + site"
    # Ordinal Response Model
    data =
        dataset("Ecdat", "Kakadu") |>
        (data -> select(data, [:RecParks, :Sex, :Age, :Schooling]))
    data.RecParks = convert(Vector{Int}, data.RecParks)
    data.RecParks = levels!(categorical(data.RecParks, ordered = true), collect(1:5))
    model = fit(EconometricModel, @formula(RecParks ~ Age * Age + Sex), data)
    @test Econometrics.clean_fm(model) == "Formula: RecParks ~ Age + Sex + Age & Age"
end
