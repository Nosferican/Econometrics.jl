using Econometrics, Test

using CSV, RDatasets
using MLJBase: MLJBase
using MLJModelInterface: MLJModelInterface

@testset "MLJ model interface" begin
    @testset "Econometrics.jl + MLJModelInterface.jl" begin
        @testset "specify contrasts" begin
            contrasts = Dict(:insure => DummyCoding(base = "Uninsure"))
            formula = @formula(insure ~ age + male + nonwhite + site)
            label = :insure
            model = EconometricsMLJModel(;
                model = EconometricModel,
                contrasts = contrasts,
                formula = formula,
                label = label,
            )

            data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
                CSV.File |>
                (data -> CSV.select(data, [:insure, :age, :male, :nonwhite, :site])) |>
                CSV.dropmissing |>
                (data -> CSV.categorical!(data, [:insure, :site]))
            X = data[!, [:age, :male, :nonwhite, :site]]
            y = data[!, :insure]
            verbosity = 1
            fitresult, _, _ = MLJModelInterface.fit(model, verbosity, X, y)
            @test fitresult isa EconometricModel
            Xnew = deepcopy(X)
            @test_broken ynew = MLJModelInterface.predict(model, fitresult, Xnew)
        end
        @testset "don't specify contrasts" begin
            formula = @formula(insure ~ age + male + nonwhite + site)
            label = :insure
            model = EconometricsMLJModel(;
                model = EconometricModel,
                formula = formula,
                label = label,
                )

            data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
                CSV.File |>
                (data -> CSV.select(data, [:insure, :age, :male, :nonwhite, :site])) |>
                CSV.dropmissing |>
                (data -> CSV.categorical!(data, [:insure, :site]))
            X = data[!, [:age, :male, :nonwhite, :site]]
            y = data[!, :insure]
            verbosity = 1
            fitresult, _, _ = MLJModelInterface.fit(model, verbosity, X, y)
            @test fitresult isa EconometricModel
            Xnew = deepcopy(X)
            @test_broken ynew = MLJModelInterface.predict(model, fitresult, Xnew)
        end
    end
    @testset "Econometrics.jl + MLJBase.jl" begin
        contrasts = Dict(:insure => DummyCoding(base = "Uninsure"))
        formula = @formula(insure ~ age + male + nonwhite + site)
        label = :insure
        model = EconometricsMLJModel(;
            model = EconometricModel,
            contrasts = contrasts,
            formula = formula,
            label = label,
            )

        data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
                CSV.File |>
                (data -> CSV.select(data, [:insure, :age, :male, :nonwhite, :site])) |>
                CSV.dropmissing |>
                (data -> CSV.categorical!(data, [:insure, :site]))
        X = data[!, [:age, :male, :nonwhite, :site]]
        y = data[!, :insure]
        mach = MLJBase.machine(model, X, y)
        MLJBase.fit!(mach)
        @test mach.fitresult isa EconometricModel
        Xnew = deepcopy(X)
        @test_broken ynew = MLJBase.predict(mach, Xnew)
        mach = MLJBase.machine(model, X, y)
        @test_broken cv_result = MLJBase.evaluate!(mach, resampling=MLJBase.CV(; nfolds = 6, shuffle = true), measure=MLJBase.brier_score, operation=MLJBase.predict)
    end
end
