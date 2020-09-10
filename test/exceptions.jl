# Exceptions
@testset "Exceptions" begin
    data = dataset("Ecdat", "Crime")
    data.AvgSen2 = 2 * data.AvgSen
    # Rank-deficient model
    @test isa(
        fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + AvgSen2),
            data,
        ),
        EconometricModel,
    )
    @test isa(
        fit(
            RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County,
            time = :Year,
        ),
        EconometricModel,
    )
    # Model specifics
    @test_throws(
        ArgumentError("The between estimator requires the panel identifier."),
        fit(
            BetweenEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
            data,
        )
    )
    @test_throws(
        ArgumentError("Absorbing features is only implemented for least squares."),
        fit(
            BetweenEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
            data,
            panel = :County,
        )
    )
    @test_throws(
        ArgumentError("The random effects estimator requires the panel and temporal identifiers."),
        fit(
            RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
            data,
            panel = :County,
        )
    )
    @test_throws(
        ArgumentError("Absorbing features is only implemented for least squares."),
        fit(
            RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
            data,
            panel = :County,
            time = :Year,
        )
    )
    data = dataset("datasets", "iris")
    @test_throws(
        ConvergenceException,
        fit(EconometricModel, @formula(Species ~ SepalLength + SepalWidth), data)
    )
    # Robust variance covariance estimators
    @test_throws(
        ArgumentError("Robust variance covariance estimators only available for continous response models."),
        fit(
            EconometricModel,
            @formula(Species ~ SepalLength + SepalWidth),
            data,
            vce = HC1,
        )
    )
    @test_throws(
        ArgumentError("When absorbing features, only HC2-HC4 are unavailable."),
        fit(
            EconometricModel,
            @formula(SepalLength ~ SepalLength + absorb(Species)),
            data,
            vce = HC2,
        )
    )
end
