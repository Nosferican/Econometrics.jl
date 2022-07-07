"""
    EconometricsModel <: RegressionModel

Abstract type provided by Econometrics.jl
"""
abstract type EconometricsModel <: RegressionModel end
implicit_intercept(::Type{<:EconometricsModel}) = true
# Estimators
"""
    ModelEstimator

Abstract type for model estimators.
"""
abstract type ModelEstimator end
"""
    LinearModelEstimators

Abstract type for linear model estimators.
"""
abstract type LinearModelEstimators <: ModelEstimator end
"""
    ContinuousResponse(groups::Vector{Vector{Vector{Int}}}) <: LinearModelEstimators

Continuous response estimator with potential features absorption.
"""
struct ContinuousResponse <: LinearModelEstimators
    groups::Vector{Vector{Vector{Int}}}
end
show(io::IO, ::MIME"text/plain", estimator::ContinuousResponse) = println(io, "Continuous Response Model")
"""
    BetweenEstimator(effect::Symbol,
                     groups::Vector{Vector{Int}}) <: LinearModelEstimators

Continuous response estimator collapsing a dimension in a longitudinal setting.
"""
struct BetweenEstimator <: LinearModelEstimators
    effect::Symbol
    groups::Vector{Vector{Int}}
end
function show(io::IO, ::MIME"text/plain", estimator::BetweenEstimator)
    @unpack effect, groups = estimator
    println(io, "Between Estimator")
    println(io, "$effect with $(sum(length, groups)) groups")
    T = length.(groups)
    if !isempty(T)
        E = extrema(T)
        if E[1] ≈ E[2]
            println(io, "Balanced groups with size $(E[1])")
        else
            T̄ = round(harmmean(T), digits = 2)
            println(io, "Groups of sizes [$(E[1]), $(E[2])] with harmonnic mean of $(T̄)")
        end
    end
end
"""
    RandomEffectsEstimator(pid::Tuple{Symbol,Vector{Vector{Int}}},
                           tid::Tuple{Symbol,Vector{Vector{Int}}},
                           idiosyncratic::Float64,
                           individual::Float64,
                           θ::Vector{Float64}) <: LinearModelEstimators

Swamy-Arora estimator.
"""
struct RandomEffectsEstimator <: LinearModelEstimators
    pid::Tuple{Symbol,Vector{Vector{Int}}}
    tid::Tuple{Symbol,Vector{Vector{Int}}}
    idiosyncratic::Float64
    individual::Float64
    θ::Vector{Float64}
    RandomEffectsEstimator(pid, tid) =
        new((pid, Vector{Vector{Int}}()), (tid, Vector{Vector{Int}}()), NaN, NaN, zeros(0))
    function RandomEffectsEstimator(pid, tid, X, y, z, Z, wts)
        Xbe, ybe, βbe, Ψbe, ŷbe, wtsbe, pivbe =
            fit(BetweenEstimator(pid[1], pid[2]), X, y, z, Z, wts)
        Xfe, yfe, βfe, Ψfe, ŷfe, wtsfe, pivfe =
            fit(ContinuousResponse([pid[2]]), X, y, z, Z, wts)
        ivk = size(Z, 2)
        T = length.(pid[2])
        T̄ = harmmean(T)
        σₑ² =
            sum(wᵢ * (yᵢ - ŷᵢ)^2 for (wᵢ, yᵢ, ŷᵢ) in zip(wtsfe, yfe, ŷfe)) /
            (sum(wtsfe) - length(βfe) - ivk - length(pid[2]) + 1)
        σᵤ² = max(
            0,
            sum((yᵢ - ŷᵢ)^2 for (yᵢ, ŷᵢ) in zip(ybe, ŷbe)) /
            (length(ybe) - length(βbe) - ivk) - σₑ² / T̄,
        )
        θ = 1 .- sqrt.(σₑ² ./ (T .* σᵤ² .+ σₑ²))
        new(pid, tid, √σₑ², √σᵤ², θ)
    end
end
function show(io::IO, ::MIME"text/plain", estimator::RandomEffectsEstimator)
    pid, D = estimator.pid
    tid, T = estimator.tid
    @unpack idiosyncratic, individual = estimator
    println(io, "One-way Random Effect Model")
    println(io, "Longitudinal dataset: $pid, $tid")
    L = length.(D)
    if !isempty(L)
        E = extrema(L)
        if E[1] ≈ E[2]
            println(io, "Balanced dataset with $(length(L)) panels of length $(E[1])")
        else
            T̄ = round(harmmean(L), digits = 2)
            println(io, "Unbalanced dataset with $(length(L)) panels")
            println(io, "Panel lengths [$(E[1]), $(E[2])] with harmonnic mean of $(T̄)")
        end
    end
    println(io, "individual error component: $(round(individual, digits = 4))")
    println(io, "idiosyncratic error component: $(round(idiosyncratic, digits = 4))")
    println(io, "ρ: $(round(individual^2 / (individual^2 + idiosyncratic^2), digits = 4))")
end
"""
    NominalResponse(categories::ContrastsMatrix) <: ModelEstimator

Multinomial logistic regression.
"""
struct NominalResponse{T} <: ModelEstimator
    categories::Vector{T}
    function NominalResponse(obj::ContrastsMatrix)
        categories = isnothing(obj.contrasts.base) ? obj.levels :
            union(vcat(obj.contrasts.base, obj.levels))
        new{eltype(categories)}(categories)
    end
end
function show(io::IO, ::MIME"text/plain", estimator::NominalResponse)
    println(io, "Probability Model for Nominal Response")
    println(io, "Categories: $(join(estimator.categories, ", "))")
end
"""
    OrdinalResponse(categories::ContrastsMatrix) <: ModelEstimator

Proportional odds logistic regression.
"""
struct OrdinalResponse{T} <: ModelEstimator
    categories::Vector{T}
    function OrdinalResponse(obj::ContrastsMatrix)
        categories = isnothing(obj.contrasts.levels) ? obj.levels : obj.contrasts.levels
        new{eltype(categories)}(categories)
    end
end
function show(io::IO, ::MIME"text/plain", estimator::OrdinalResponse)
    println(io, "Probability Model for Ordinal Response")
    println(io, "Categories: $(join(estimator.categories, " < "))")
end
"""
    VCE

Variance-covariance estimators.

- Observed Information Matrix (OIM)
- Heteroscedasticity Consistent: HC0, HC1, HC2, HC3, HC4
"""
@enum VCE begin
    OIM = -1
    HC0 = 0
    HC1 = 1
    HC2 = 2
    HC3 = 3
    HC4 = 4
end
