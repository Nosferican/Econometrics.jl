"""
    EconometricModel(f::FormulaTerm, data;
                     contrasts::Dict{Symbol} = Dict{Symbol,Union{<:AbstractContrasts,<:AbstractTerm}},
                     weights::Union{Nothing,Symbol} = nothing,
                     panel::Union{Nothing,Symbol} = nothing,
                     time::Union{Nothing,Symbol} = nothing,
                     estimator::ModelEstimator = ModelEstimator)

    Use fit(EconometricModel, f, data, contrasts = contrasts)
    Formula has syntax: @formula(response ~ exogenous + (endogenous ~ instruments))
    For absorbing categorical features use the term `absorb(features)`
    For the between estimator use the term `between(features)`
    For the one-way random effects model use the terms `PID(pid) + TID(tid)`
"""
mutable struct EconometricModel{E<:ModelEstimator,
                                F<:FormulaTerm,
                                Y<:AbstractVecOrMat{<:Number},
                                W<:FrequencyWeights,
                                Ŷ<:AbstractVecOrMat{<:Float64},
                                N<:Tuple{<:Union{<:AbstractVector{<:AbstractString},
                                                 <:AbstractString},
                                         <:AbstractVector{<:AbstractString}},
                                VC<:Union{<:Type{<:VCE},<:VCE}} <: EconometricsModel
    estimator::E
    f::F
    data::DataFrame
    X::Matrix{Float64}
    y::Y
    w::W
    β::Vector{Float64}
    Ψ::Hermitian{Float64,Matrix{Float64}}
    ŷ::Ŷ
    vars::N
    iv::Int
    vce::VC
end
function show(io::IO, obj::EconometricModel{<:LinearModelEstimators})
    show(io, obj.estimator)
    println(io, @sprintf("Number of observations: %i", nobs(obj)))
    println(io, @sprintf("Null Loglikelihood: %.2f", nullloglikelihood(obj)))
    println(io, @sprintf("Loglikelihood: %.2f", loglikelihood(obj)))
    println(io, @sprintf("R-squared: %.4f", r2(obj)))
    W, F, p = wald(obj)
    if !isnan(p)
        println(io, @sprintf("Wald: %.2f ∼ F(%i, %i) ⟹ Pr > F = %.4f", W, params(F)..., p))
    end
    println(io, clean_fm(obj))
    println(io, "Variance Covariance Estimator: $(obj.vce)")
    show(io, coeftable(obj))
end
function show(io::IO, obj::EconometricModel{<:ContinuousResponse})
    show(io, obj.estimator)
    ℓℓ₀ = nullloglikelihood(obj)
    ℓℓ = loglikelihood(obj)
    absorbed = !isempty(obj.estimator.groups)
    println(io, @sprintf("Number of observations: %i", nobs(obj)))
    println(io, @sprintf("Null Loglikelihood: %.2f", ℓℓ₀))
    println(io, @sprintf("Loglikelihood: %.2f", ℓℓ))
    println(io, @sprintf("R-squared: %.4f", r2(obj)))
    if !absorbed
        lr = 2(ℓℓ - ℓℓ₀)
        vars = coefnames(obj)
        k = count(x -> !occursin("(Intercept)", x), vars[2])
        if k > zero(k)
            χ² = Chisq(k)
            p = ccdf(χ², lr)
            println(io, @sprintf("LR Test: %.2f ∼ χ²(%i) ⟹  Pr > χ² = %.4f", lr, k, p))
        end
    else
        W, F, p = wald(obj)
        if !isnan(p)
            println(io, @sprintf("Wald: %.2f ∼ F(%i, %i) ⟹ Pr > F = %.4f", W, params(F)..., p))
        end
    end
    println(io, clean_fm(obj))
    println(io, "Variance Covariance Estimator: $(obj.vce)")
    show(io, coeftable(obj))
end
function show(io::IO, obj::EconometricModel{<:NominalResponse})
    show(io, obj.estimator)
    ℓℓ₀ = nullloglikelihood(obj)
    ℓℓ = loglikelihood(obj)
    println(io, @sprintf("Number of observations: %i", nobs(obj)))
    println(io, @sprintf("Null Loglikelihood: %.2f", ℓℓ₀))
    println(io, @sprintf("Loglikelihood: %.2f", ℓℓ))
    println(io, @sprintf("R-squared: %.4f", r2(obj)))
    lr = 2(ℓℓ - ℓℓ₀)
    vars = coefnames(obj)
    k = length(vars[1]) * count(x -> !occursin("(Intercept)", x), vars[2])
    if k > zero(k)
        χ² = Chisq(k)
        p = ccdf(χ², lr)
        println(io, @sprintf("LR Test: %.2f ∼ χ²(%i) ⟹ Pr > χ² = %.4f", lr, k, p))
    end
    println(io, clean_fm(obj))
    show(io, coeftable(obj))
end
function show(io::IO, obj::EconometricModel{<:OrdinalResponse})
    show(io, obj.estimator)
    ℓℓ₀ = nullloglikelihood(obj)
    ℓℓ = loglikelihood(obj)
    println(io, @sprintf("Number of observations: %i", nobs(obj)))
    println(io, @sprintf("Null Loglikelihood: %.2f", ℓℓ₀))
    println(io, @sprintf("Loglikelihood: %.2f", ℓℓ))
    println(io, @sprintf("R-squared: %.4f", r2(obj)))
    lr = 2(ℓℓ - ℓℓ₀)
    vars = coefnames(obj)
    k = length(vars[2])
    if k > zero(k)
        χ² = Chisq(k)
        p = ccdf(χ², lr)
        println(io, @sprintf("LR Test: %.2f ∼ χ²(%i) ⟹ Pr > χ² = %.4f", lr, k, p))
    end
    println(io, clean_fm(obj))
    show(io, coeftable(obj))
end
function fit(estimator::Type{<:Union{EconometricModel,ModelEstimator}},
             f::FormulaTerm,
             data;
             contrasts::Dict{Symbol} = Dict{Symbol,Union{<:AbstractContrasts,<:AbstractTerm}}(),
             weights::Union{Nothing,Symbol} = nothing,
             panel::Union{Nothing,Symbol} = nothing,
             time::Union{Nothing,Symbol} = nothing,
             vce::VCE = OIM)
    data, exogenous, iv, estimator, X, y, z, Z, wts =
        decompose(deepcopy(f), data, contrasts, weights, panel, time, estimator, vce)
    X, y, β, Ψ, ŷ, wts, piv = solve(estimator, X, y, z, Z, wts)
    vars = (coefnames(exogenous.lhs),
            convert(Vector{String}, vcat(coefnames(exogenous.rhs), coefnames(iv.lhs))[piv]))
    EconometricModel(estimator, f, data, X, y, wts, β, Ψ, ŷ, vars, size(Z, 2), vce)
end
