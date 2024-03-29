dispersion(obj::EconometricModel{<:LinearModelEstimators}) = true
dispersion(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}}) = false
isiv(obj::EconometricModel{<:LinearModelEstimators}) = obj.iv > 0
coef(obj::EconometricModel) = obj.β
coefnames(obj::EconometricModel) = obj.vars[2]
responsename(obj::EconometricModel) = obj.vars[1]
deviance(obj::EconometricModel) = -2 * loglikelihood(obj)
deviance(obj::EconometricModel{<:LinearModelEstimators}) =
    weights(obj) |> (
        wts -> isa(wts, FrequencyWeights) ?
            sum(w * (y - ŷ)^2 for (w, y, ŷ) in zip(wts, response(obj), fitted(obj))) :
            sum((y - ŷ)^2 for (y, ŷ) in zip(response(obj), fitted(obj)))
    )
"""
	hasintercept(obj::EconometricModel)::Bool

Return whether the model has an intercept.
"""
hasintercept(obj::EconometricModel) = true
hasintercept(obj::EconometricModel{<:LinearModelEstimators}) =
    !any(t -> isa(t, InterceptTerm{false}), terms(obj.f.rhs))
islinear(obj::EconometricModel{<:LinearModelEstimators}) =
    !(isa(obj.estimator, NominalResponse) || isa(obj.estimator, OrdinalResponse))
nulldeviance(obj::EconometricModel{<:LinearModelEstimators}) =
    meanresponse(obj) |> (ȳ -> sum((y - ȳ)^2 for y in response(obj)))
loglikelihood(obj::EconometricModel{<:LinearModelEstimators}) =
    √(deviance(obj) / dof_residual(obj)) |> (
        ϕ -> (sum(
            w * logpdf(Normal(μ, ϕ), y)
            for (w, μ, y) in zip(weights(obj), response(obj), fitted(obj))
        ))
    )
function loglikelihood(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}})
    y = response(obj)
    categories = levels(y)
    b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
    μ = predict(obj)
    wts = weights(obj)
    sum(
        wts[idx[1]] * logpdf(Categorical(collect(μ[idx[1], :])), idx[2])
        for idx in findall(b)
    )
end
function nullloglikelihood(obj::EconometricModel{<:LinearModelEstimators})
    ϕ = √(nulldeviance(obj) / (dof_residual(obj) + (dof(obj) - hasintercept(obj))))
    μ = meanresponse(obj)
    sum(w * logpdf(Normal(μ, ϕ), y) for (w, y) in zip(weights(obj), response(obj)))
end
function nullloglikelihood(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}})
    y = response(obj)
    categories = levels(y)
    b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
    μ = collect(vec(mean(b, dims = 1)))
    wts = weights(obj)
    sum(wᵢ * logpdf(Categorical(μ), yᵢ) for (yᵢ, wᵢ) in zip(y, wts))
end
# StatsBase.score(obj::StatisticalModel) = error("score is not defined for $(typeof(obj)).")
nobs(obj::EconometricModel) = sum(weights(obj))
dof(obj::EconometricModel{<:LinearModelEstimators}) =
    length(coef(obj)) + dispersion(obj) + obj.iv
dof(obj::EconometricModel{<:ContinuousResponse}) =
    length(coef(obj)) +
    dispersion(obj) +
    obj.iv +
    (obj.estimator.groups |> (g -> isempty(g) ? 0 : sum(length(group) - 1 for group in g)))
dof(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}}) =
    length(coef(obj)) + dispersion(obj)
dof_residual(obj::EconometricModel{<:LinearModelEstimators}) =
    nobs(obj) - dof(obj) + dispersion(obj) + hasintercept(obj.f)
# Verify the dof_residual(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}})
dof_residual(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}}) =
    nobs(obj) - dof(obj) + dispersion(obj) + (length(obj.estimator.categories) - 1)
mss(obj::EconometricModel{<:LinearModelEstimators}) =
    meanresponse(obj) |>
    (ȳ -> sum(w * abs2(ŷ - ȳ) for (ŷ, w) in zip(fitted(obj), weights(obj))))
rss(obj::EconometricModel{<:LinearModelEstimators}) =
    residuals(obj) |> (û -> û' * Diagonal(weights(obj)) * û)
r2(obj::EconometricModel) = 1 - loglikelihood(obj) / nullloglikelihood(obj)
r2(obj::EconometricModel{<:LinearModelEstimators}) =
    (isiv(obj) || !hasintercept(obj)) ? NaN : rss(obj) |> (r -> 1 - r / (r + mss(obj)))
function adjr2(obj::EconometricModel{<:LinearModelEstimators})
    (isiv(obj) || !hasintercept(obj)) && return (NaN)
    ℓℓ = loglikelihood(obj)
    ℓℓ₀ = nullloglikelihood(obj)
    k = dof(obj)
    if variant == :McFadden
        1 - (ll - k) / ll0
    else
        error(":McFadden is the only currently supported variant")
    end
end
informationmatrix(obj::EconometricModel; expected::Bool = true) = obj.Ψ
"""
	vcov(obj::EconometricModel)
	vcov(obj::EconometricModel{<:LinearModelEstimators}, vce::VCE = obj.vce)

Return the variance-covariance matrix for the coefficients of the model.
The `vce` argument allows to request variance estimators.
"""
vcov(obj::EconometricModel) = informationmatrix(obj)
function vcov(obj::EconometricModel{<:LinearModelEstimators})
    vcov(obj::EconometricModel{<:LinearModelEstimators}, obj.vce)
end
function vcov(obj::EconometricModel{<:LinearModelEstimators}, vce::VCE)
    vce == OIM && return deviance(obj) / dof_residual(obj) * informationmatrix(obj)
    X = modelmatrix(obj)
    Ψ = informationmatrix(obj)
    û = residuals(obj)
    m = nobs(obj)
    k = dof(obj) - hasintercept(obj)
    p = dof_residual(obj)
    λ = vce == HC1 ? m / p : 1
    if vce == HC0 || vce == HC1
        ũ = û .^ 2
    else
        h = leverage(obj)
        if vce == HC2
            ũ = û .^ 2 ./ (1 .- h)
        elseif vce == HC3
            ũ = û .^ 2 ./ (1 .- h) .^ 2
        elseif vce == HC4
            ũ = û .^ 2 ./ (1 .- h) .^ min.(4, m / k * h)
        end
    end
    Ω = X' * Diagonal(ũ) * X
    Hermitian(λ * Ψ * Ω * Ψ)
end
"""
	stderror(obj::EconometricModel)
	stderror(obj::EconometricModel{<:LinearModelEstimators}, vce::VCE = obj.vce)

Return the standard errors for the coefficients of the model.
The `vce` argument allows to request variance estimators.
"""
stderror(obj::EconometricModel, vce::VCE) = sqrt.(diag(vcov(obj, vce)))
"""
	confint(obj::EconometricModel; se::AbstractVector{<:Real} = stderror(obj), level::Real = 0.95)

Compute the confidence intervals for coefficients, with confidence level `level` (by default, 95%).
`se` can be provided as a precomputed value.
"""
function confint(
    obj::EconometricModel;
    se::AbstractVector{<:Real} = stderror(obj),
    level::Real = 0.95,
)
    @assert zero(level) ≤ level ≤ one(level)
    α = 1 - level
    se * quantile(TDist(dof_residual(obj)), 1 - α / 2) |>
    (σ -> coef(obj) |> (β -> hcat(β .- σ, β .+ σ)))
end
weights(obj::EconometricModel) = obj.wts
isfitted(obj::EconometricModel) = !isempty(obj.Ψ)
fitted(obj::EconometricModel) = obj.ŷ
response(obj::EconometricModel) = obj.y
meanresponse(obj::EconometricModel{<:LinearModelEstimators}) =
    mean(response(obj), weights(obj))
modelmatrix(obj::EconometricModel) = obj.X
function leverage(obj::EconometricModel)
    X = modelmatrix(obj)
    Ψ = informationmatrix(obj)
    ω = weights(obj)
    diag(X * Ψ * X' * Diagonal(ω))
end
residuals(obj::EconometricModel{<:LinearModelEstimators}) = response(obj) - fitted(obj)
function residuals(obj::EconometricModel{<:NominalResponse})
    @unpack y = obj
    @unpack categories = obj.estimator
    # Change from levels to the categories after StatsModels patch
    l = levels(y)
    y = [findfirst(isequal(x), l) for x in y]
    b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
    b - predict(obj)
end
predict(obj::EconometricModel) = fitted(obj)
predict(obj::EconometricModel{<:NominalResponse}) =
    mapslices(softmax, fitted(obj), dims = 2)
function predict(obj::EconometricModel{<:OrdinalResponse})
    y = response(obj)
    ŷ = fitted(obj)
    β = coef(obj)
    outcomes = obj.estimator.categories
    ζ = vcat(-Inf, β[end-length(outcomes)+2:end], Inf)
    mapreduce(
        row ->
            (
                cdf.(Logistic(), ζ[2:end] .- ŷ[row]) .-
                cdf.(Logistic(), ζ[1:end-1] .- ŷ[row])
            )',
        vcat,
        eachindex(y),
    )
end
"""
	coeftable(obj::EconometricModel;
		  level::Real = 0.95)
	coeftable(obj::EconometricModel{<:LinearModelEstimators};
		  level::Real = 0.95,
		  vce::VCE = obj.vce)

Return a table of class `CoefTable` with coefficients and related statistics.
`level` determines the level for confidence intervals (by default, 95%).
`vce` determines the variance-covariance estimator (by default, `OIM`).
"""
function coeftable(obj::EconometricModel; level::Real = 0.95, vce::VCE = obj.vce)
    β = coef(obj)
    σ = stderror(obj, vce)
    t = β ./ σ
    p = 2 * ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj, se = σ, level = level))
    lims = (100 * (1 - level) / 2, 100 * (1 - (1 - level) / 2))
    colnms = [
        "PE   ",
        "SE   ",
        "t-value",
        "Pr > |t|",
        string(@sprintf("%.2f", lims[1]), "%"),
        string(@sprintf("%.2f", lims[2]), "%"),
    ]
    rownms = coefnames(obj)
    CoefTable(mat, colnms, rownms, 4)
end
function coeftable(obj::EconometricModel{<:NominalResponse}; level::Real = 0.95)
    @assert zero(level) ≤ level ≤ one(level)
    β = coef(obj)
    σ = stderror(obj)
    t = β ./ σ
    p = 2 * ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj, se = σ, level = level))
    lims = (100 * (1 - level) / 2, 100 * (1 - (1 - level) / 2))
    colnms = [
        "PE   ",
        "SE   ",
        "t-value",
        "Pr > |t|",
        string(@sprintf("%.2f", lims[1]), "%"),
        string(@sprintf("%.2f", lims[2]), "%"),
    ]
    vars = coefnames(obj)
    lhs = responsename(obj)
    rownms = [ string(lhs, " ~ ", rhs) for lhs in lhs for rhs in vars ]
    CoefTable(mat, colnms, rownms, 4)
end
function coeftable(obj::EconometricModel{<:OrdinalResponse}; level::Real = 0.95)
    @assert zero(level) ≤ level ≤ one(level)
    β = coef(obj)
    σ = stderror(obj)
    t = β ./ σ
    p = 2 * ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj, se = σ, level = level))
    lims = (100 * (1 - level) / 2, 100 * (1 - (1 - level) / 2))
    colnms = [
        "PE   ",
        "SE   ",
        "t-value",
        "Pr > |t|",
        string(@sprintf("%.2f", lims[1]), "%"),
        string(@sprintf("%.2f", lims[2]), "%"),
    ]
    vars = coefnames(obj)
    outcomes = obj.estimator.categories
    rownms = vcat(
        vars,
        [
            string("(Intercept): ", outcomes[i], " | ", outcomes[i+1])
            for i in 1:length(outcomes)-1
        ],
    )
    CoefTable(mat, colnms, rownms, 4)
end
