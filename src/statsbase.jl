dispersion(obj::EconometricModel{<:LinearModelEstimators}) = true
dispersion(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}}) = false
isiv(obj::EconometricModel{<:LinearModelEstimators}) = obj.iv > 0
coef(obj::EconometricModel) = obj.β
coefnames(obj::EconometricModel) = obj.vars
# StatsBase.confint(obj::StatisticalModel) = error("coefint is not defined for $(typeof(obj)).")
deviance(obj::EconometricModel) = -2loglikelihood(obj)
deviance(obj::EconometricModel{<:LinearModelEstimators}) =
	weights(obj) |>
	(wts -> isa(wts, FrequencyWeights) ?
			sum(w * (y - ŷ)^2 for (w, y, ŷ) ∈ zip(wts, response(obj), fitted(obj))) :
			sum((y - ŷ)^2 for (y, ŷ) ∈ zip(response(obj), fitted(obj))))
hasintercept(obj::EconometricModel) = true
hasintercept(obj::EconometricModel{<:LinearModelEstimators}) =
	!any(t -> isa(t, InterceptTerm{false}), terms(obj.f.rhs))
islinear(obj::EconometricModel{<:LinearModelEstimators}) =
	!(isa(obj.estimator, NominalResponse) || isa(obj.estimator, OrdinalResponse))
nulldeviance(obj::EconometricModel{<:LinearModelEstimators}) =
	meanresponse(obj) |> (ȳ -> sum((y - ȳ)^2 for y ∈ response(obj)))
loglikelihood(obj::EconometricModel{<:LinearModelEstimators}) =
	√(deviance(obj) / dof_residual(obj)) |>
	(ϕ -> (sum(w * logpdf(Normal(μ, ϕ), y)
		   for (w, μ, y) ∈ zip(weights(obj), response(obj), fitted(obj)))))
function loglikelihood(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}})
	y = response(obj)
	categories = levels(y)
	b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
	μ = predict(obj)
	wts = weights(obj)
	sum(wts[idx[1]] * logpdf(Categorical(collect(μ[idx[1],:])), idx[2]) for idx in findall(b))
end
function nullloglikelihood(obj::EconometricModel{<:LinearModelEstimators})
	ϕ = √(nulldeviance(obj) / (dof_residual(obj) + (length(coef(obj)) - hasintercept(obj))))
	μ = meanresponse(obj)
	sum(w * logpdf(Normal(μ, ϕ), y) for (w, y) ∈ zip(weights(obj), response(obj)))
end
function nullloglikelihood(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}})
	y = response(obj)
	categories = levels(y)
	b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
	μ = collect(vec(mean(b, dims = 1)))
	wts = weights(obj)
	sum(wᵢ * logpdf(Categorical(μ), yᵢ) for (yᵢ, wᵢ) ∈ zip(y, wts))
end
# StatsBase.score(obj::StatisticalModel) = error("score is not defined for $(typeof(obj)).")
nobs(obj::EconometricModel) = sum(obj.w)
dof(obj::EconometricModel{<:LinearModelEstimators}) =
	length(coef(obj)) + dispersion(obj) + obj.iv
dof(obj::EconometricModel{<:ContinuousResponse}) =
	length(coef(obj)) + dispersion(obj) + obj.iv +
	(obj.estimator.groups |>
	 (g -> isempty(g) ? 0 : sum(length(group) - 1 for group ∈ g)))
dof(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}}) =
	length(coef(obj)) + dispersion(obj)
dof_residual(obj::EconometricModel{<:LinearModelEstimators}) =
	nobs(obj) - dof(obj) + dispersion(obj) + hasintercept(obj.f)
# Verify the dof_residual(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}})
dof_residual(obj::EconometricModel{<:Union{NominalResponse,OrdinalResponse}}) =
	nobs(obj) - dof(obj) + dispersion(obj) + (length(obj.estimator.categories) - 1)
mss(obj::EconometricModel{<:LinearModelEstimators}) =
	meanresponse(obj) |> (ȳ -> sum(w * abs2(ŷ - ȳ) for (ŷ, w) ∈ zip(fitted(obj), weights(obj))))
rss(obj::EconometricModel{<:LinearModelEstimators}) =
	residuals(obj) |> (û -> û' * Diagonal(weights(obj)) * û)
r2(obj::EconometricModel) = 1 - loglikelihood(obj) / nullloglikelihood(obj)
r2(obj::EconometricModel{<:LinearModelEstimators}) =
	(isiv(obj) || !hasintercept(obj)) ? NaN : rss(obj) |> (r -> 1 - r / (r + mss(obj)))
function adjr2(obj::EconometricModel{<:LinearModelEstimators})
	(isiv(obj) || !hasintercept(obj)) && return(NaN)
	ℓℓ = loglikelihood(obj)
	ℓℓ₀ = nullloglikelihood(obj)
	k = dof(obj)
	if variant == :McFadden
		1 - (ll - k)/ll0
	else
		error(":McFadden is the only currently supported variant")
	end
end
informationmatrix(obj::EconometricModel; expected::Bool = true) = obj.Ψ
vcov(obj::EconometricModel) = informationmatrix(obj)
vcov(obj::EconometricModel{<:LinearModelEstimators}) =
	deviance(obj) / dof_residual(obj) * informationmatrix(obj)
stderror(obj::EconometricModel) = sqrt.(diag(vcov(obj)))
confint(obj::EconometricModel, α = 0.05) =
    stderror(obj) * quantile(TDist(dof_residual(obj)), 1 - α / 2) |>
    (σ -> coef(obj) |>
        (β -> hcat(β .- σ, β .+ σ)))
weights(obj::EconometricModel) = obj.w
isfitted(obj::EconometricModel) = !isempty(obj.Ψ)
fitted(obj::EconometricModel) = obj.ŷ
response(obj::EconometricModel) = obj.y
meanresponse(obj::EconometricModel{<:LinearModelEstimators}) = mean(response(obj), weights(obj))
modelmatrix(obj::EconometricModel) = obj.X
# leverage(obj::RegressionModel) = error("leverage is not defined for $(typeof(obj)).")
residuals(obj::EconometricModel{<:LinearModelEstimators}) = response(obj) - fitted(obj)
function residuals(obj::EconometricModel{<:NominalResponse})
	@unpack y = obj
	@unpack categories = obj.estimator
	# Change from levels to the categories after StatsModels patch
	l = levels(y)
	y = [ findfirst(isequal(x), l) for x ∈ y ]
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
	ζ = vcat(-Inf, β[end - length(outcomes) + 2:end], Inf)
	mapreduce(row -> (cdf.(Logistic(), ζ[2:end] .- ŷ[row]) .-
					  cdf.(Logistic(), ζ[1:end - 1] .- ŷ[row]))',
			  vcat,
			  eachindex(y))
end
function coeftable(obj::EconometricModel)
    β = coef(obj)
    σ = stderror(obj)
    t = β ./ σ
    p = 2ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj))
    colnms = ["PE   ", "SE   ", "t-value", "Pr > |t|", "2.5%", "97.5%"]
    rownms = obj.vars[2]
    CoefTable(mat, colnms, rownms, 4)
end
function coeftable(obj::EconometricModel{<:NominalResponse})
    β = coef(obj)
    σ = stderror(obj)
    t = β ./ σ
    p = 2ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj))
    colnms = ["PE   ", "SE   ", "t-value", "Pr > |t|", "2.5%", "97.5%"]
	vars = coefnames(obj)
    rownms = [ string(lhs, " ~ ", rhs) for lhs ∈ vars[1] for rhs ∈ vars[2] ]
    CoefTable(mat, colnms, rownms, 4)
end
function coeftable(obj::EconometricModel{<:OrdinalResponse})
    β = coef(obj)
    σ = stderror(obj)
    t = β ./ σ
    p = 2ccdf.(TDist(dof_residual(obj)), abs.(t))
    mat = hcat(β, σ, t, p, confint(obj))
    colnms = ["PE   ", "SE   ", "t-value", "Pr > |t|", "2.5%", "97.5%"]
	vars = coefnames(obj)
	outcomes = obj.estimator.categories
	rownms = vcat(vars[2],
				  [ string("(Intercept): ", outcomes[i], " | ", outcomes[i + 1])
				    for i ∈ 1:length(outcomes) - 1 ])
    CoefTable(mat, colnms, rownms, 4)
end
