"""
    within(obj::AbstractMatrix{<:Real},
           D::AbstractVector{<:AbstracVector{<:AbstracVector{<:Integer}}},
           wts::AbstractVector)

This function performs the within transformation given a model matrix and fixed effects using the method of alternating projections.
"""
@views within(obj::AbstractMatrix{<:Real},
              D::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}},
              wts::AbstractVector) =
    isempty(D) ? obj : mapslices(col -> within(col, D, wts), obj, dims = 1)
@views function within(obj::AbstractVector{<:Real},
                       D::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}},
                       wts::AbstractVector = Ones(length(obj)))
    isempty(D) && return obj
    current = copy(obj)
    output = copy(obj)
    er = Inf
    μ = mean(obj, wts)
    while er > 1e-8
        for dimension ∈ D
            current = copy(output)
            for group ∈ dimension
                output[group] .-= mean(current[group], FrequencyWeights(wts[group]))
            end
        end
        er = √sum((x - y)^2 for (x, y) ∈ zip(output, current))
    end
    output .+= μ
end
"""
    partialwithin(obj::AbstractVecOrMat{<:Real},
                  D::AbstractVector{<:AbstractVector{<:Integer}},
                  θ::AbstractVector{<:Real})

This function performs the partial within transformation given a model matrix subgroups and subgroup specific error components.
"""
@views function partialwithin(obj::AbstractVecOrMat{<:Real},
                              panel::AbstractVector{<:AbstractVector{<:Integer}},
                              θ::AbstractVector{<:Real},
                              wts::AbstractVector)
    output = copy(obj)
    if !isempty(wts)
        for (panel, θ) ∈ zip(panel, θ)
            output[panel,:] .-= θ * mean(obj[panel, :], FrequencyWeights(wts[panel]), dims = 1)
        end
    else
        for (panel, θ) ∈ zip(panel, θ)
            output[panel,:] .-= θ * mean(obj[panel, :], dims = 1)
        end
    end
    output
end
"""
    transform(estimator::LinearModelEstimators, wts::FrequencyWeights)
    transform(estimator::BetweenEstimator, wts::FrequencyWeights)
    transform(estimator::ContinuousResponse,
              obj::AbstractVecOrMat{<:Number},
              wts::FrequencyWeights)
    transform(estimator::BetweenEstimator,
              obj::AbstractVector{<:Number},
              wts::FrequencyWeights)
    transform(estimator::BetweenEstimator,
              obj::AbstractMatrix{<:Number},
              wts::FrequencyWeights)
    transform(estimator::RandomEffectsEstimator,
              obj::AbstractVecOrMat,
              wts::FrequencyWeights)

Applies a transformation to a model component based on the model estimator.
"""
function transform end
transform(estimator::LinearModelEstimators, wts::FrequencyWeights) = wts
transform(estimator::BetweenEstimator, wts::FrequencyWeights) =
    FrequencyWeights(ones(length(estimator.groups)))
transform(estimator::ContinuousResponse,
          obj::AbstractVecOrMat{<:Number},
          wts::FrequencyWeights) = isempty(obj) ? obj : within(obj, estimator.groups, wts)
transform(estimator::BetweenEstimator,
          obj::AbstractVector{<:Number},
          wts::FrequencyWeights) = isempty(obj) ? obj :
    mapreduce(group -> mean(obj[group], FrequencyWeights(wts[group])),
              vcat,
              estimator.groups)
transform(estimator::BetweenEstimator,
          obj::AbstractMatrix{<:Number},
          wts::FrequencyWeights) = isempty(obj) ? obj :
    mapreduce(group -> mean(obj[group, :], FrequencyWeights(wts[group]), dims = 1),
              vcat,
              estimator.groups)
transform(estimator::RandomEffectsEstimator,
          obj::AbstractVecOrMat,
          wts::FrequencyWeights) = isempty(obj) ? obj :
    partialwithin(obj, estimator.pid[2], estimator.θ, wts)
