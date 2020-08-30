import MLJModelInterface
import Tables

mutable struct EconometricsMLJModel{M, C, F, L} <: MLJModelInterface.Probabilistic
    contrasts::C
    formula::F
    label::L
end

function EconometricsMLJModel(; model::Type{M},
                                contrasts::C = nothing,
                                formula::F,
                                label::L) where M where C where F where L
    return EconometricsMLJModel{M, C, F, L}(contrasts, formula, label)
end

function _merge_X_and_y(X, # X must be a table
                        y::AbstractVector,
                        label)
    return merge(Tables.columntable(X), NamedTuple{(label,)}((y,)))
end

function MLJModelInterface.fit(model::EconometricsMLJModel{M},
                               verbosity::Integer,
                               X, # X must be a table
                               y::AbstractVector,
                               w = nothing,
                               ) where M
    contrasts = model.contrasts
    formula = model.formula
    label = model.label
    data = _merge_X_and_y(X, y, model.label)
    if contrasts === nothing
        fitresult = fit(M, formula, data)
    else
        fitresult = fit(M, formula, data; contrasts = contrasts)
    end
    cache = nothing
    report = NamedTuple{}()
    return (fitresult, cache, report)
end

function MLJModelInterface.clean!(model::EconometricsMLJModel)
    warning = ""
    return warning
end

# TODO: implement the `MLJModelInterface.predict` method
function MLJModelInterface.predict(model::EconometricsMLJModel,
                                   fitresult,
                                   Xnew, # Xnew must be a table
                                   )
    throw(ErrorException("The predict method has not yet been implemented."))
end
