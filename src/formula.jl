function absorb end

function decompose(f::FormulaTerm,
                   data::AbstractDataFrame,
                   contrasts::Dict{Symbol},
                   wts::Union{Nothing,Symbol},
                   panel::Union{Nothing,Symbol},
                   time::Union{Nothing,Symbol},
                   estimator::Type{<:Union{EconometricsModel,ModelEstimator}})
    rhs = isa(f.rhs, Tuple) ? collect(f.rhs) : [f.rhs]
    absorbed = findall(t -> isa(t, FunctionTerm{typeof(absorb)}), rhs)
    # Conditions based on panel and temporal indices and absorbed features
    if isa(estimator, Type{<:BetweenEstimator}) && isnothing(panel)
        throw(ArgumentError("The between estimator requires the panel identifier."))
    elseif isa(estimator, Type{<:RandomEffectsEstimator}) && (isnothing(panel) || isnothing(time))
        throw(ArgumentError("The random effects estimator requires the panel and temporal identifiers."))
    elseif (isa(estimator, Type{<:BetweenEstimator}) ||
            isa(estimator, Type{<:RandomEffectsEstimator})) && !isempty(absorbed)
        throw(ArgumentError("Absorbing features is only implemented for least squares."))
    end
    pns = convert(Vector{Symbol}, filter!(!isnothing, union(termvars(f), [panel, time, wts])))
    data = copy(data[pns])
    if isa(wts, Symbol)
        data[wts] = ifelse(data[wts] .≤ 0, missing, data[wts])
    end
    data = dropmissing(data[pns])
    if isempty(absorbed)
        absorbed_features = ""
        absorbed = Vector{Vector{Vector{Int}}}()
    elseif length(absorbed) == 1
        absorbed = rhs[absorbed[1]]
        if isa(absorbed, AbstractVector)
            absorbed = unique(flatten(termvars(t) for t ∈ absorbed))
        else
            absorbed = termvars(absorbed)
        end
        absorbed_features =
        absorbed = [ [ findall(isequal(l), data[dim]) for l ∈ levels(data[dim]) ] for dim ∈ absorbed ]
    else
        throw(ArgumentError("There can only be at most one absorb term"))
    end
    exo_rhs = filter(t -> !(isa(t, FormulaTerm) || isa(t, FunctionTerm)), rhs)
    exogenous = FormulaTerm(f.lhs,
                            isempty(exo_rhs) ? Tuple([ConstantTerm(1)]) : Tuple(exo_rhs))
    @assert length(terms(exogenous.lhs)) == 1 "Response should be a single variable"
    iv = findall(t -> isa(t, FormulaTerm), rhs)
    if length(iv) == 0
        iv = @eval(@formula(0 ~ 0))
    elseif length(iv) == 1
        iv = rhs[iv[1]]
    else
        throw(ArgumentError("Formula syntax is response ~ exogenous + (endogenous ~ instruments)"))
    end
    sc = schema(data, contrasts)
    f = apply_schema(f, sc)
    exogenous = apply_schema(exogenous, sc, EconometricsModel)
    iv = apply_schema(iv, sc)
    # Conditions based on response type and instrumental variables
    if isa(f, FormulaTerm{<:CategoricalTerm})
        isa(iv.lhs, InterceptTerm{false}) ||
            throw(ArgumentError("Instrumental variables are only implemented for linear models."))
        if isordered(data[f.lhs.sym])
            any(x -> isa(x, InterceptTerm{false}), f.rhs.terms) &&
                throw(ArgumentError("This estimator requires an intercept term."))
            estimator = OrdinalResponse(f.lhs.contrasts)
        else
            estimator = NominalResponse(f.lhs.contrasts)
        end
    elseif isa(estimator, Type{<:RandomEffectsEstimator}) &&
           any(x -> isa(x, InterceptTerm{false}), f.rhs.terms)
        throw(ArgumentError("The random effects estimator requires an intercept term."))
    end
    if isa(exogenous.lhs, CategoricalTerm)
        y = data[exogenous.lhs.sym]
        X = modelcols(exogenous.rhs, data)
        if isordered(y)
            X = X[:,2:end]
        end
    else
        y, X = modelcols(exogenous, data)
    end
    if !isa(iv.lhs, InterceptTerm)
        z, Z = modelcols(iv, data)
        z = isa(z, Tuple) ? hcat(z...) : z
    else
        z = zeros(0, 0)
        Z = zeros(0, 0)
    end
    wts = isnothing(wts) ? FrequencyWeights(Ones(size(y, 1))) : data[wts]
    if !isa(wts, FrequencyWeights)
        wts = FrequencyWeights(wts)
    end
    if isa(estimator, Type{<:RandomEffectsEstimator})
        panel = (panel, [ findall(isequal(l), data[panel]) for l ∈ levels(data[panel]) ])
        time = (time, [ findall(isequal(l), data[time]) for l ∈ levels(data[time]) ])
        estimator = RandomEffectsEstimator(panel, time, X, y, z, Z, wts)
    elseif isa(estimator, Type{<:BetweenEstimator})
        estimator = BetweenEstimator(panel, [ findall(isequal(l), data[panel]) for l ∈ levels(data[panel]) ])
    elseif isa(estimator, Type{<:EconometricModel}) || isa(estimator, Type{<:ContinuousResponse})
        estimator = ContinuousResponse(absorbed)
    end
    data, exogenous, iv, estimator, X, y, z, Z, wts
end
