"""
    absorb

Function for constructing the `FunctionTerm{typeof(absorb)}` used in decompose.
"""
function absorb end
"""
    decompose(f::FormulaTerm,
              data,
              contrasts::Dict{Symbol},
              wts::Union{Nothing,Symbol},
              panel::Union{Nothing,Symbol},
              time::Union{Nothing,Symbol},
              estimator::Type{<:Union{EconometricsModel,ModelEstimator}},
              vce::VCE)

Decomposes the arguments passed to a fit(::EconometricModel) into its components.

# Returns

- data
- exogenous::FormulaTerm
- iv::FormulaTerm
- estimator::ModelEstimator
- X::Matrix{Float64}
- y::VecOrMat{<:Union{Bool,Int,Float64}}
- z::Vector{Float64}
- Z::Matrix{Float64}
- wts::FrequencyWeights
"""
function decompose(f::FormulaTerm,
                   data,
                   contrasts::Dict{Symbol},
                   wts::Union{Nothing,Symbol},
                   panel::Union{Nothing,Symbol},
                   time::Union{Nothing,Symbol},
                   estimator::Type{<:Union{EconometricsModel,ModelEstimator}},
                   vce::VCE)
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
    data = select(data, pns...)
    if isa(wts, Symbol)
		wts = ifelse.(getproperty(data, wts) .≤ 0, missing, getproperty(data, wts))
    end
	categorical_variables = Tables.schema(data) |>
		(x -> [name for (name, type) ∈ zip(x.names, x.types) if
			isa(type, Type{<:Union{AbstractString, CategoricalValue}})])
	data_categorical =
		zip(categorical_variables,
			(levels(col), isordered(col)) for
		    col ∈ eachcolumn(select(data, categorical_variables...))) |>
		Dict
	data = materializer(data)(row for row ∈ rows(data) if
        all(pn -> !ismissing(getproperty(row, pn)), propertynames(row)))
    for cn ∈ categorical_variables
		col = getproperty(data, cn)
		levels!(col, intersect(data_categorical[cn][1],
							   unique(col)))
		ordered!(col, data_categorical[cn][2])
	end
    if isempty(absorbed)
        absorbed = Vector{Vector{Vector{Int}}}()
    elseif length(absorbed) == 1
        vce ≥ HC2 &&
            throw(ArgumentError("When absorbing features, only HC2-HC4 are unavailable."))
        absorbed = rhs[absorbed[1]]
        if isa(absorbed, AbstractVector)
            absorbed = unique(flatten(termvars(t) for t ∈ absorbed))
        else
            absorbed = termvars(absorbed)
        end
        absorbed = [ [ findall(isequal(l), getproperty(data, dim)) for l ∈ levels(getproperty(data, dim)) ] for dim ∈ absorbed ]
    else
        throw(ArgumentError("There can only be at most one absorb term"))
    end
    exo_rhs = filter(t -> !(isa(t, FormulaTerm) || isa(t, FunctionTerm{typeof(absorb)})), rhs)
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
        vce == OIM ||
            throw(ArgumentError("Robust variance covariance estimators only available for continous response models."))
        if isordered(getproperty(data, f.lhs.sym))
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
        y = getproperty(data, exogenous.lhs.sym)
        X = modelcols(exogenous.rhs, data)
        if isordered(y)
            X = X[:,2:end]
        end
    else
        y, X = modelcols(exogenous, data)
    end
    if !isa(iv.lhs, InterceptTerm)
        z, Z = modelcols(iv, data)
        z = isa(z, Tuple) ? hcat(z...) : convert(Matrix{Float64}, reshape(z, length(z), 1))
    else
        z = zeros(0, 0)
        Z = zeros(0, 0)
    end
	wts = isnothing(wts) ?
		  FrequencyWeights(ones(length(y))) :
		  FrequencyWeights(collect(wts))
    if isa(estimator, Type{<:RandomEffectsEstimator})
        panel = (panel,
                 [ findall(isequal(l), getproperty(data, panel)) for l ∈ levels(getproperty(data, panel)) ])
        time = (time, [ findall(isequal(l), getproperty(data, time)) for l ∈ levels(getproperty(data, time)) ])
        estimator = RandomEffectsEstimator(panel, time, X, y, z, Z, FrequencyWeights(wts))
    elseif isa(estimator, Type{<:BetweenEstimator})
        estimator = BetweenEstimator(panel, [ findall(isequal(l), getproperty(data, panel)) for l ∈ levels(getproperty(data, panel)) ])
    elseif isa(estimator, Type{<:EconometricModel}) || isa(estimator, Type{<:ContinuousResponse})
        estimator = ContinuousResponse(absorbed)
    end
    data, exogenous, iv, estimator, X, y, z, Z, wts
end

remove_intercept(f::AbstractString) =
    occursin(" ~ 1 + ", f) ? replace(f, " ~ 1 + " => " ~ ") : f
add_intercept(f::AbstractString) =
    !occursin(r" ~ -?[0-1] ?", f) ? replace(f, r"(^.*?) ~ " => s"\1 ~ 1 + ") : f
clean_lhs(obj::AbstractTerm) = string(obj)
clean_lhs(obj::FunctionTerm) = string(obj)[3:end - 1]
clean_rhs(obj::Tuple) = mapreduce(clean_rhs, (x, y) -> "$x + $y", obj)
clean_rhs(obj::AbstractTerm) = string(obj)
clean_rhs(obj::InteractionTerm) = mapreduce(clean_rhs, (x, y) -> "$x & $y", obj.terms)
clean_rhs(obj::FunctionTerm) = string(obj)[2:end]
clean_rhs(obj::FormulaTerm) = string("(", obj.lhs, " ~ ", clean_rhs(obj.rhs), ")")
clean_rhs(obj::FunctionTerm{typeof(absorb)}) = string(obj)[3:end - 1]
clean_fm(obj::EconometricsModel) =
    obj.f |>
    (f -> string(clean_lhs(f.lhs), " ~ ", clean_rhs(f.rhs))) |>
    (f -> isa(obj, EconometricModel{<:Union{RandomEffectsEstimator,
                                            OrdinalResponse}}) ?
          remove_intercept(f) : add_intercept(f)) |>
    (f -> "Formula: $f")
