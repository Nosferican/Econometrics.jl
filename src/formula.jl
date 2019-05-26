function absorb end
function PID end
function TID end
function between end

function decompose(f::FormulaTerm, data::AbstractDataFrame, contrasts::Dict{Symbol})
    data = dropmissing(data[termvars(f)], disallowmissing = true)
    rhs = isa(f.rhs, Tuple) ? collect(f.rhs) : [f.rhs]
    absorbed = findall(t -> isa(t, FunctionTerm{typeof(absorb)}), rhs)
    if isempty(absorbed)
        absorbed = (Vector{Symbol}(), Vector{Vector{Vector{Int}}}())
    elseif length(absorbed) == 1
        absorbed = rhs[absorbed[1]]
        if isa(absorbed, AbstractVector)
            absorbed = unique(flatten(termvars(t) for t ∈ absorbed))
        else
            absorbed = termvars(absorbed)
        end
        absorbed = (absorbed,
                    [ [ findall(isequal(l), data[dim]) for l ∈ levels(data[dim]) ] for dim ∈ absorbed ])
    else
        throw(ArgumentError("There can only be at most one absorb term"))
    end
    effect = findall(t -> isa(t, FunctionTerm{typeof(between)}), rhs)
    if isempty(effect)
        effect = (Vector{Symbol}(), Vector{Vector{Vector{Int}}}())
    elseif length(effect) == 1
        effect = termvars(f.rhs[effect[1]])
        effect = (effect,
                  [ findall(isequal(l), data[effect[1]]) for l ∈ levels(data[effect[1]]) ])
    else
        throw(ArgumentError("There can only be at most one effect for the between estimator"))
    end
    pid = findall(t -> isa(t, FunctionTerm{typeof(PID)}), rhs)
    if isempty(pid)
        pid = (Vector{Symbol}(), Vector{Vector{Vector{Int}}}())
    elseif length(pid) == 1
        pid = termvars(f.rhs[pid[1]])[1]
        pid = ([pid], [ findall(isequal(l), data[pid]) for l ∈ levels(data[pid]) ])
    else
        throw(ArgumentError("There can only be one panel identifier"))
    end
    tid = findall(t -> isa(t, FunctionTerm{typeof(TID)}), rhs)
    if isempty(tid)
        tid = (Vector{Symbol}(), Vector{Vector{Vector{Int}}}())
    elseif length(tid) == 1
        tid = termvars(f.rhs[tid[1]])[1]
        tid = ([tid], [ findall(isequal(l), data[tid]) for l ∈ levels(data[tid]) ])
    else
        throw(ArgumentError("There can only be one time identifier"))
    end
    wts = findall(t -> isa(t, FunctionTerm{typeof(weights)}), rhs)
    if isempty(wts)
        wts = FrequencyWeights(zeros(0))
    elseif length(wts) == 1
        wts = data[termvars(f.rhs[wts[1]])[1]]
    else
        throw(ArgumentError("There can only be one weight variable"))
    end
    exo_rhs = filter(t -> !(isa(t, FormulaTerm) ||
                     isa(t, FunctionTerm{<:Union{<:typeof(absorb),
                                                 <:typeof(PID),
                                                 <:typeof(TID),
                                                 <:typeof(weights),
                                                 <:typeof(between)}})),
                    rhs)
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
    if isa(exogenous.lhs, CategoricalTerm)
        y = data[exogenous.lhs.sym]
        X = modelcols(exogenous.rhs, data)
        if isordered(y)
            X = X[:,2:end]
        end
    else
        y, X = modelcols(exogenous, data)
    end
    wts = isempty(wts) ? FrequencyWeights(Fill(1, size(y, 1))) : wts
    if !isa(iv.lhs, InterceptTerm)
        z, Z = modelcols(iv, data)
        z = isa(z, Tuple) ? hcat(z...) : z
    else
        z = zeros(0, 0)
        Z = zeros(0, 0)
    end
    data, f, exogenous, iv, absorbed, pid, tid, wts, effect, y, X, z, Z
end