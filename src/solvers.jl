@views function fit(estimator::LinearModelEstimators,
					X::AbstractMatrix,
					y::AbstractVector,
					z::AbstractMatrix,
					Z::AbstractMatrix,
					wts::AbstractVector)
	if !isa(wts, Weights)
        wts = FrequencyWeights(wts)
    else
        isa(wts, FrequencyWeights) || throw(ArgumentError("Only frequency weights are supported."))
    end
    X = transform(estimator, X, wts)
    y = transform(estimator, y, wts)
    z = transform(estimator, z, wts)
    Z = transform(estimator, Z, wts)
    wts = transform(estimator, FrequencyWeights(wts))
    if !isempty(z)
        Z̃ = hcat(X, Z)
        F = bunchkaufman(Hermitian(Z̃' * Diagonal(wts) * Z̃), true, check = false)
        bkr = count(x -> abs(x) ≥ √eps(), F.D)
		if bkr < size(F, 2)
            lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
            count(x -> lin_ind > size(X, 2) + 1) ≥ size(z, 2) ||
                throw(ArgumentError("Insufficient number of instruments."))
            Z̃ = Z̃[:,lin_ind]
            F = bunchkaufman(Hermitian(Z̃' * Diagonal(wts) * Z̃), true)
        else
            lin_ind = collect(1:size(F, 2))
        end
        γ = F \ (Z̃' * Diagonal(wts) * z)
        X̃ = hcat(X, Z̃ * γ)
    else
        X̃ = X
    end
    F = bunchkaufman(Hermitian(X̃' * Diagonal(wts) * X̃), true, check = false)
    bkr = count(x -> abs(x) ≥ √eps(), F.D)
    if bkr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
        X̃ = convert(Matrix{Float64}, X̃[:,lin_ind])
        F = bunchkaufman(Hermitian(X̃' * Diagonal(wts) * X̃), true)
    else
        lin_ind = collect(1:size(F, 2))
    end
    β = F \ (X̃' * Diagonal(wts) * y)
    Ψ = Hermitian(inv(F))
    ŷ = isempty(z) ? X̃ * β : hcat(X, z)[:,lin_ind] * β
	X̃, y, β, Ψ, ŷ, wts, lin_ind
end
@views function fit!(obj::EconometricModel{<:LinearModelEstimators})
	@unpack estimator, X, y, z, Z, wts = obj
    if !isa(wts, Weights)
        wts = FrequencyWeights(wts)
    else
        isa(wts, FrequencyWeights) || throw(ArgumentError("Only frequency weights are supported."))
    end
    X = transform(estimator, X, wts)
    y = transform(estimator, y, wts)
    z = transform(estimator, z, wts)
    Z = transform(estimator, Z, wts)
    wts = transform(estimator, FrequencyWeights(wts))
    if !isempty(z)
        Z̃ = hcat(X, Z)
        F = bunchkaufman(Hermitian(Z̃' * Diagonal(wts) * Z̃), true, check = false)
        bkr = count(x -> abs(x) ≥ √eps(), F.D)
		if bkr < size(F, 2)
            lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
            count(x -> lin_ind > size(X, 2) + 1) ≥ size(z, 2) ||
                throw(ArgumentError("Insufficient number of instruments."))
            Z̃ = Z̃[:,lin_ind]
            F = bunchkaufman(Hermitian(Z̃' * Diagonal(w) * Z̃), true)
        else
            lin_ind = collect(1:size(F, 2))
        end
        γ = F \ (Z̃' * Diagonal(wts) * z)
        X̃ = hcat(X, Z̃ * γ)
    else
        X̃ = X
    end
    F = bunchkaufman(Hermitian(X̃' * Diagonal(wts) * X̃), true, check = false)
    bkr = count(x -> abs(x) ≥ √eps(), F.D)
    if bkr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
        X̃ = convert(Matrix{Float64}, X̃[:,lin_ind])
        F = bunchkaufman(Hermitian(X̃' * Diagonal(wts) * X̃), true)
    else
        lin_ind = collect(1:size(F, 2))
    end
    β = F \ (X̃' * Diagonal(wts) * y)
    Ψ = Hermitian(inv(F))
    ŷ = isempty(z) ? X̃ * β : hcat(X, z)[:,lin_ind] * β
	X = X̃
	wts = FrequencyWeights(collect(wts))
	@pack! obj = X, y, β, Ψ, ŷ, wts
	obj.vars = (obj.vars[1], obj.vars[2][lin_ind])
	obj
end
@views function fit!(obj::EconometricModel{<:ContinuousResponse})
	@unpack estimator, X, y, z, Z, wts = obj
    if !isa(wts, Weights)
        wts = FrequencyWeights(wts)
    else
        isa(wts, FrequencyWeights) || throw(ArgumentError("Only frequency weights are supported."))
    end
    y₀ = y
    X = transform(estimator, X, wts)
    y = transform(estimator, y, wts)
    z = transform(estimator, z, wts)
    Z = transform(estimator, Z, wts)
    w = transform(estimator, FrequencyWeights(wts))
    if !isempty(z)
        Z̃ = hcat(X, Z)
        F = bunchkaufman(Hermitian(Z̃' * Diagonal(w) * Z̃), true, check = false)
        bkr = count(x -> abs(x) ≥ √eps(), F.D)
		if bkr < size(F, 2)
            lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
            F₀ = bunchkaufman(Hermitian(X' * Diagonal(w) * X), true, check = false)
            num_of_linear_independent_exogenous_vars = count(x -> abs(x) ≥ √eps(), F₀.D)
            (length(lin_ind) - num_of_linear_independent_exogenous_vars) ≥ size(z, 2) ||
                throw(ArgumentError("Insufficient number of instruments."))
            Z̃ = Z̃[:,lin_ind]
            F = bunchkaufman(Hermitian(Z̃' * Diagonal(w) * Z̃), true)
        else
            lin_ind = collect(1:size(F, 2))
        end
        γ = F \ (Z̃' * Diagonal(w) * z)
        X̃ = hcat(X, Z̃ * γ)
    else
        X̃ = X
    end
    F = bunchkaufman(Hermitian(X̃' * Diagonal(w) * X̃), true, check = false)
    bkr = count(x -> abs(x) ≥ √eps(), F.D)
    if bkr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
        X̃ = convert(Matrix{Float64}, X̃[:,lin_ind])
        F = bunchkaufman(Hermitian(X̃' * Diagonal(w) * X̃), true)
    else
        lin_ind = collect(1:size(F, 2))
    end
    β = F \ (X̃' * Diagonal(w) * y)
    Ψ = Hermitian(inv(F))
    ŷ = isempty(z) ? X̃ * β : hcat(X, z)[:,lin_ind] * β
    û = y - ŷ
    ŷ .= y₀ .- û
	X = X̃
	y = y₀
	@pack! obj = X, y, β, Ψ, ŷ
	obj.vars = (obj.vars[1], obj.vars[2][lin_ind])
	obj
end
"""
    obtain_Ω(A::AbstractMatrix{<:Real},
             μ::AbstractMatrix{<:Real},
             ω::FrequencyWeights)
Obtain Ω for a multinomial regression by building the matrix by blocks.
"""
@views function obtain_Ω(A, μ, wts)
  Aᵀ = transpose(A)
  k = size(μ, 2)
  p = size(A, 2)
  Σ = zeros(p * (k - 1), p * (k - 1))
  for (idx₀, col) ∈ enumerate(1:p:(p * (k - 1)))
    for (idx₁, row) ∈ enumerate(col:p:(p * (k - 1)))
      rows = col:col + p - 1
      cols = row:row + p - 1
      if col == row
        Σ[rows, cols] = Aᵀ * Diagonal(μ[:,idx₀ + 1] .* (1 .- μ[:,idx₀ + 1]) .* wts) * A
      else
        Σ[rows, cols] = -(Aᵀ * Diagonal(μ[:,idx₀ + 1] .* μ[:,idx₁ + 1] .* wts) * A)
      end
    end
  end
  Hermitian(Σ)
end
@views function fit!(obj::EconometricModel{<:NominalResponse})
	@unpack estimator, X, y, z, Z, wts = obj
	@unpack categories = estimator
    @assert isempty(z) && isempty(Z) "Nominal response models can only contain exogenous features"
    b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
    F = qr(X, ColumnNorm())
    qrr = count(x -> abs(x) ≥ √eps(), diag(F.R))
    if qrr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
        X = convert(Matrix{Float64}, X[:,lin_ind])
        F = qr(X, ColumnNorm())
    else
        lin_ind = collect(1:size(F, 2))
    end
    Q = Matrix(F.Q)
    Q⊤ = transpose(Q)
    m, p = size(F)
    k = size(b, 2)
    η = zeros(m, k)
    μ, μ′, w = zero(η), zero(η), zero(η)
    β = zeros(p, k)
    ℓℓ = [Inf, Inf, Inf]
    max_iteration = 250
    tol = 1e-9
    converged = false
    for iteration ∈ 1:max_iteration
      converged && break
      for row ∈ 1:m
        μ[row,:] .= softmax(η[row,:])
      end
      μ′ .= max.(μ .* (one(Float64) .- μ), √eps())
      w .= wts .* μ′
      ℓℓ[1] = ℓℓ[2]
      ℓℓ[2] = sum(wts[idx[1]] * logpdf(Categorical(collect(μ[idx[1],:])), idx[2]) for idx in findall(b))
      if isone(iteration)
          ℓℓ₀ = ℓℓ[2]
      end
      η .+= (b .- μ) ./ μ′
      for idx ∈ 2:size(b, 2)
        C = cholesky!(Hermitian(Q⊤ * Diagonal(w[:,idx]) * Q)).factors
        β[:,idx] = LowerTriangular(transpose(C)) \ (Q⊤ * Diagonal(w[:,idx]) * η[:,idx])
        β[:,idx] = UpperTriangular(C) \ β[:,idx]
      end
      η .= Q * β
      ℓℓ[3] = abs(ℓℓ[2] - ℓℓ[1])
      converged = ℓℓ[3] < 1e-9 || iszero(ℓℓ[1])
    end
    converged || throw(ConvergenceException(max_iteration, ℓℓ[3], tol))
    β .= F \ η
    Ψ = Hermitian(inv(bunchkaufman!(obtain_Ω(X, μ, wts))))
    ŷ = X * β
    β = collect(vec(β)[size(X, 2) + 1:end])
	@pack! obj = X, y, β, Ψ, ŷ
	obj.vars = (obj.vars[1], obj.vars[2][lin_ind])
	obj
end
@views function fit!(obj::EconometricModel{<:OrdinalResponse})
	@unpack estimator, X, y, z, Z, wts = obj
	@unpack categories = estimator
    @assert isempty(z) && isempty(Z) "Ordinal response models can only contain exogenous features"
    F = qr(X, ColumnNorm())
    qrr = count(x -> abs(x) ≥ √eps(), diag(F.R))
    if qrr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:bkr]])
        X = convert(Matrix{Float64}, X[:,lin_ind])
    else
        lin_ind = collect(1:size(F, 2))
    end
    m, n = size(X)
    xs = 1:n
    ks = length(categories) - 1
    ζs = n + 1:(n + ks)
    δ₀ = mapreduce(x -> (1:ks .== x - 1)', vcat, y)
    δ₁ = mapreduce(x -> (1:ks .== x)', vcat, y)
    η = Vector{Union{Float64,Dual}}(undef, m)
    ζ = vcat(-Inf, similar(η, ks), Inf)
    y₁ = y .+ 1
    Y₀ = similar(η)
    Y₁ = similar(η)
    F₀ = similar(η)
    F₁ = similar(η)
    f₀ = similar(η)
    f₁ = similar(η)
    pr = similar(η)
    θ₁ = ones(ks)
    function f(β)
      η .= X * β[xs]
      ζ[2:end - 1] .= cumsum(vcat(β[n + 1], exp.(β[n + 2:end])))
      Y₀ .= max.(-100, ζ[y] - η)
      Y₁ .= min.(100, ζ[y₁] - η)
      F₀ .= cdf.(Logistic(), Y₀)
      F₁ .= cdf.(Logistic(), Y₁)
      pr .= F₁ .- F₀
      -sum(w * log(p) for (w, p) ∈ zip(wts, pr))
    end
    function g!(G, β)
      ζ[2:end - 1] .= cumsum(vcat(β[n + 1], exp.(β[n + 2:end])))
      η .= X * β[xs]
      Y₀ .= max.(-100, ζ[y] - η)
      Y₁ .= min.(100, ζ[y₁] - η)
      F₀ .= cdf.(Logistic(), Y₀)
      F₁ .= cdf.(Logistic(), Y₁)
      pr .= F₁ .- F₀
      f₀ .= pdf.(Logistic(), Y₀)
      f₁ .= pdf.(Logistic(), Y₁)
      G[xs] .= X' * (wts .* (f₁ .- f₀) ./ pr)
      g2 = - (δ₁ .* f₁ .- δ₀ .* f₀)' * (wts ./ pr)
      θ₁[2:end] .= exp.(β[ζs[2:end]])
      G[ζs] .= diagm((d => θ₁[1:ks - d] for d ∈ 0:ks - 1)...) * g2
    end
    β₀ = vcat(zero(xs), log.(1:ks))
    td = TwiceDifferentiable(f, g!, β₀, autodiff = :forwarddiff)
    β = optimize(td, β₀) |> minimizer
    H = hessian!(td, β)
    Ψ = inv(bunchkaufman!(Hermitian(H)))
    A = Matrix(1.0I, size(Ψ)...)
    A[ζs, ζs] .= diagm((d => θ₁[1:ks + d] for d ∈ 0:-1:1 - ks)...)
    Ψ = Hermitian(A * Ψ * A')
    ŷ = X * β[1:size(X, 2)]
    β[n + 1:end] .= cumsum(vcat(β[n + 1], exp.(β[n + 2:end])))
	@pack! obj = X, y, β, Ψ, ŷ
	obj.vars = (obj.vars[1], obj.vars[2][lin_ind .+ 1])
	obj
end
