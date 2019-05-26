"""
    Econometrics

    Econometrics in Julia.
"""
module Econometrics
    using Base.Iterators: flatten
    using LinearAlgebra: bunchkaufman, bunchkaufman!, cholesky!, diag, diagm, Diagonal,
                         Hermitian, I, LowerTriangular, qr, UpperTriangular
    using DataFrames: AbstractDataFrame, categorical!, DataFrame, dropmissing,
                      # CategoricalArrays
                      AbstractCategoricalVector, categorical, levels, isordered
    using Distributions: cdf, ccdf, Categorical, Chisq, FDist, logpdf, Logistic, Normal,
                         pdf, TDist,
                         # Statistics
                         mean, quantile, var,
                         # StatsFuns
                         softmax
    using FillArrays: Fill
    using Optim: hessian!, optimize, minimizer, TwiceDifferentiable,
                 # ForwardDiff
                 ForwardDiff.Dual, ForwardDiff.value
    using Parameters: @unpack
    using Printf: @sprintf
    using StatsModels: AbstractContrasts, AbstractTerm, apply_schema, CategoricalTerm,
                       ConstantTerm, ContrastsMatrix, DummyCoding, @formula, FormulaTerm,
                       FunctionTerm, InterceptTerm, MatrixTerm, modelcols, schema, terms,
                       termvars,
                       # StatsBase
                       aic, aicc, bic, harmmean, FrequencyWeights, CoefTable,
                       ConvergenceException
    import Base: show
    import StatsModels: hasintercept, implicit_intercept,
                        # StatsBase
                        coef, coefnames, coeftable, confint, deviance, islinear, nulldeviance,
                        loglikelihood, nullloglikelihood, score, nobs, dof, dof_residual,
                        mss, rss, informationmatrix, vcov, stderror, weights, isfitted, fit,
                        fit!, r2, adjr2, fitted, response, meanresponse, modelmatrix, leverage,
                        residuals, predict, predict!, dof_residual, RegressionModel,
                        StatsBase.params

    foreach(file -> include(joinpath(dirname(@__DIR__), "src", "$file.jl")),
            ["structs", "transformations", "formula", "solvers", "main", "statsbase", "wald"])

    export categorical!, DataFrame, @formula, DummyCoding, aic, aicc, bic, coef, coefnames,
           coeftable, confint, deviance, islinear, nulldeviance, loglikelihood,
           nullloglikelihood, score, nobs, dof, mss, rss, informationmatrix, vcov, stderror,
           weights, isfitted, fit, fit!, r2, adjr2, fitted, response, meanresponse,
           modelmatrix, leverage, residuals, predict, predict!, dof_residual,
           params, EconometricModel, absorb, between, PID, TID
end
