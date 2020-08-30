"""
    Econometrics

    Econometrics in Julia.
"""
module Econometrics
using Base.Iterators: flatten
using CategoricalArrays: CategoricalValue, levels, levels!, isordered, ordered!
using Distributions:
    cdf,
    ccdf,
    Categorical,
    Chisq,
    FDist,
    logpdf,
    Logistic,
    Normal,
    pdf,
    TDist,
# Statistics
    mean,
    quantile,
    var
using FillArrays: Ones
using ForwardDiff: Dual, value
using LinearAlgebra:
    bunchkaufman,
    bunchkaufman!,
    cholesky!,
    diag,
    diagm,
    Diagonal,
    Hermitian,
    I,
    LowerTriangular,
    qr,
    UpperTriangular
using Optim: hessian!, optimize, minimizer, TwiceDifferentiable
using Parameters: @unpack, @pack!
using Printf: @sprintf
using StatsBase:
    aic, aicc, bic, harmmean, FrequencyWeights, CoefTable, ConvergenceException, Weights
using StatsFuns: softmax
using StatsModels:
    AbstractContrasts,
    AbstractTerm,
    apply_schema,
    CategoricalTerm,
    ConstantTerm,
    ContrastsMatrix,
    DummyCoding,
    @formula,
    FormulaTerm,
    FunctionTerm,
    InteractionTerm,
    InterceptTerm,
    MatrixTerm,
    modelcols,
    schema,
    terms,
    termvars
using TableOperations:
    select,
# Tables
    Tables,
    Tables.columntable,
    Tables.columns,
    Tables.eachcolumn,
    Tables.materializer,
    Tables.rows
import Base: show
import StatsBase:
    coef,
    coefnames,
    coeftable,
    confint,
    deviance,
    islinear,
    nulldeviance,
    loglikelihood,
    nullloglikelihood,
    score,
    nobs,
    dof,
    dof_residual,
    mss,
    rss,
    informationmatrix,
    vcov,
    stderror,
    weights,
    isfitted,
    fit,
    fit!,
    r2,
    adjr2,
    fitted,
    response,
    meanresponse,
    modelmatrix,
    leverage,
    residuals,
    predict,
    predict!,
    dof_residual,
    RegressionModel,
    params
import StatsModels: hasintercept, implicit_intercept
# Compat
if !@isdefined(isnothing)
    isnothing(::Any) = false
    isnothing(::Nothing) = true
end
if !@isdefined(ismissing)
    isnothing(::Any) = false
    isnothing(::Missing) = true
end
foreach(
    file -> include(joinpath(dirname(@__DIR__), "src", "$file.jl")),
    ["structs", "transformations", "formula", "main", "mlj-model-interface", "solvers", "statsbase", "wald"],
)
export @formula,
    DummyCoding,
    aic,
    aicc,
    bic,
    coef,
    coefnames,
    coeftable,
    confint,
    deviance,
    islinear,
    nulldeviance,
    loglikelihood,
    nullloglikelihood,
    nobs,
    dof,
    mss,
    rss,
    informationmatrix,
    vcov,
    stderror,
    weights,
    isfitted,
    fit,
    fit!,
    r2,
    adjr2,
    fitted,
    response,
    meanresponse,
    modelmatrix,
    leverage,
    residuals,
    predict,
    dof_residual,
    params,
    hasintercept,
    EconometricModel,
    EconometricsMLJModel,
    absorb,
    BetweenEstimator,
    RandomEffectsEstimator,
    ContinuousResponse,
    OIM,
    HC0,
    HC1,
    HC2,
    HC3,
    HC4
end
