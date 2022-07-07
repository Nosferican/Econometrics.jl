"""
    Econometrics

    Econometrics in Julia.
"""
module Econometrics
using Base.Iterators: flatten
using CategoricalArrays: CategoricalArrays, CategoricalValue, categorical, levels, levels!, isordered, ordered!
using Distributions:
    Distributions,
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
using FillArrays: FillArrays, Ones
using ForwardDiff: ForwardDiff, Dual, value
using LinearAlgebra:
    LinearAlgebra,
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
    ColumnNorm,
    UpperTriangular
using MLJModelInterface: MLJModelInterface, MLJModelInterface, Probabilistic
using Optim: Optim, hessian!, optimize, minimizer, TwiceDifferentiable
using Parameters: Parameters, @unpack, @pack!
using Printf: Printf, @sprintf
using StatsBase:
    StatsBase, aic, aicc, bic, harmmean, FrequencyWeights, CoefTable, ConvergenceException, Weights
using StatsFuns: StatsFuns, softmax
using StatsModels:
    StatsModels,
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
    TableOperations,
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
    responsename,
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
import MLJModelInterface: clean!

"""
    const DATAPATH::String = realpath(joinpath(pkgdir(Econometrics), "data"))
Return the path to the data directory for the Econometrics.jl module.
"""
const DATAPATH = realpath(joinpath(dirname(@__FILE__), "..", "data"))
foreach(
    file -> include(joinpath(dirname(@__DIR__), "src", "$file.jl")),
    ["structs", "transformations", "formula", "main", "mlj-model-interface", "solvers", "statsbase", "wald"],
    )
export @formula,
    DummyCoding,
    categorical,
    levels!,
    ordered!,
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
    responsename,
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
    HC4,
    levels!,
    Hermitian
end
