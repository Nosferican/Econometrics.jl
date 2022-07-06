using Econometrics, Test
using Econometrics: ConvergenceException, Hermitian
using CSV, RDatasets, CategoricalArrays
using MLJBase: MLJBase
using MLJModelInterface: MLJModelInterface

for file in ["exceptions", "linear_models", "mlogit", "ologit", "formula_display", "mlj", "docs"]
    include("$file.jl")
end
