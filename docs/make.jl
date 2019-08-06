using Documenter, Weave, Econometrics
using StatsBase

for file ∈ readdir(joinpath(dirname(pathof(Econometrics)), "..", "docs", "jmd"))
      weave(joinpath(dirname(pathof(Econometrics)), "..", "docs", "jmd", file),
            out_path = joinpath(dirname(pathof(Econometrics)), "..", "docs", "src"),
            doctype = "github")
end

makedocs(format = Documenter.HTML(assets = ["assets/custom.css"]),
         modules = [Econometrics, StatsBase],
         sitename = "Econometrics.jl",
         pages = ["Introduction" => "index.md",
                  "Getting Started" => "getting_started.md",
                  "Estimators" => "estimators.md"]
    )

deploydocs(repo = "github.com/Nosferican/Econometrics.jl.git")
