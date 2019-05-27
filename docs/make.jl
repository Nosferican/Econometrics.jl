using Documenter, Econometrics

makedocs(format = Documenter.HTML(assets = ["assets/custom.css"]),
         modules = [Econometrics],
         sitename = "Econometrics.jl"
         )
deploydocs(repo = "github.com/Nosferican/Econometrics.jl.git")
