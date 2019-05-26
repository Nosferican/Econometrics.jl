using Documenter, Econometrics

makedocs(format = Documenter.HTML(assets = ["documenter.css"]),
         modules = [Econometrics],
         sitename = "Econometrics.jl"
         )
deploydocs(repo = "github.com/Nosferican/Econometrics.jl.git")
