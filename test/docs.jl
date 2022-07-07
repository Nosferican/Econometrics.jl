# Documentation
@testset "Documentation" begin
    using Documenter, Econometrics, Weave, StatsBase, StatsAPI

    prefix = ".."
    # prefix = "."

    for file in filter!(file -> endswith(file, ".jmd"), readdir(joinpath(prefix, "docs", "jmd"), join = true))
        weave(
            file,
            out_path = joinpath(prefix, "docs", "src"),
            doctype = "github"
            )
    end

    DocMeta.setdocmeta!(Econometrics,
                       :DocTestSetup,
                       :(using Econometrics, Documenter, CSV, RDatasets, StatsBase;
                         ENV["COLUMNS"] = 120;
                         ENV["LINES"] = 30;),
                       recursive = true)
    # doctest(Econometrics, fix = true)
    makedocs(sitename = "Econometrics",
             format = Documenter.HTML(assets = [joinpath("assets", "custom.css")]),
             modules = [Econometrics, StatsAPI],
             pages = [
                 "Introduction" => "index.md",
                 "Getting Started" => "getting_started.md",
                 "Estimators" => "estimators.md",
                 "API" => "api.md",
                 ],
             source = joinpath(prefix, "docs", "src"),
             build = joinpath(prefix, "docs", "build"),
             )
    @test true
end
