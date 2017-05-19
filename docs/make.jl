using Documenter, ForwardDiff

makedocs(modules=[ForwardDiff],
         doctest = false,
         format = :html,
         sitename = "ForwardDiff",
         pages = ["User Documentation" => [
                    "Introduction" => "user/index.md",
                    "Limitations of ForwardDiff" => "user/limitations.md",
                    "Differentiation API" => "user/api.md",
                    "Advanced Usage Guide" => "user/advanced.md",
                    "Upgrading from Older Versions" => "user/upgrade.md"],
                  "Developer Documentation" => [
                    "How ForwardDiff Works" => "dev/how_it_works.md",
                    "How to Contribute" => "dev/contributing.md"]])

deploydocs(repo = "github.com/JuliaDiff/ForwardDiff.jl.git",
           osname = "linux",
           julia = "0.6",
           target = "build",
           deps = nothing,
           make = nothing)
