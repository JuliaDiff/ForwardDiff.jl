using ForwardDiff, JuliaC, Test
include("functions.jl")

@testset "juliac --trim=safe" begin
    mktempdir() do dir
        image = ImageRecipe(output_type = "--output-exe", file = joinpath(@__DIR__, "app.jl"),
                            trim_mode = "safe", project = @__DIR__)
        compile_products(image)
        exe = joinpath(dir, "app")
        link_products(LinkRecipe(image_recipe = image, outname = exe))
        # below and above the default chunk threshold
        for n in (3, 20)
            expected = mapreduce(vec, vcat, results(inputs(n)))
            @test parse.(Float64, readlines(`$exe $n`)) ≈ expected
        end
    end
end
