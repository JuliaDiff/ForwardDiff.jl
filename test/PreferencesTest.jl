module PreferencesTest

using Test
using ForwardDiff

const DEPRECATION_MESSAGE = "The `nansafe_mode` preference of ForwardDiff is deprecated. " *
    "NaN-safe mode is enabled by default since ForwardDiff 1.5.0, and the preference to disable it is now called `unsafe_mode`. " *
    "For now, `nansafe_mode = false` still disables NaN-safe mode unless `unsafe_mode` is set. " *
    "Please remove the `nansafe_mode` preference with `Preferences.delete_preferences!(ForwardDiff, \"nansafe_mode\"; force = true)` and restart Julia."

@testset "deprecated `nansafe_mode` preference" begin
    # ForwardDiff is an extra so that it is loaded from the test environment
    # `__clear__` discards preferences inherited from the test environment
    # Julia 1.10 hashes only the values of preferences for precompile caches, so we set
    # `default_chunk_threshold` (to its default) to avoid reusing the cache of the test process
    for (preferences, unsafe_mode_enabled) in (
        ("nansafe_mode = false\n__clear__ = [\"unsafe_mode\"]", true),
        ("nansafe_mode = false\nunsafe_mode = false", false),
        ("nansafe_mode = true\n__clear__ = [\"unsafe_mode\"]", false),
    )
        mktempdir() do dir
            write(joinpath(dir, "Project.toml"), """
                [extras]
                ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"

                [preferences.ForwardDiff]
                default_chunk_threshold = 12
                $preferences
                """)
            cmd = `$(Base.julia_cmd()) --startup-file=no --depwarn=no --project=$dir -e 'using ForwardDiff; print(ForwardDiff.UNSAFE_MODE_ENABLED)'`
            load_path = join(["@", Base.load_path()...], Sys.iswindows() ? ';' : ':')
            out = IOBuffer()
            err = IOBuffer()
            run(pipeline(addenv(cmd, "JULIA_LOAD_PATH" => load_path); stdout = out, stderr = err))
            @test String(take!(out)) == string(unsafe_mode_enabled)
            @test occursin(DEPRECATION_MESSAGE, String(take!(err)))
        end
    end
end

end # module
