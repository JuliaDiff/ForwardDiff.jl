module GradientTest

using Base.Test
using ForwardDiff
using ForwardDiff: default_value, KWARG_DEFAULTS

########################
# @gradient/@gradient! #
########################

const ALL_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :all))})
const CHUNK_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :chunk))})
const INPUT_LENGTH_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :input_length))})
const MULTITHREAD_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :multithread))})
const OUTPUT_MUTATES_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :output_mutates))})

@test macroexpand(:(ForwardDiff.@gradient(sin))) == :(ForwardDiff.gradient(sin, $ALL_DEFAULT, $CHUNK_DEFAULT, $INPUT_LENGTH_DEFAULT, $MULTITHREAD_DEFAULT, $OUTPUT_MUTATES_DEFAULT))
@test macroexpand(:(ForwardDiff.@gradient(sin; output_mutates=1, all=2, multithread=3, chunk=4, input_length=5))) == :(ForwardDiff.gradient(sin, Val{2}, Val{4}, Val{5}, Val{3}, Val{1}))
@test macroexpand(:(ForwardDiff.@gradient(sin, chunk=1, output_mutates=2))) == :(ForwardDiff.gradient(sin, $ALL_DEFAULT, Val{1}, $INPUT_LENGTH_DEFAULT, $MULTITHREAD_DEFAULT, Val{2}))

@test macroexpand(:(ForwardDiff.@gradient(sin, x))) == :(ForwardDiff.gradient(sin, x, $ALL_DEFAULT, $CHUNK_DEFAULT, $INPUT_LENGTH_DEFAULT, $MULTITHREAD_DEFAULT))
@test macroexpand(:(ForwardDiff.@gradient(sin, x, input_length=1, all=2, multithread=3, chunk=4))) == :(ForwardDiff.gradient(sin, x, Val{2}, Val{4}, Val{1}, Val{3}))
@test macroexpand(:(ForwardDiff.@gradient(sin, x; chunk=1, multithread=2))) == :(ForwardDiff.gradient(sin, x, $ALL_DEFAULT, Val{1}, $INPUT_LENGTH_DEFAULT, Val{2}))

@test macroexpand(:(ForwardDiff.@gradient!(sin, output, x))) == :(ForwardDiff.gradient!(sin, output, x, $ALL_DEFAULT, $CHUNK_DEFAULT, $INPUT_LENGTH_DEFAULT, $MULTITHREAD_DEFAULT))
@test macroexpand(:(ForwardDiff.@gradient!(sin, output, x, input_length=1, all=2, multithread=3, chunk=4))) == :(ForwardDiff.gradient!(sin, output, x, Val{2}, Val{4}, Val{1}, Val{3}))
@test macroexpand(:(ForwardDiff.@gradient!(sin, output, x; chunk=1, multithread=2))) == :(ForwardDiff.gradient!(sin, output, x, $ALL_DEFAULT, Val{1}, $INPUT_LENGTH_DEFAULT, Val{2}))

end # module
