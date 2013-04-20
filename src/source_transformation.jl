import Calculus

export autodiff_transform

# This is extremely crude! It only works for scalar values.

# The idea here is to use Calculus.differentiate to do a symbolic
# differentiation of each line that needs it. If that fails, we could
# fall back to using the operator overloading approach or numerical
# approximations.
#
# Thesis on source transformation in Matlab:
# https://dspace.lib.cranfield.ac.uk/bitstream/1826/7298/1/KharchePhD2011.pdf
# The author (Rahul Kharche) used a hybrid source transformation and
# operator overloading approach for his Masters thesis (I can't find
# it).

function get_expr(m::LambdaStaticData)
    ## ast = Base.uncompress_ast(methods(fun).defs.func.code) 
    ast = Base.uncompress_ast(m) 
    arg = ast.args[1][1]
    vars = ast.args[2][1] # contains a list of variables that might be useful for dependency analysis.
    e = ast.args[3]
    e.head = :block
    return e, arg, vars
end


anyactive(s::Symbol, isactive::Dict{Symbol,Bool}) = get(isactive, s, false)

function anyactive(e::Expr, isactive::Dict{Symbol,Bool})
    for ea in e.args
        if anyactive(ea, isactive)
            return true
        end
    end
    return false
end


find_dependents(x, arg::Symbol, vars, isactive::Dict{Symbol,Bool}) = nothing

function find_dependents(e::Expr, arg::Symbol, vars, isactive::Dict{Symbol,Bool})
    if  e.head == :(=) && !isactive[e.args[1]] && anyactive(e, isactive)
        isactive[e.args[1]] = true 
    else
        for ea in e.args
            find_dependents(ea, arg, vars, isactive)
        end
    end
end


wrap_value(x) = x
wrap_value(s::Symbol) = :(value($s))

function wrap_value(e::Expr)
    for i in 1:length(e.args)
        e.args[i] = wrap_value(e.args[i])
    end
    return e
end


add_derivatives(x, var::Symbol) = nothing

function add_derivatives(e::Expr, var::Symbol, isactive::Dict{Symbol, Bool})
    if  e.head == :(=) && anyactive(e, isactive)
        # replace the assignment equation with an assignment including derivatives
        deriv = Calculus.differentiate(e.args[2], var)
        # need to convert `z` to `value(z)` where z is any symbol
        e.args[2] = :(ADForward($(e.args[2]), [$deriv]))
        wrap_value(e.args[2])
    else
        for ea in e.args
            add_derivatives(ea, var)
        end
    end
end


function autodiff_transform(f::Function, funname::Symbol, types)
    # This takes `f` and creates a new method `f_der` that calculates
    # the value and derivatives.
    m = methods(f, types)[1][3]
    e, arg, vars = get_expr(m)
    isactive = Dict{Symbol,Bool}(vars, fill(false, length(vars)))
    find_dependents(e, arg, vars, isactive)
    add_derivatives(e, arg, isactive)
    eval(m.module, :(function $funname($arg::ADForward) $e end))
end
autodiff_transform(f::Function, types) = autodiff_transform(f, methods(f).defs.func.code.name, types)
