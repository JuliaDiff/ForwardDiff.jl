using Calculus

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
    varsym = ast.args[1][1]
    # ast.args[2] contains a list of variables that might be useful for dependency analysis.
    e = ast.args[3]
    e.head = :block
    return e, varsym
end

function depends_on(e::Expr, var::Symbol)
    # this needs to look at chains of variable assignments
    return true
end

add_derivatives(x, var::Symbol) = nothing

function add_derivatives(e::Expr, var::Symbol)
    if  e.head == :(=) && depends_on(e, var)
        # replace the assignment equation with an assignment including derivatives
        deriv = differentiate(e.args[2], var) 
        e.args[2] = :(ADForward($(e.args[2]), [$deriv]))
    else
        for ea in e.args
            add_derivatives(ea, var)
        end
    end
end

function autodiff_transform(f::Function, types...)
    # This takes `f` and creates a new method `f_der` that calculates
    # the value and derivatives.
    m = methods(f, types)[1][3]
    e, var = get_expr(m)
    add_derivatives(e, var)
    newsym = symbol(string(f.env.name) * "_der")
    eval(m.module, :(function $newsym($var::ADForward) $e end))
end
    
