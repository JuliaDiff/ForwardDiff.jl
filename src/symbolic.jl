### Code for forming the basis of a Symbolic package
### This code was initially written by Jonas Rauch
### I debugged it to work with the current dev version of Julia
### I will extend it further

ExprOrSymbol=Union(Expr, Symbol) 

zero(::Type{Expr}) = :(+0)

+(x::ExprOrSymbol, y::ExprOrSymbol)=:($x+$y)
+(x::ExprOrSymbol, y)= y==0 ? x : :($x+$y)
+(x, y::ExprOrSymbol)= x==0 ? y : :($x+$y)

-(x::ExprOrSymbol)=:(-$x)
-(x::ExprOrSymbol, y::ExprOrSymbol)=:($x-$y)
-(x::ExprOrSymbol, y)= y==0 ? x : :($x-$y)
-(x, y::ExprOrSymbol)= x==0 ? -y : :($x-$y)

*(x::ExprOrSymbol, y::ExprOrSymbol)=:($x*$y)
*(x::ExprOrSymbol, y)= y == 0 ? 0 : ( y == 1 ? x : :($x*$y)) 
*(x, y::ExprOrSymbol)= x == 0 ? 0 : ( x == 1 ? y : :($x*$y))

/(x::ExprOrSymbol, y::ExprOrSymbol)=:($x/$y)
/(x::ExprOrSymbol, y)= y == 0 ? NaN : ( y == 1 ? x : :($x/$y)) 
/(x, y::ExprOrSymbol)= x == 0 ? ( y == 0 ? NaN : 0 ) : :($x/$y)

D(E::Symbol, x::Symbol)=(E==x) ? 1 : 0
function D(E::Expr, x::Symbol)
    if E.head==:call
        f=E.args[1]
        args=E.args[2:end]
        if f == :+
            return +({D(ex,x) for ex=args}...)
        elseif f == :-
            return -({D(ex,x) for ex=args}...)
        elseif f == :*
            return +({ *(vcat(args[1:i-1], { D(args[i], x) }, args[i+1:end])...) 
              for i=1:length(args)}...)
        elseif f == :/
            return D(args[1], x)/args[2] - args[1]*D(args[2], x) / (args[2]*args[2])
        end
    end
    error("derivative unknown for ", E)
end
D(E::Array{Expr}, x::Symbol) = [D(ex, x) for ex=E]
#constants
D(E, x::Symbol)=0
#multiple derivatives
D(E, X::Array{Symbol, 1}) = [D(E, x) for x=X ]
D(E, X::Tuple) = [D(E, x) for x=X ]

#example
#h(x,y)=x+2*y+x^2*y
#E=h(:x,:y)
#H=@eval (x,y)->$E             # == h(x,y) 
#dhdx=@eval (x,y)->$(D(E, :x)) # == 1+2*x*y
#dhdy=@eval (x,y)->$(D(E, :y)) # == 2+x^2

#helper functions

#call D(h, (:x,:y), :x) to compute dhdx
D(F::Function, args::Tuple, x::Symbol) = begin
  E=D(F(args...), x)
  ArgExpr=Expr(:tuple, {arg for arg=args}, Any)
  FuncExpr=Expr(:->, {ArgExpr, E}, Any)
  @eval $FuncExpr
end 
#call Gradient(h, (:x,:y)) 
Gradient(F::Function, args::Tuple) = begin
  E=D(F(args...), args)
  ArgExpr=Expr(:tuple, {arg for arg=args}, Any)
  VectorExpr=Expr(:vcat, E, Any)
  FuncExpr=Expr(:->, {ArgExpr, VectorExpr}, Any)
  @eval $FuncExpr
end
#call D(h, 2, 1) to compute dhdx
D(F::Function, nargs::Int, xindex::Int) = begin
  args=gensym(nargs)
  x=args[xindex]
  D(F, args, x)
end
#call Gradient(h, 2) 
Gradient(F::Function, nargs) = begin
  args=gensym(nargs)
  Gradient(F, args)
end

