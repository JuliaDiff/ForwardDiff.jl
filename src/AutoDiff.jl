module AutoDiff

  include("dual.jl")
  include("ad_jonas_rauch.jl")
  include("source_transformation.jl")

  export
    # Export Dual type and dual functions
    Dual,
    Dual128,
    Dual64,
    DualPair,
    real,
    imag,
    dual,
    dual128,
    dual64,
    isdual,
    real_valued,
    integer_valued,
    isfinite,
    reim,
    dual_show,
    show,
    showcompact,
    read,
    write,
    ==,
    isequal,
    hash,
    conj,
    abs,
    abs2,
    inv,
    +,
    -,
    *,
    /,
    sqrt,
    cbrt,
    ^,
    exp,
    log,
    log2,
    log10,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh

end # module Autodiff
