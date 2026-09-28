module PkgB

using ForwardDiff: Dual, Tag
using TagDefs: FA, FB, TA, TB

mix() = Dual{TA}(1.0, 1.0) * Dual{TB}(2.0, 1.0)

Tag(FB(), Float64)
Tag(FA(), Float64)
mix()

end
