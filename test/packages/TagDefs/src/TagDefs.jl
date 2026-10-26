module TagDefs

using ForwardDiff: Tag

struct FA end
struct FB end
const TA = Tag{FA,Float64}
const TB = Tag{FB,Float64}

end
