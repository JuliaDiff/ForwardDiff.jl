module SparseArraysExt

using ForwardDiff: Dual
using LinearAlgebra: LinearAlgebra, norm
using SparseArrays: SparseArrays, AbstractSparseMatrixCSC, SparseVectorUnion

# Resolve method ambiguities
for A in (:AbstractSparseMatrixCSC, :SparseVectorUnion)
    @eval begin
        function LinearAlgebra.norm(x::Union{$A{Dual{T,V,N}},$A{Complex{Dual{T,V,N}}}}, p::Real=2) where {T,V,N}
            return invoke(norm, Tuple{$A,Real}, x, p)
        end
        function LinearAlgebra.norm(x::$A{<:Number}, p::Dual)
            return invoke(norm, Tuple{$A,Real}, x, p)
        end
        function LinearAlgebra.norm(x::Union{$A{Dual{T,V,N}},$A{Complex{Dual{T,V,N}}}}, p::Dual) where {T,V,N}
            return invoke(norm, Tuple{$A,Real}, x, p)
        end
    end
end

end # module
