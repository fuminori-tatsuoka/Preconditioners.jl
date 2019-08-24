
"""
    SORPreconditioner(d, E, F, ω, m)

the SOR iteration is

[1^]: Y. Saad, Sparse Linear Systems
"""
mutable struct SORPreconditioner{T, S <: AbstractSparseMatrix{T}} <: AbstractPreconditioner
    L::S # ``D - ωE``
    U::S # ``ωF + (1-ω)D``
    ω::Float64 # relaxation parameter
    m::Int # the number of inner iterations
end


function SORPreconditioner(A::AbstractMatrix, ω::T, m::Int=20) where {T <: Real}
    n = size(A, 1)
    d = diag(A)
    ω_f64 = convert(Float64, ω)
    L = ω_f64 * tril(A, -1) + spdiagm(0 => d)
    U = -ω_f64 * triu(A, 1) + (1-ω_f64) * spdiagm(0 => d)
    return SORPreconditioner{eltype(d), typeof(L)}(L, U, ω_f64, m)
end


function UpdatePreconditioner!(M::SORPreconditioner, K::AbstractMatrix)
    M.d .= diag(K, 0)
    M.E = -tril(K, -1)
    M.F = -triu(K, 1)
    return M
end


function ldiv!(y, C::SORPreconditioner, b)
    y .*= 0.0
    ωb = C.ω * b
    for i = 1:C.m
        y .= C.U * y .+ ωb
        y .= C.L \ y
    end
    return y
end


function (\)(C::SORPreconditioner, b)
    y = zero(b)
    ωb = C.ω * b
    for i = 1:C.m
        y .= C.U * y .+ ωb
        y .= C.L \ y
    end
    return y
end
