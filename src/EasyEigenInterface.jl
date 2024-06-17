module EasyEigenInterface

using Libdl: Libdl, dlext

using CxxWrap

abstract type AbstractEigenMatrix{T} <: AbstractMatrix{T} end
abstract type AbstractEigenVector{T} <: AbstractVector{T} end

const AbstractEigenVecOrMat{T} = Union{AbstractEigenVector{T}, AbstractEigenMatrix{T}} where T

@wrapmodule(
    () -> joinpath(pkgdir(@__MODULE__), "deps", "build", "lib", "libjl_easy_eigen_interface.$(dlext)"),
    :easy_eigen_interface,
    Libdl.RTLD_GLOBAL
)

Base.iterate(m::EigenMatrix, state=1) = state > length(m) ? nothing : (m[state], state+1)
Base.size(m::EigenMatrix) = (rows(m),cols(m))
Base.IndexStyle(::Type{<:EigenMatrix}) = IndexCartesian()

Base.iterate(m::EigenVector, state=1) = state > length(m) ? nothing : (m[state], state+1)
Base.size(m::EigenVector) = (rows(m),)
Base.IndexStyle(::Type{<:EigenVector}) = IndexCartesian()

export MatrixXd, MatrixXf
export VectorXd, VectorXf
export rows, cols, resize!

const MatrixXd = EigenMatrix{Float64}
const MatrixXf = EigenMatrix{Float32}

const VectorXd = EigenVector{Float64}
const VectorXf = EigenVector{Float32}

# constructor
function EigenMatrix{T}(m::AbstractMatrix{T}) where T <: Union{Float32, Float64}
    ret = EigenMatrix{T}(size(m, 1), size(m, 2))
    for (i, e) in enumerate(m)
        ret[i] = e
    end
    return ret
end

EigenMatrix{T}(o::EigenMatrix{T}) where T<:Union{Float32, Float64} = o

# constructor
function EigenVector{T}(v::AbstractVector{T}) where T <: Union{Float32, Float64}
    ret = EigenVector{T}(size(v, 1))
    for (i, e) in enumerate(v)
        ret[i] = e
    end
    return ret
end

EigenVector{T}(o::EigenVector{T}) where T<:Union{Float32, Float64} = o

function Base.convert(::Type{T1}, ev::EigenVector{T}) where {T1 <: MatrixXd, T <: Union{Float32, Float64}}
    r = size(ev, 1)
    em = EigenMatrix{T}(r, 1)
    for i in eachindex(ev)
        em[i, 1] = ev[i]
    end
    return em
end

Base.promote_rule(::Type{T1}, ::Type{T2}) where {T1 <: EigenVector, T2 <: EigenMatrix} = T2

function __init__()
    @initcxx
end

end # module EasyEigenInterface
