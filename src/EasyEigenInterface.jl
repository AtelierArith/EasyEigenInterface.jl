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

Base.iterate(m::AbstractEigenMatrix, state=1) = state > length(m) ? nothing : (m[state], state+1)
Base.size(m::AbstractEigenMatrix) = (rows(m),cols(m))
Base.IndexStyle(::Type{<:AbstractEigenMatrix}) = IndexCartesian()

Base.iterate(m::AbstractEigenVector, state=1) = state > length(m) ? nothing : (m[state], state+1)
Base.size(m::AbstractEigenVector) = (rows(m),)
Base.IndexStyle(::Type{<:AbstractEigenVector}) = IndexCartesian()

export MatrixXd, MatrixXf
export VectorXd, VectorXf

export Matrix2d, Matrix2f
export Vector2d, Vector2f

export Matrix3d, Matrix3f
export Vector3d, Vector3f

export Matrix4d, Matrix4f
export Vector4d, Vector4f

export rows, cols, resize!

const MatrixXd = EigenMatrixX{Float64}
const MatrixXf = EigenMatrixX{Float32}

const VectorXd = EigenVectorX{Float64}
const VectorXf = EigenVectorX{Float32}

const Matrix2d = EigenMatrix2{Float64}
const Matrix2f = EigenMatrix2{Float32}

const Matrix3d = EigenMatrix3{Float64}
const Matrix3f = EigenMatrix3{Float32}

const Matrix4d = EigenMatrix4{Float64}
const Matrix4f = EigenMatrix4{Float32}

const Vector2d = EigenVector2{Float64}
const Vector2f = EigenVector2{Float32}

const Vector3d = EigenVector3{Float64}
const Vector3f = EigenVector3{Float32}

const Vector4d = EigenVector4{Float64}
const Vector4f = EigenVector4{Float32}

# constructor
function EigenMatrixX{T}(m::AbstractMatrix{T}) where T <: Union{Float32, Float64}
    ret = EigenMatrixX{T}(size(m, 1), size(m, 2))
    for (i, e) in enumerate(m)
        ret[i] = e
    end
    return ret
end

EigenMatrixX{T}(o::EigenMatrixX{T}) where T<:Union{Float32, Float64} = o

# constructor
function EigenVectorX{T}(v::AbstractVector{T}) where T <: Union{Float32, Float64}
    ret = EigenVectorX{T}(size(v, 1))
    for (i, e) in enumerate(v)
        ret[i] = e
    end
    return ret
end

EigenVectorX{T}(o::EigenVectorX{T}) where T<:Union{Float32, Float64} = o

macro staticsizedconstructor(typename, N)

    A = esc(Symbol(:Abstract, typename)) # AbstractMatrix or AbstractVector

    name = esc(Symbol(:Eigen, typename, N))
    quote
        function $(name){T}(a::$(A){T}) where T <: Union{Float32, Float64}
            ret = $(name){T}()
            for i in 1:ndims(a)
                size(a, i) == $(N) || throw(ArgumentError("size mismatch"))
            end

            for (i, e) in enumerate(a)
                ret[i] = e
            end
            return ret
        end

        $(name){T}(o::$(name){T}) where T<:Union{Float32, Float64} = o
    end
end

for N in [2,3,4]
    name = :Matrix
    @eval @staticsizedconstructor $(name) $N

    name = :Vector
    @eval @staticsizedconstructor $(name) $N
end

function Base.convert(::Type{M}, ev::AbstractEigenVector{T}) where {M <: EigenMatrixX, T <: Union{Float32, Float64}}
    r = size(ev, 1)
    em = M(r, 1)
    for i in eachindex(ev)
        em[i, 1] = ev[i]
    end
    return em
end

Base.promote_rule(::Type{T1}, ::Type{T2}) where {T1 <: AbstractEigenVector, T2 <: AbstractEigenMatrix} = T2

function __init__()
    @initcxx
end

end # module EasyEigenInterface
