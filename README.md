# EasyEigenInterface.jl

This Julia package wraps some data types in `Eigen`, a C++ template library for linear algebra such as `MatrixXd`, `VectorXd`.

The design is strongly inspired by the work of `@barche` [JuliaCon 2020 workshop on CxxWrap.jl](https://github.com/barche/cxxwrap-juliacon2020/tree/master).

## Usage

```julia
using EasyEigenInterface

x = Float64[1 2 3; 4 5 6]
@assert MatrixXd(MatrixXd(x)) == MatrixXd(x) == x

m = MatrixXd(x)
@assert rows(m) == 2
@assert cols(m) == 3
jlm = Matrix{Float64}(undef, 2, 3)
jlm .= m
@assert jlm == m

resize!(m, 3, 2)
@assert rows(m) == 3
@assert cols(m) == 2
```

I believe this may be helpful for those interested in using C++ libraries from Julia.