using Test

using Aqua

using EasyEigenInterface

@testset begin
	Aqua.test_undefined_exports(EasyEigenInterface)
end

@testset "Matrix" begin
	x = Float32[1 2 3; 4 5 6]
	@test MatrixXf(MatrixXf(x)) == MatrixXf(x) == x

	x = Float64[1 2 3; 4 5 6]
	@test MatrixXd(MatrixXd(x)) == MatrixXd(x) == x

	m = MatrixXd(x)
	@test rows(m) == 2
	@test cols(m) == 3

	jlm = Matrix{Float64}(undef, 2, 3)
	jlm .= m
	@test jlm == m
end

@testset "Vector" begin
	x = Float32[1, 2, 3]
	@test VectorXf(VectorXf(x)) == VectorXf(x) == x

	x = Float64[1, 2, 3]
	@test VectorXd(VectorXd(x)) == VectorXd(x) == x

	v = VectorXd(x)
	jlv = Vector{Float64}(undef, 3)
	jlv .= v
	@test jlv == v
end

@testset "convert" begin
	x = Float64[1,2,3]
	v = VectorXd(x)
	m = convert(MatrixXd, v)
	@test size(m) == (length(x), 1)
end

@testset "resize!" begin
	jlm = Float64[1 2 3; 4 5 6]
	m = MatrixXd(jlm)
	rows(m) == 2
	cols(m) == 3

	resize!(m, 3, 2)
	rows(m) == 3
	cols(m) == 2

	@test m == Float64[
			1  5
			4  3
			2  6
	]
end

@testset "example1" begin
	x = rand(3,3)
	m = MatrixXd(x)
	@test EasyEigenInterface.example1(m) == 3x

	x = rand(3)
	v = VectorXd(x)
	m = convert(MatrixXd, v)
	@test vec(EasyEigenInterface.example1(m)) == 3x
end
