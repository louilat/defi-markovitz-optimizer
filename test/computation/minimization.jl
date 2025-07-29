include("../../src/computation/minimization.jl")

using Test

@testset "Compute gradient" begin
    @test compute_gradient(x -> 2 * x[1]^2 + 3 * x[2]^3 + 2 * x[1] * x[3], [1, 2, 0]) ≈ [4, 36, 2] atol = 0.001
    @test compute_gradient(x -> x[1]^2 + x[1] + exp(x[2]) + 2 * x[1] * x[3], [3, 0, 1]) ≈ [9, 1, 6] atol = 0.001
end
