using Test

include("../../src/computation/model.jl")
include("../../src/computation/parameters.jl")
include("../../src/computation/integrals.jl")
include("../../src/computation/moments.jl")


T = 2.
C = [1 .5 .3; .5 1 .4; .3 .4 1]
μ = [.1, .15, -.11]
σ = [.30, .35, .30]
r = [.05, .02, .03]
lt = [.75, .80, .85]
lb = [1.05, 1.10, 1.05]
ω = [10, 15, -18] / 7

model = DefiMarkovitzModel(
    T, μ, r, σ, C, lt, lb, ω,
)

ν, δ, λ, γ = compute_nu_delta_lambda_gamma(model)

@test ν ≈ [0.3214, 0.6, -0.7714] atol=0.001
@test δ ≈ .2143 atol=0.001
@test λ ≈ .6471 atol=0.001
@test γ ≈ .8557 atol=0.001

ψ, ϕ = compute_psi_phi(model)

norm_x, x1, norm_y, y1, norm_z, z1, x_dot_y, x_plus_y_dot_z = compute_x_y_z(model, ν, γ, ψ)

@test norm_x ≈ 1.001 atol=0.001
@test x1 ≈ .9838 atol=0.001
@test norm_y ≈ 0.8523 atol=0.001
@test y1 ≈ -0.8522 atol=0.001
@test norm_z ≈ 0.2298 atol=0.001
@test z1 ≈ -0.1316 atol=0.001
@test x_dot_y ≈ -0.8376 atol=0.001
@test x_plus_y_dot_z ≈ -0.0528 atol=0.001

c1, c2, c3 = compute_c(model, ψ, ϕ)

@test c1 ≈ 1.5686 atol=0.001
@test c2 ≈ 0.0915 atol=0.001
@test c3 ≈ -1.5169 atol=0.001

η = λ / γ
I0 = compute_I0(η, δ, γ, T)
@test I0 ≈ 0.80192 atol=0.00001

I1 = compute_I1(η, δ, γ, T)
@test I1 ≈ -.13650 atol=0.00001

I2 = compute_I2(η, δ, γ, T)
@test I2 ≈ -.10672 atol=0.00001

M1 = compute_first_order_moment_return(c1, c2, c3, η, δ, λ, γ, y1, z1, I0, I1)
@test M1 ≈ 0.8503 atol=0.001

Λ0, Λ1, Λ2, Λ3, Λ4, Λ5, Λ6 = compute_all_lambdas(
    norm_x,
    norm_y,
    norm_z,
    x1,
    y1,
    z1,
    x_dot_y,
    x_plus_y_dot_z,
    λ,
    δ,
    γ,
    c1,
    c2,
    c3,
    T,
)

println(Λ0)
println(Λ1)
println(Λ2)
println(Λ3)
println(Λ4)
println(Λ5)
println(Λ6)

M2 = compute_second_order_moment_return(η, δ, γ, Λ0, Λ1, Λ2, Λ3, Λ4, Λ5, Λ6, I0, I1, I2, T)
@test M2 ≈ 2.4197 atol=0.001