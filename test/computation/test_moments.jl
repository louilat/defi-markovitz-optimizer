using Test
include("../../src/computation/moments.jl")


T = 2.
C = [1 .5 .3; .5 1 .4; .3 .4 1]
μ = [.1, .15, -.11]
σ = [.30, .35, .30]
r = [.05, .02, .03]
lt = [.75, .80, .85]
ω = [10, 15, -18] / 7

model = DefiMarkovitzModel(
    T, μ, r, σ, C, lt, ω
)

ν, δ, λ, γ = compute_nu_delta_lambda_gamma(model)
@test ν ≈ [0.3214, 0.6, -0.7714] atol=0.001
@test δ ≈ .2143 atol=0.001
@test λ ≈ .6471 atol=0.001
@test γ ≈ .8557 atol=0.001

norm_x, x1, norm_y, y1, x_dot_y = compute_x_and_y(model, ν, γ)
@test norm_x ≈ 1.001 atol=0.001
@test x1 ≈ .9838 atol=0.001
@test norm_y ≈ 1.4579 atol=0.001
@test y1 ≈ -1.3759 atol=0.001
@test x_dot_y ≈ -1.2660 atol=0.001

c1, c2, c3 = compute_c(model)
@test c1 ≈ 1.5686 atol=0.001
@test c2 ≈ -.04 atol=0.001
@test c3 ≈ -.4086 atol=0.001

η = λ / γ
I0 = compute_I0(η, δ, γ, T)
@test I0 ≈ 0.80192 atol=0.00001

I1 = compute_I1(η, δ, γ, T)
@test I1 ≈ -.13650 atol=0.00001

I2 = compute_I2(η, δ, γ, T)
@test I2 ≈ -.10672 atol=0.00001

M1 = compute_first_order_moment_return(c1, c2, c3, η, δ, γ, y1, I0, I1)
@test M1 ≈ 1.6755 atol=0.001

ϕ0 = compute_phi0(c1, c2, c3, δ, γ, x1, y1, norm_y, x_dot_y, T)
ϕ1 = compute_phi1(c1, c2, c3, η, δ, γ, x1, y1)
ϕ2 = compute_phi2(c2, η, x1, y1)
@test ϕ0 ≈ -.9780 atol=0.001
@test ϕ1 ≈ -3.4041 atol=0.001
@test ϕ2 ≈ -.8527 atol=0.001

M2 = compute_second_order_moment_return(c1, η, δ, γ, norm_x, ϕ0, ϕ1, ϕ2, I0, I1, I2, T)
@test M2 ≈ 4.2743 atol=0.001