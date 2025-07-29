# using Optimization
# using Zygote
# using OptimizationOptimJL
# using OptimizationMOI, Ipopt
# using Flux
# using AmplNLWriter, Ipopt_jll

include("objective.jl")

struct modelParams 
    T::Real
    C::Matrix{<:Real} 
    μ::Vector{<:Real}
    σ::Vector{<:Real}
    r::Vector{<:Real}
    lt::Vector{<:Real}
    lb::Vector{<:Real}
end

T = 1.
C = [1 .3; .3 1]
μ = [.1, -.11]
σ = [.35, .40]
r = [.05, .02]
lt = [.75, .80]
lb = [1.05, 1.10]

# println("Starting gradient computation...")

function wrap_objective(params::modelParams)::Function
    return x -> compute_objective(
        DefiMarkovitzModel(
            params.T, params.μ, params.r, params.σ, params.C, params.lt, params.lb, x
        )
    )
end

function compute_gradient(objective::Function, x::Vector{<:Real})::Vector{<:Real}
    d = length(x)
    g = zeros(d)
    ε = 1e-5
    for i in 1:d
        δ = zeros(d)
        δ[i] = ε
        g[i] = (objective(x + δ) - objective(x - δ)) / (2*ε)
    end
    return g
end

function orthogonal_projection(x::Vector{<:Real})#::Vector{<:Real}
    d = length(x)
    B = Bidiagonal(ones(d), - ones(d-1), :L)[:, 1:d-1]
    P = B * inv(transpose(B) * B) * transpose(B)
    return P * x
end

function gradient_descent(objective::Function, x0::Vector{<:Real}, lr::Real, niter::Int)
    x = x0
    history = []
    for _ in 1:niter
        g = compute_gradient(objective, x)
        x = x - lr * g
        append!(history, x)
    end
    return x, history
end

function adam_optimize(
    objective::Function, niter::Int, theta::Vector{<:Real}, lr::Real, betas::Vector{<:Real}, lambda::Real, epsilon::Real; project = false
)::Vector{<:Real}
    d = length(theta)
    m = zeros(d)
    v = zeros(d)
    x = theta
    beta1 = betas[1]
    beta2 = betas[2]
    score = Inf
    a = zeros(d); a[1] = 1
    for t in 1:niter
        g = compute_gradient(objective, x)
        if project
            g = orthogonal_projection(g)
        end
        g = g + lambda * x
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g.^2
        m_hat = m ./ (1 - beta1^t)
        v_hat = v ./ (1 - beta2^t)
        x = x - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)
        if project 
            x = a + orthogonal_projection(x - a)
        end
        if t%10 == 0
            score = objective(x)
            @info "Iteration $t : current score = $score"
        end
    end
    return x
end

# x0 = [1, 0]
# params = modelParams(T, C, μ, σ, r, lt, lb)
# obj = wrap_objective(params)
# x_min = adam_optimize(obj, 5000, x0, 1e-3, [.9, .999], 0, 1e-8; project = true)
# # println(h)
# println(x_min)

# orthogonal_projection([2.73, -.5])

