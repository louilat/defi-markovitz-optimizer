include("../model/model.jl")

using Distributions

# function simulate_brownian_motion(T::Float64, n_points::Int64, C::Matrix{Float64}, σ::Vector{Float64})::Matrix{Float64}
#     d = length(σ)
#     N = MvNormal(zeros(d), C)
#     W = diagm(σ) * rand(N, n_points - 1) * sqrt(T / (n_points - 1))
#     W = hcat(zeros(d, 1), W)
#     W = cumsum(W, dims = 2)
#     return W
# end

function simulate_brownian_motion(T::Float64, n_points::Int64, C::Matrix{Float64})::Matrix{Float64}
    d = length(σ)
    N = MvNormal(zeros(d), C)
    W = rand(N, n_points - 1) * sqrt(T / (n_points - 1))
    W = hcat(zeros(d, 1), W)
    W = cumsum(W, dims = 2)
    return W
end

function compute_prices(T::Float64, W::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, p0::Vector{Float64})::Tuple{Vector{Float64}, Matrix{Float64}}
    n_points = size(W)[2]
    steps = collect(range(0, T, length=n_points))
    return steps, diagm(p0) * exp.(diagm(σ) * W + (μ - σ.^2 / 2) * transpose(steps))
end

function compute_tau(model::DefiMarkovitzModel, P::Matrix{Float64}, steps::Vector{Float64})::Tuple{Float64, Union{Nothing, Int}}
    numerator = sum(diagm(model.liqthreshs .* model.longs ./ P[:,1]) * P, dims = 1)
    denominator = sum(diagm(model.shorts ./ P[:,1]) * P, dims = 1)
    hf = vec(numerator ./ denominator)
    min_ = minimum(hf)
    max_ = maximum(hf)
    @info "Min hf = $min_\nMax hf = $max_"
    idx = findfirst(<=(1), hf)
    if isnothing(idx)
        @info "No liquidations found"
        return Inf, idx
    else
        τ = steps[idx]
        @info "Liquidation at time $τ"
        return τ, idx
    end
end

# function compute_simulated_return(model::DefiMarkovitzModel, W::Matrix{Float64}, ψ::Float64, ϕ::Float64, steps::Vector{Float64}, idx::Nothing)::Float64
#     W_last = W[:, end]
#     term0 = transpose(model.weights) * (W_last + (model.mus + model.apy) * model.horizon)
#     return term0
# end

function compute_simulated_return(
    model::DefiMarkovitzModel, W::Matrix{Float64}, ψ::Float64, ϕ::Float64, steps::Vector{Float64}, idx::Union{Int, Nothing}
)::Float64
    W_last = diagm(model.sigmas) * W[:, end]
    term0 = transpose(model.weights) * (W_last + (model.mus + model.apy) * model.horizon)
    if isnothing(idx)
        return term0
    end
    
    term1 = transpose(ψ * model.longs .- model.weights) * W_last
    W_tau = diagm(model.sigmas) * W[:, idx]
    tau = steps[idx]
    term2 = - ψ * transpose(model.longs) * W_tau
    term3 = - (transpose(model.weights) * model.apy - ψ * transpose(model.longs) * (model.mus .+ model.apy)) * tau
    term4 = transpose(ψ * model.longs .- model.weights) * (model.mus .+ model.apy) * model.horizon + ϕ
    return term0 + term1 + term2 + term3 + term4
end

function get_returns_distribution(model::DefiMarkovitzModel, p0::Vector{Float64}, nsim::Int)::Vector{Float64}
    output = Vector(undef, nsim)
    for i in 1:length(output)
        W = simulate_brownian_motion(model.horizon, 10000, model.correlations)
        steps, P = compute_prices(model.horizon, W, model.mus, model.sigmas, p0)
        _, idx = compute_tau(model, P, steps)
        ψ, ϕ = compute_psi_phi(model)
        output[i] = compute_simulated_return(model, W, ψ, ϕ, steps, idx)
    end
    return output
end
