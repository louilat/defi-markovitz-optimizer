using LinearAlgebra
using Distributions

struct DefiMarkovitzModel
    # Raw Parameters
    horizon::Float64
    mus::Vector{Float64}
    apy::Vector{Float64}
    sigmas::Vector{Float64}
    correlations::Matrix{Float64}
    liqthreshs::Vector{Float64}
    weights::Vector{Float64}
    
    # Intermediate computed Parameters
    longs::Vector{Float64}
    shorts::Vector{Float64}
    sqrt_correlations::Matrix{Float64}
end

function DefiMarkovitzModel(
    horizon::Float64, mus::Vector{Float64}, apy::Vector{Float64}, sigmas::Vector{Float64}, correlations::Matrix{Float64}, liqthreshs::Vector{Float64}, weights::Vector{Float64}
)::DefiMarkovitzModel
    check =  minimum(eigvals(correlations)) > 0 ? true : throw(DomainError(correlations, "Correlations matrix is not positive definite"))
    return DefiMarkovitzModel(horizon, mus, apy, sigmas, correlations, liqthreshs, weights, max.(weights, 0), max.(-weights, 0), sqrt(correlations))
end

function compute_nu_delta_lambda_gamma(
    model::DefiMarkovitzModel
)::Tuple{Vector{Float64}, Float64, Float64, Float64}
    nu = model.sigmas .* (model.longs .* model.liqthreshs .- model.shorts)
    delta = sum(model.longs .* model.liqthreshs .- model.shorts)
    lambda = sum((model.longs .* model.liqthreshs .- model.shorts) .* model.mus)
    gamma = norm(model.sqrt_correlations * nu, 2)
    @info "Successfully computed nu, delta, lambda, gamma\n   --> nu = $nu\n   --> delta = $delta\n   --> lambda = $lambda\n   --> gamma = $gamma"
    return nu, delta, lambda, gamma
end

function compute_x_and_y(model::DefiMarkovitzModel, nu::Vector{Float64}, gamma::Float64)::Tuple{Float64, Float64, Float64, Float64, Float64}
    norm_x = transpose(model.weights) * diagm(model.sigmas) * model.correlations * diagm(model.sigmas) * model.weights
    norm_y = transpose(model.shorts - model.weights) * diagm(model.sigmas) * model.correlations * diagm(model.sigmas) * (model.shorts - model.weights)
    norm_x = sqrt(norm_x)
    norm_y = sqrt(norm_y)

    x1 = transpose(nu) * model.correlations * diagm(model.sigmas) * model.weights / gamma
    y1 = transpose(nu) * model.correlations * diagm(model.sigmas) * (model.shorts - model.weights) / gamma

    x_dot_y = transpose(model.weights) * diagm(model.sigmas) * model.correlations * diagm(model.sigmas) * (model.shorts .- model.weights)

    @info "Successfully computed x and y\n   --> norm_x = $norm_x\n   --> x1 = $x1\n   --> norm_y = $norm_y\n   --> y1 = $y1\n   --> (x|y) = $x_dot_y"
    return norm_x, x1, norm_y, y1, x_dot_y
end

function compute_c(model::DefiMarkovitzModel)::Tuple{Float64, Float64, Float64}
    c1 = transpose(model.weights) * (model.mus .+ model.apy) * model.horizon
    c2 = transpose(model.weights .- model.shorts) * model.apy
    c3 = transpose(model.shorts .- model.weights) * (model.mus .+ model.apy) * model.horizon - sum(model.weights .- model.shorts)
    @info "Successfully computed c\n   --> c1 = $c1\n   --> c2 = $c2\n   --> c3 = $c3"
    return c1, c2, c3
end

function compute_I0(eta, delta, gamma, horizon)
    N = Normal()
    left = exp(-eta * delta / gamma) - exp(-eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) - eta * sqrt(horizon))
    right = exp(eta * delta / gamma) - exp(eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) + eta * sqrt(horizon))
    return left + right
end

function compute_I1(eta, delta, gamma, horizon)
    N = Normal()
    left = exp(eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) + eta * sqrt(horizon)) - exp(eta * delta / gamma)
    right = exp(-eta * delta / gamma) - exp(-eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) - eta * sqrt(horizon))
    return - delta / gamma * (left + right)
end

function compute_I2(eta, delta, gamma, horizon)
    N = Normal()
    coeff = delta / (eta * gamma) * exp(-eta * delta / gamma)
    left = (eta * delta / gamma + 1) * (cdf(N, delta / (gamma * sqrt(horizon)) - eta * sqrt(horizon)) - 1/2)
    right = (eta * delta / gamma - 1) * (exp(2 * eta * delta / gamma) * cdf(N, delta / (gamma * sqrt(horizon)) + eta * sqrt(horizon)) - exp(2 * eta * delta / gamma) + 1/2)
    cst =  - delta / gamma * sqrt(2 * horizon / pi) * exp(- eta^2 / 2 * horizon - (delta / gamma)^2 / (2 * horizon)) + (delta / gamma)^2 * exp(-eta * delta / gamma)
    return 1 / eta * compute_I1(eta, delta, gamma, horizon) - coeff * (left + right) + cst
end

function compute_phi0(c1, c2, c3, delta, gamma, x1, y1, norm_y, x_dot_y, horizon)
    check = gamma > 0 ? true : throw(DomainError(gamma, "gamma must be positive"))
    
    term1 = c3^2 + 2 * c1 * c3
    term2 = 2 * delta / gamma * (c3 * (x1 + y1) + c1 * y1)
    term3 = (y1^2 + 2 * x1 * y1) * (horizon + (delta / gamma)^2)
    term4 = (norm_y^2 - y1^2 + 2 * x_dot_y - 2 * x1 * y1) * horizon
    return term1 - term2 + term3 + term4
end

function compute_phi1(c1, c2, c3, eta, delta, gamma, x1, y1)
    check = eta != 0 ? true : throw(DomainError(eta, "eta must be positive"))
    check = gamma > 0 ? true : throw(DomainError(gamma, "gamma must be positive"))

    term1 = 2 * c2 * (x1 + y1) / eta * (delta / gamma + 1 / eta)
    term2 = 2 / eta * (c1 * c2 + c2 * c3)
    term3 = 2 * (c3 * (x1 + y1) + c1 * y1)
    term4 = c2^2 / eta^3 + 2 * delta / gamma * (y1^2 + 2 * x1 * y1)
    return term1 - term2 + term3 - term4
end

function compute_phi2(c2, eta, x1, y1)
    check = eta != 0 ? true : throw(DomainError(eta, "eta must be positive"))
    return y1^2 + 2 * x1 * y1 - 2 * c2 / eta * (x1 + y1) + (c2 / eta)^2
end

function compute_first_order_moment_return(c1, c2, c3, eta, delta, gamma, y1, I0, I1)
    coeff = exp(-eta * delta / gamma)
    left = (y1 - c2 / eta) * I1
    right = (c3 - y1 * delta / gamma) * I0
    return c1 + coeff * (left + right)
end

function compute_second_order_moment_return(c1, eta, delta, gamma, norm_x, phi0, phi1, phi2, I0, I1, I2, horizon)
    return norm_x^2 * horizon + c1^2 + exp(-eta * delta / gamma) * (phi0 * I0 + phi1 * I1 + phi2 * I2)
end

function compute_variance(model::DefiMarkovitzModel)::Float64
    ν, δ, λ, γ = compute_nu_delta_lambda_gamma(model)
    norm_x, x1, norm_y, y1, x_dot_y = compute_x_and_y(model, ν, γ)
    c1, c2, c3 = compute_c(model)
    η = λ / γ
    T = model.horizon

    if δ > 0
        I0 = compute_I0(η, δ, γ, T)
        I1 = compute_I1(η, δ, γ, T)
        I2 = compute_I2(η, δ, γ, T)

        M1 = compute_first_order_moment_return(c1, c2, c3, η, δ, γ, y1, I0, I1)

        ϕ0 = compute_phi0(c1, c2, c3, δ, γ, x1, y1, norm_y, x_dot_y, T)
        ϕ1 = compute_phi1(c1, c2, c3, η, δ, γ, x1, y1)
        ϕ2 = compute_phi2(c2, η, x1, y1)
        M2 = compute_second_order_moment_return(c1, η, δ, γ, norm_x, ϕ0, ϕ1, ϕ2, I0, I1, I2, T)
    else
        M1 = c1 + c3
        M2 = norm_x^2 * T + c1^2 + c3^2 + 2 * c1 * c3 + (norm_y^2 - y1^2 + 2 * x_dot_y - 2 * x1 * y1) * T + (y1^2 + 2 * x1 * y1) * T
    end
    return M2 - M1^2
end