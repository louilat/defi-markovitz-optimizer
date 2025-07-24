function compute_first_order_moment_return(
    c1::Float64,
    c2::Float64,
    c3::Float64,
    eta::Float64,
    delta::Float64,
    lambda::Float64,
    gamma::Float64,
    y1::Float64,
    z1::Float64,
    I0::Float64,
    I1::Float64,
)
    coeff = exp(-eta * delta / gamma)
    term0 = c3 - delta / gamma * (y1 + z1)
    term1 = y1 + 1 / eta * (z1 * lambda / gamma - c2)
    return c1 + coeff * (term0 * I0 + term1 * I1)
end

function compute_second_order_moment_return(
    η::Float64,
    δ::Float64,
    γ::Float64,
    Λ0::Float64,
    Λ1::Float64,
    Λ2::Float64,
    Λ3::Float64,
    Λ4::Float64,
    Λ5::Float64,
    Λ6::Float64,
    I0::Float64,
    I1::Float64,
    I2::Float64,
    T::Float64,
)::Float64
    coeff = exp(- η * δ / γ)
    term0 = Λ1 - δ / γ * Λ4 + (T + (δ/γ)^2) * Λ6
    term1 = 1/η * (δ/γ + 1/η) * Λ5 - 1/η * Λ2 - 1/η^3 * Λ3 - 2 * δ/γ * Λ6 + Λ4
    term2 = 1/η^2 * Λ3 + Λ6 - 1/η * Λ5
    return Λ0 + coeff * (term0 * I0 + term1 * I1 + term2 * I2)
end