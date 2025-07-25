include("../model/model.jl")
include("integrals.jl")
include("moments.jl")
include("parameters.jl")


function compute_variance(model::DefiMarkovitzModel)::Float64
    ν, δ, λ, γ = compute_nu_delta_lambda_gamma(model)
    if δ <= 0 || γ <= 0 || λ == 0
        return NaN
    end
    ψ, φ = compute_psi_phi(model)
    norm_x, x1, norm_y, y1, norm_z, z1, x_dot_y, x_plus_y_dot_z = compute_x_y_z(model, ν, γ, ψ)
    c1, c2, c3 = compute_c(model, ψ, φ)
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
        model.horizon,
    )

    η = λ / γ
    I0 = compute_I0(η, δ, γ, model.horizon)
    I1 = compute_I1(η, δ, γ, model.horizon)
    I2 = compute_I2(η, δ, γ, model.horizon)
    M1 = compute_first_order_moment_return(c1, c2, c3, η, δ, λ, γ, y1, z1, I0, I1)
    M2 = compute_second_order_moment_return(
        η,
        δ,
        γ,
        Λ0,
        Λ1,
        Λ2,
        Λ3,
        Λ4,
        Λ5,
        Λ6,
        I0,
        I1,
        I2,
        model.horizon,
    )
    return M2 - M1^2 - M1
    # return M1
end