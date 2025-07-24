using LinearAlgebra

struct DefiMarkovitzModel
    # Raw Parameters
    horizon::Float64
    mus::Vector{Float64}
    apy::Vector{Float64}
    sigmas::Vector{Float64}
    correlations::Matrix{Float64}
    liqthreshs::Vector{Float64}
    liqbonus::Vector{Float64}
    weights::Vector{Float64}
    
    # Intermediate computed Parameters
    longs::Vector{Float64}
    shorts::Vector{Float64}
    sqrt_correlations::Matrix{Float64}
end

function DefiMarkovitzModel(
    horizon::Float64,
    mus::Vector{Float64},
    apy::Vector{Float64},
    sigmas::Vector{Float64},
    correlations::Matrix{Float64},
    liqthreshs::Vector{Float64},
    liqbonus::Vector{Float64},
    weights::Vector{Float64}
)::DefiMarkovitzModel
    check =  minimum(eigvals(correlations)) > 0 ? true : throw(
        DomainError(correlations, "Correlations matrix is not positive definite")
    )
    return DefiMarkovitzModel(
        horizon,
        mus,
        apy,
        sigmas,
        correlations,
        liqthreshs,
        liqbonus,
        weights,
        max.(weights, 0),
        max.(-weights, 0),
        sqrt(correlations)
    )
end