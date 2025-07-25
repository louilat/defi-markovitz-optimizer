include("src/computation/objective.jl")

using Plots; pythonplot()

T = 1.
C = [1 .3; .3 1]
μ = [.1, -.11]
σ = [.35, .40]
r = [.05, .02]
lt = [.75, .80]
lb = [1.05, 1.10]

function f(x::Float64, y::Float64)::Float64
    model = DefiMarkovitzModel(
        T, μ, r, σ, C, lt, lb, [x, y],
    )
    return compute_variance(model)
end

π1 = range(-3, 3, length=100)
π2 = range(-3, 3, length=100)
v = @. f(π1', π2)

plot1 = contour(π1, π2, v, levels=15, color=:turbo, clabels=true, cbar=false, lw=1, fill = false, fmt = :png)
savefig(plot1, "contour.png")

println(f(1/sqrt(2), 1/sqrt(2)))
println(f(sqrt(2), sqrt(2)))
