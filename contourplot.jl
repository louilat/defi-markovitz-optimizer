include("src/computation/objective.jl")
include("src/computation/minimization.jl")

using Plots; pythonplot()

T = 1.
C = [1 .3; .3 1]
μ = [.1, -.11]
σ = [.40, .45]
r = [.05, .02]
lt = [.75, .80]
lb = [1.25, 1.30]
x0 = [0, 1]

params = modelParams(T, C, μ, σ, r, lt, lb)
obj = wrap_objective(params)
minimizer = adam_optimize(obj, 5000, x0, 5e-3, [.9, .999], 0, 1e-8; project = true)

# function f(x::Float64, y::Float64)::Float64
#     model = DefiMarkovitzModel(
#         T, μ, r, σ, C, lt, lb, [x, y],
#     )
#     return compute_variance(model)
# end
f(x, y) = obj([x, y])

function b(x::Float64)::Float64
    LT1 = lt[1]
    LT2 = lt[2]
    if x <= 0
        return - 1 / LT2 * x
    end
    return - LT1 * x
end

π1 = range(-3, 3, length=100)
π2 = range(-3, 3, length=100)
v = @. f(π1', π2)


plot1 = contour(π1, π2, v, levels=15, color=:turbo, clabels=true, cbar=false, lw=1, fill = false, fmt = :png)
plot!(xlim=xlims(), ylim=ylims())
plot!(π1, b.(π1), fillrange = fill(-3, 100), fillstyle = :/, fc = :gray, c = :gray)
scatter!([minimizer[1]], [minimizer[2]], color = "green", label = "", markersize = 10)
savefig(plot1, "contour3.png")

println(minimizer)
