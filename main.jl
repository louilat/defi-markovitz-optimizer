include("src/simulation/return.jl")

using Base.Threads
using Plots

println(Threads.nthreads())

T = 2.
C = [1 .5 .3; .5 1 .4; .3 .4 1]
μ = [.1, .15, -.11]
σ = [.30, .35, .30]
r = [.05, .02, .03]
lt = [.75, .80, .85]
lb = [1.05, 1.10, 1.05]
ω = [10, 15, -18] / 7
p0 = [10., 15., 10.]

model = DefiMarkovitzModel(
    T, μ, r, σ, C, lt, lb, ω,
)

function full_simulation()
    M1 = Vector(undef, 200)
    M2 = Vector(undef, 200)
    m1_lock = ReentrantLock()
    m2_lock = ReentrantLock()

    Threads.@threads for i in 1:200
        D = get_returns_distribution(model, p0, 2_000)
        @lock m1_lock M1[i] = mean(D)
        @lock m2_lock M2[i] = mean(D.^2)
    end
    return M1, M2
end

M1, M2 = full_simulation()

plot1 = histogram(M1; fmt = :png)
plot!([0.8503], seriestype = :vline)
plot2 = histogram(M2; fmt = :png)
plot!([2.4197], seriestype = :vline)

savefig(plot1, "plot1.png")
savefig(plot2, "plot2.png")

# println(M1)
# println(M2)
