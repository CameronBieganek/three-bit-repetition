
using Distributions
using StochasticAD
using Optimisers
using CairoMakie


# Sampler for the random variable Y.
y(; θ) = rand(Bernoulli(θ))

# Sampler for the random variable X.
function x(y; p)
    effective_ps = [p, 1-p]
    p_effective = effective_ps[y + 1]
    rand(Binomial(3, p_effective))
end

# Samplers for the estimators (random variables) Ŷ_maj and Ŷ_min.
ŷ_maj(x) = Int(x > 1.5)
ŷ_min(x) = Int(x < 1.5)

# Sampler for the estimator (random variable) Ŷ_MAP.
function ŷ_MAP(x; θ, p)
    msg(param) = "Parameter $param must be in the interval (0, 1)."
    (0 < θ < 1) || throw(ArgumentError(msg("θ")))
    (0 < p < 1) || throw(ArgumentError(msg("p")))

    boundary(θ, p) = 1.5 + log((1-θ)/θ) / log((1-p)/p)

    if p < 0.5
        Int(x > boundary(θ, p))
    elseif p > 0.5
        Int(x < boundary(θ, p))
    else # In this case p == 0.5
        if θ == 0.5
            # In this case, the posterior probability of Y==0 and Y==1 are
            # the same (both equal to 1/2) for all values of x. So there is
            # no unique argmax, so we can pick an arbitrary value for Ŷ.
            1
        else
            Int(θ > 0.5)
        end
    end
end

function voting_accuracies(Ŷ_sampler; θ, n=50)
    ps = range(start=0.01, stop=0.99, length=99)
    map(ps) do p
        mean(1:n) do _
            y_ = y(; θ)
            x_ = x(y_; p)
            ŷ_ = Ŷ_sampler(x_)
            Int(ŷ_ == y_)
        end
    end
end

function MAP_accuracies(; θ, n=50)
    ps = range(start=0.01, stop=0.99, length=99)
    map(ps) do p
        mean(1:n) do _
            y_ = y(; θ)
            x_ = x(y_; p)
            ŷ_ = ŷ_MAP(x_; θ, p)
            Int(ŷ_ == y_)
        end
    end
end

function plot_accuracies(; θ, n)
    ps = range(start=0.01, stop=0.99, length=99)

    majority_rule_accuracies = voting_accuracies(ŷ_maj; θ, n)
    minority_rule_accuracies = voting_accuracies(ŷ_min; θ, n)
    MAP_accuracies_ = MAP_accuracies(; θ, n)

    f = Figure()
    ax = Axis(f, title="ŷ_MAP accuracy", xlabel="p", ylabel="accuracy")
    f[1, 1] = ax

    lines!(ax, ps, MAP_accuracies_, label="MAP")
    lines!(ax, ps, majority_rule_accuracies, label="maj")
    lines!(ax, ps, minority_rule_accuracies, label="min")

    # ylims!(ax, (-0.1, 1.1))

    f
end
