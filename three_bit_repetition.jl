
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

p_range() = range(start=0.01, stop=0.99, length=99)

function voting_accuracies(Ŷ_sampler; θ, n=50)
    map(p_range()) do p
        mean(1:n) do _
            y_ = y(; θ)
            x_ = x(y_; p)
            ŷ_ = Ŷ_sampler(x_)
            Int(ŷ_ == y_)
        end
    end
end

function MAP_accuracies(; θ, n=50)
    map(p_range()) do p
        mean(1:n) do _
            y_ = y(; θ)
            x_ = x(y_; p)
            ŷ_ = ŷ_MAP(x_; θ, p)
            Int(ŷ_ == y_)
        end
    end
end

function plot_accuracies(; θ, n)
    majority_rule_accuracies = voting_accuracies(ŷ_maj; θ, n)
    minority_rule_accuracies = voting_accuracies(ŷ_min; θ, n)
    MAP_accuracies_ = MAP_accuracies(; θ, n)

    f = Figure()
    ax = Axis(f, title="Ŷ estimator accuracies", xlabel="p", ylabel="accuracy")
    f[1, 1] = ax

    ps = p_range()
    lines!(ax, ps, MAP_accuracies_, linewidth=2, label="MAP")
    lines!(ax, ps, majority_rule_accuracies, linewidth=2, label="majority rule")
    lines!(ax, ps, minority_rule_accuracies, linewidth=2, label="minority rule")

    f[1, 2] = Legend(f, ax, framevisible = false)

    f
end

function joint_pdf(x, y; θ, p)
    binomial(3, x)  *
        (θ * p^(3-x) * (1-p)^x)^y  *
            ((1-θ) * p^x * (1-p)^(3-x))^(1-y)
end

function analytical_voting_accuracy(Ŷ_sampler; θ, p)
    x_y = Iterators.product(0:3, 0:1)

    sum(x_y) do (x, y)
        ŷ = Ŷ_sampler(x)
        accuracy_x_y = Int(ŷ == y)
        accuracy_x_y * joint_pdf(x, y; θ, p)
    end
end

function analytical_MAP_accuracy(; θ, p)
    x_y = Iterators.product(0:3, 0:1)

    sum(x_y) do (x, y)
        ŷ = ŷ_MAP(x; θ, p)
        accuracy_x_y = Int(ŷ == y)
        accuracy_x_y * joint_pdf(x, y; θ, p)
    end
end

function analytical_voting_accuracies(Ŷ_sampler; θ)
    map(p -> analytical_voting_accuracy(Ŷ_sampler; θ, p), p_range())
end

function analytical_MAP_accuracies(; θ)
    map(p -> analytical_MAP_accuracy(; θ, p), p_range())
end

function plot_analytical_accuracies(; θ)
    majority_rule_accuracies = analytical_voting_accuracies(ŷ_maj; θ)
    minority_rule_accuracies = analytical_voting_accuracies(ŷ_min; θ)
    MAP_accuracies = analytical_MAP_accuracies(; θ)

    f = Figure()
    ax = Axis(f, title="Analytical Ŷ estimator accuracies", xlabel="p", ylabel="accuracy")
    f[1, 1] = ax

    ps = p_range()
    lines!(ax, ps, MAP_accuracies, linewidth=2, label="MAP")
    lines!(ax, ps, majority_rule_accuracies, linewidth=2, label="majority rule")
    lines!(ax, ps, minority_rule_accuracies, linewidth=2, label="minority rule")

    f[1, 2] = Legend(f, ax, framevisible = false)

    f
end

# Assume data is an iterator of tuples (x, y).
function maximum_likelihood_estimator(data)
    θ̂ = mean(last, data)

    p̂ = mean(data) do (x, y)
        (1/3)*x - (2/3)*x*y + y
    end

    θ̂, p̂
end

function generate_data(n; θ, p)
    map(1:n) do _
        y_ = y(; θ)
        x_ = x(y_; p)
        x_, y_
    end
end
