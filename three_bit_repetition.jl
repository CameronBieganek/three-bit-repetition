
using StatsBase
using Distributions
using StochasticAD
# using Optimisers  # StochasticAD can't take the derivative of my `loss` function.
using CairoMakie


# Sampler for the random variable Y.
y(; θ) = rand(Bernoulli(θ))

# Sampler for the random variable X.
function x(y; p)
    effective_ps = [p, 1-p]
    p_effective = effective_ps[y + 1]
    rand(Binomial(3, p_effective))
end

# Sample from the joint distribution of Y and X.
function y_x(; θ, p)
    y_ = y(; θ)
    x_ = x(y_; p)
    y_, x_
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
            y, x = y_x(; θ, p)
            ŷ = Ŷ_sampler(x)
            Int(ŷ == y)
        end
    end
end

function MAP_accuracies(; θ, n=50)
    map(p_range()) do p
        mean(1:n) do _
            y, x = y_x(; θ, p)
            ŷ = ŷ_MAP(x; θ, p)
            Int(ŷ == y)
        end
    end
end

function plot_accuracies(path; θ, n)
    majority_rule_accuracies = voting_accuracies(ŷ_maj; θ, n)
    minority_rule_accuracies = voting_accuracies(ŷ_min; θ, n)
    MAP_accuracies_ = MAP_accuracies(; θ, n)

    f = Figure()
    ax = Axis(f, title="Ŷ estimator accuracies, θ=$θ", xlabel="p", ylabel="accuracy")
    f[1, 1] = ax

    ps = p_range()
    lines!(ax, ps, MAP_accuracies_, linewidth=2, label="MAP")
    lines!(ax, ps, majority_rule_accuracies, linewidth=2, label="majority rule")
    lines!(ax, ps, minority_rule_accuracies, linewidth=2, label="minority rule")

    f[1, 2] = Legend(f, ax, framevisible = false)

    save(path, f; px_per_unit=2)
end

# y before x because y "generates" or "causes" x.
function joint_pdf(y, x; θ, p)
    binomial(3, x)  *
        (θ * p^(3-x) * (1-p)^x)^y  *
            ((1-θ) * p^x * (1-p)^(3-x))^(1-y)
end

y_x_range() = sort(vec(collect(Iterators.product(0:1, 0:3))); by=first)

function analytical_voting_accuracy(Ŷ_sampler; θ, p)
    y_x = y_x_range()

    sum(y_x) do (y, x)
        ŷ = Ŷ_sampler(x)
        accuracy_y_x = Int(ŷ == y)
        accuracy_y_x * joint_pdf(y, x; θ, p)
    end
end

function analytical_MAP_accuracy(; θ, p)
    y_x = y_x_range()

    sum(y_x) do (y, x)
        ŷ = ŷ_MAP(x; θ, p)
        accuracy_y_x = Int(ŷ == y)
        accuracy_y_x * joint_pdf(y, x; θ, p)
    end
end

function analytical_voting_accuracies(Ŷ_sampler; θ)
    map(p -> analytical_voting_accuracy(Ŷ_sampler; θ, p), p_range())
end

function analytical_MAP_accuracies(; θ)
    map(p -> analytical_MAP_accuracy(; θ, p), p_range())
end

function plot_analytical_accuracies(path; θ)
    majority_rule_accuracies = analytical_voting_accuracies(ŷ_maj; θ)
    minority_rule_accuracies = analytical_voting_accuracies(ŷ_min; θ)
    MAP_accuracies = analytical_MAP_accuracies(; θ)

    f = Figure()
    ax = Axis(f, title="Analytical Ŷ estimator accuracies, θ=$θ", xlabel="p", ylabel="accuracy")
    f[1, 1] = ax

    ps = p_range()
    lines!(ax, ps, MAP_accuracies, linewidth=2, label="MAP")
    lines!(ax, ps, majority_rule_accuracies, linewidth=2, label="majority rule")
    lines!(ax, ps, minority_rule_accuracies, linewidth=2, label="minority rule")

    f[1, 2] = Legend(f, ax, framevisible = false)

    save(path, f; px_per_unit=2)
end

# Create an iterator of samples (y, x).
generate_data(; n, θ, p) = (y_x(; θ, p) for _ in 1:n)

function maximum_likelihood_estimator(data)
    θ̂ = mean(first, data)

    p̂ = mean(data) do (y, x)
        (1/3)*x - (2/3)*x*y + y
    end

    θ̂, p̂
end

function make_empirical_pdf(data)
    cm = countmap(data)
    n = length(data)

    function empirical_pdf(y, x)
        # If (y, x) is not in the count map dict, that means
        # that there were zero counts for that value in the
        # data set, hence the default of 0 in `get`.
        count = get(cm, (y, x), 0)
        count/n
    end
end

function make_loss(data; m)
    observed_pdf = make_empirical_pdf(data)

    # Sampler for the loss random variable.
    function loss(ψ)
        θ, p = ψ

        # `data` is the observed data, and `samples` is generated
        # data that we would like to match to the observed data.
        samples = generate_data(; n=m, θ, p)
        sampled_pdf = make_empirical_pdf(samples)

        sum(y_x_range()) do (y, x)
            abs(sampled_pdf(y, x) - observed_pdf(y, x))
        end
    end
end

function plot_loss(data)
    loss = make_loss(data; m=length(data))

    θs = 0.1:0.1:0.5
    ps = p_range()

    mean_loss = map(θs) do θ
        [mean(loss([θ, p]) for _ in 1:1000) for p in ps]
    end

    f = Figure()
    ax = Axis(f, title="Mean Loss", xlabel="p", ylabel="loss")
    f[1, 1] = ax

    foreach(θs, mean_loss) do θ, loss
        lines!(ax, ps, loss, linewidth=2, label="θ=$θ")
    end

    f[1, 2] = Legend(f, ax, framevisible = false)
    f
end

data = generate_data(n=1000, θ=0.5, p=0.2)
θ̂_MLE, p̂_MLE = maximum_likelihood_estimator(data)
