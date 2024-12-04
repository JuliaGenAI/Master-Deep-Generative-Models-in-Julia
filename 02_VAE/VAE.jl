using Lux
using Random
using Statistics: mean
using Optimisers
# import MLDatasets: MNIST
import MLDatasets: FashionMNIST as MNIST
using ComponentArrays: ComponentArray
using MLUtils: DataLoader
using Zygote
using ImageCore: colorview, Gray
using ImageInTerminal

TRAIN, TEST = MNIST(:train), MNIST(:test)

struct Encoder{H} <: LuxCore.AbstractLuxContainerLayer{(:h, :μ, :logσ)}
    h::H
    μ::Dense
    logσ::Dense
end

function (m::Encoder)(x, ps, st)
    h, st_h = m.h(x, ps.h, st.h)
    μ, st_μ = m.μ(h, ps.μ, st.μ)
    logσ, st_logσ = m.logσ(h, ps.logσ, st.logσ)
    (μ, logσ), (h=st_h, μ=st_μ, logσ=st_logσ)
end

struct VAE{B,D} <: LuxCore.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::Encoder{B}
    decoder::D
end

function Lux.initialstates(rng::AbstractRNG, m::VAE)
    encoder = Lux.initialstates(rng, m.encoder)
    decoder = Lux.initialstates(rng, m.decoder)
    (encoder=encoder, decoder=decoder, rng=rng)
end

function (m::VAE)(x, ps, st)
    (μ, logσ), st_encoder = m.encoder(x, ps.encoder, st.encoder)
    σ = exp.(logσ)
    ϵ = randn(st.rng, size(μ))
    z = μ .+ σ .* ϵ
    x̂, st_decoder = m.decoder(z, ps.decoder, st.decoder)
    (x̂, μ, logσ), (encoder=st_encoder, decoder=st_decoder, rng=st.rng)
end

function loss_fn(x̂, μ, logσ, x)
    reconstruction_loss = -mean(sum(x .* log.(1e-10 .+ x̂) .+ (1 .- x) .* log.(1e-10 .+ 1 .- x̂); dims=1))
    kl_loss = -0.5 * mean(sum(1 .+ logσ .- μ .^ 2 .- exp.(logσ); dims=1))
    reconstruction_loss + kl_loss
end

hidden_dim = 400
latent_dim = 20

m = VAE(
    Encoder(
        Dense(784 => hidden_dim, relu),
        Dense(hidden_dim => latent_dim),
        Dense(hidden_dim => latent_dim)
    ),
    Chain(
        Dense(latent_dim => hidden_dim, relu),
        Dense(hidden_dim => 784, sigmoid),
    )
)


function train()
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, m)
    ps = ps |> ComponentArray

    opt = Adam()
    opt_st = Optimisers.setup(opt, ps)

    n_epoch = 30
    batchsize = 100

    for epoch in 0:n_epoch-1
        for (i, x) in enumerate(DataLoader(TRAIN.features; batchsize=batchsize, shuffle=true, rng=rng))
            x = reshape(x, :, size(x, 3))
            grads = Zygote.gradient(ps) do ps
                (x̂, μ, logσ), st = m(x, ps, st)
                L = loss_fn(x̂, μ, logσ, x)
                if i == 1
                    println("Epoch $epoch \t Loss $L")
                end
                L
            end
            Optimisers.update!(opt_st, ps, grads[1])
        end
    end
    ps, st, opt, opt_st
end

ps, st, opt, opt_st = train()

function compare(ps, st)
    x = reshape(TEST.features[:, :, rand(1:size(TEST.features, 3))], 784, 1)
    st = Lux.testmode(st)
    (x̂, _, _), _ = m(x, ps, st)
    colorview(Gray, hcat(reshape(x, 28, 28)', reshape(x̂, 28, 28)'))
end

function generate(ps, st)
    z = randn(latent_dim, 1)
    st = Lux.testmode(st.decoder)
    x̂, _ = m.decoder(z, ps.decoder, st)
    colorview(Gray, reshape(x̂, 28, 28)')
end

compare(ps, st)