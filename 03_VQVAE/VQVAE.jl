using Lux
using Random
using Statistics: mean, std
using Optimisers
import MLDatasets: FashionMNIST as MNIST
using ComponentArrays: ComponentArray
using MLUtils: DataLoader
using Zygote
using ImageCore: colorview, Gray, RGB
using ImageInTerminal
using ChainRulesCore: ignore_derivatives

TRAIN = MNIST(:train).features
TEST = MNIST(:test).features

function samples(n=10)
    xs = TRAIN[:, :, rand(1:size(TRAIN)[end], n)]
    xs = permutedims(xs, (2, 1, 3))
    xs = reshape(xs, size(xs, 1), :)
    colorview(Gray, xs)
end

#####

struct Codebook <: LuxCore.AbstractLuxContainerLayer{(:codebook,)}
    codebook::Embedding
end

function (m::Codebook)(x, ps, st)
    # WHCB -> CWHB
    x = permutedims(x, (3, 1, 2, 4))
    # add dimension for broadcasting
    x = reshape(x, size(x, 1), 1, size(x)[2:end]...)
    distances = dropdims(sum((ps.codebook.weight .- x) .^ 2, dims=1), dims=1)
    I = argmin(distances, dims=1)
    indices = getindex.(I, 1)
    # println("mean indices $(mean(distances[I]))")
    quantized, st_cb = m.codebook(indices, ps.codebook, st.codebook)
    # drop dimension & CWHB -> WHCB
    permutedims(dropdims(quantized, dims=2), (2, 3, 1, 4)), (codebook=st_cb,)
end

#####

@kwdef struct VQVAE{C,E,D} <: LuxCore.AbstractLuxContainerLayer{(:encoder, :decoder, :codebook)}
    codebook::C
    encoder::E
    decoder::D
end

function (m::VQVAE)(x, ps, st)
    z, st_encoder = m.encoder(x, ps.encoder, st.encoder)
    quantized, st_codebook = m.codebook(z, ps.codebook, st.codebook)
    q = z + ignore_derivatives(quantized - z)
    x̂, st_decoder = m.decoder(q, ps.decoder, st.decoder)
    (x̂, z, quantized), (codebook=st_codebook, encoder=st_encoder, decoder=st_decoder,)
end

m = VQVAE(
    codebook=Codebook(Embedding(256 => 32, init_weight=(rng, out_dims, in_dims) -> (rand(rng, Float32, out_dims, in_dims) .* 2 .- 1) ./ in_dims)),
    encoder=Chain(
        Conv((3, 3), 1 => 16, stride=1, pad=1),
        MaxPool((2, 2)),
        WrappedFunction(Base.Fix1(broadcast, gelu)),
        Conv((3, 3), 16 => 32, stride=1, pad=1),
        MaxPool((2, 2)),
    ),
    decoder=Chain(
        Upsample(2),
        Conv((3, 3), 32 => 16, stride=1, pad=1),
        WrappedFunction(Base.Fix1(broadcast, gelu)),
        Upsample(2),
        Conv((3, 3), 16 => 1, stride=1, pad=1),
        WrappedFunction(x -> clamp.(x, -1, 1)),
    )
)

reconstruct_loss_fn = MAELoss()
vq_loss_fn = MSELoss()
opt = AdamW(3.0f-4)

function train(batch_size=256, n_epoch=30, alpha=10, beta=0.95f0)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, m)
    ps = ps |> ComponentArray

    opt_st = Optimisers.setup(opt, ps)

    n_update = 0
    for epoch in 1:n_epoch
        for x in DataLoader(TRAIN; batchsize=batch_size, shuffle=true, rng=rng)
            x = (reshape(x, size(x)[1:2]..., 1, size(x, 3)) .- 0.5f0) ./ 0.5f0

            grads = Zygote.gradient(ps) do ps
                (x̂, z, q), st = m(x, ps, st)
                recon_loss = reconstruct_loss_fn(x̂, x)
                vq_loss_a = vq_loss_fn(z, ignore_derivatives(q))
                vq_loss_b = vq_loss_fn(ignore_derivatives(z), q)
                if n_update % 10 == 0
                    println("Epoch $epoch Iter $n_update \t Recon Loss: $(recon_loss) VQ Loss A: $(vq_loss_a) VQ Loss B: $(vq_loss_b)")
                end
                recon_loss + alpha * ((1 - beta) * vq_loss_a + beta * vq_loss_b)
            end
            Optimisers.update!(opt_st, ps, grads[1])

            n_update += 1
        end
    end
    ps, st, opt_st
end

function compare(ps, st, n=10)
    st = Lux.testmode(st)

    ids = rand(1:size(TEST, 3), n)
    x = TEST[:, :, ids]
    x′ = (reshape(x, size(TEST)[1:2]..., 1, n) .- 0.5f0) ./ 0.5f0
    (x̂, _, _), _ = m(x′, ps, st)
    x̂ = reshape((x̂ .+ 1) ./ 2, size(x)...)
    # colorview(Gray, reshape(permutedims(x, (2, 1, 3)), size(TEST, 2), :)), colorview(Gray, reshape(permutedims(x̂, (2, 1, 3)), size(TEST, 2), :))
    colorview(Gray, vcat(reshape(permutedims(x, (2, 1, 3)), size(TEST, 2), :), reshape(permutedims(x̂, (2, 1, 3)), size(TEST, 2), :)))
end
