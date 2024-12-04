using Lux
using Random
using Optimisers
import MLDatasets: FashionMNIST as MNIST
using ComponentArrays: ComponentArray
using MLUtils: DataLoader
using Zygote
using ImageCore: colorview, Gray
using ImageInTerminal

TRAIN, TEST = MNIST(:train).features, MNIST(:test).features

struct GAN{G,D} <: LuxCore.AbstractLuxContainerLayer{(:generator, :discriminator)}
    generator::G
    discriminator::D
end

function (m::GAN)(z, ps, st)
    x, st_gen = m.generator(z, ps.generator, st.generator)
    y, st_disc = m.discriminator(x, ps.discriminator, st.discriminator)
    y, (generator=st_gen, discriminator=st_disc)
end

m = GAN(
    Chain(
        Dense(100 => 32, x -> leakyrelu(x, 0.2)),
        Dropout(0.3),
        Dense(32 => 64, x -> leakyrelu(x, 0.2)),
        Dropout(0.3),
        Dense(64 => 128, x -> leakyrelu(x, 0.2)),
        Dropout(0.3),
        Dense(128 => 784, tanh),
    ),
    Chain(
        Dense(784 => 128, x -> leakyrelu(x, 0.2)),
        Dropout(0.3),
        Dense(128 => 64, x -> leakyrelu(x, 0.2)),
        Dropout(0.3),
        Dense(64 => 32, x -> leakyrelu(x, 0.2)),
        Dropout(0.3),
        Dense(32 => 1),
    )
)

loss_fn = BinaryCrossEntropyLoss(; logits=true)
g_opt = Adam()
d_opt = Adam()

function train(n_epochs=50, batchsize=64)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, m)
    ps = ps |> ComponentArray

    n_update = 0

    let ps_d = ps.discriminator, st_d = st.discriminator, ps_g = ps.generator, st_g = st.generator
        opt_st_d = Optimisers.setup(d_opt, ps_d)
        opt_st_g = Optimisers.setup(g_opt, ps_g)

        for epoch in 1:n_epochs
            for x in DataLoader(TRAIN; batchsize=batchsize, shuffle=true, rng=rng)
                x_real = reshape(x, :, size(x, 3)) .* 2 .- 1

                z = rand(rng, Float32, (100, batchsize)) .* 2 .- 1
                x_fake, _ = m.generator(z, ps_g, Lux.testmode(st_g))

                grads, = Zygote.gradient(ps_d) do ps_d
                    y_real, st_d = m.discriminator(x_real, ps_d, st_d)
                    y_fake, st_d = m.discriminator(x_fake, ps_d, st_d)
                    loss_real = loss_fn(y_real, ones(size(y_real)))
                    loss_fake = loss_fn(y_fake, zeros(size(y_fake)))
                    if n_update % 100 == 0
                        println("Epoch $epoch Iter $n_update \t discriminator real loss $loss_real fake loss $loss_fake")
                    end
                    loss_real + loss_fake
                end
                opt_st_d, ps_d = Optimisers.update!(opt_st_d, ps_d, grads)

                z = rand(rng, Float32, (100, batchsize)) .* 2 .- 1
                grads, = Zygote.gradient(ps_g) do ps_g
                    x_fake, st_g = m.generator(z, ps_g, st_g)
                    y, _ = m.discriminator(x_fake, ps_d, st_d)
                    L = loss_fn(y, ones(size(y)))
                    if n_update % 10 == 0
                        println("Epoch $epoch Iter $n_update \t generator loss $L")
                    end
                    L
                end
                opt_st_g, ps_g = Optimisers.update!(opt_st_g, ps_g, grads)

                n_update += 1
            end
        end
        ps = (generator=ps_g, discriminator=ps_d)
        st = (generator=st_g, discriminator=st_d)
    end
    ps, st
end