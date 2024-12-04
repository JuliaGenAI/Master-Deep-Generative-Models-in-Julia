#=
Adapted from https://github.com/LuxDL/Lux.jl/blob/main/examples/SimpleChains/main.jl

Everything is explicit. Concepts like `TrainState` are removed to keep the code simple.
=#
using Random
using Lux
using MLUtils: DataLoader, splitobs
import MLDatasets: FashionMNIST as MNIST
using Optimisers
using OneHotArrays
using Printf: @printf
using ComponentArrays: ComponentArray
using Zygote
using Statistics: mean
using ImageCore: colorview, Gray
using ImageInTerminal
using UnicodePlots

TRAIN, TEST = MNIST(:train), MNIST(:test)

m = Chain(
    FlattenLayer(2),
    Dense(784 => 20, relu),
    Dense(20 => 10)
)

loss_fn = CrossEntropyLoss(; logits=true)

opt = Adam()

function acc(m, ps, st)
    st = Lux.testmode(st)
    xs = TEST.features
    y = TEST.targets
    ŷ, _ = m(xs, ps, st)
    mean(onecold(ŷ) .== (1 .+ y)) # !!! original labels start from 0
end

function train(n_epoch=10, batchsize=128)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, m)
    ps = ps |> ComponentArray

    opt_st = Optimisers.setup(opt, ps)

    losses = []

    for epoch in 0:n_epoch-1
        @printf "Epoch %d: Test accuracy %.4f\n" epoch acc(m, ps, st)
        for (xs, y) in DataLoader((TRAIN.features, TRAIN.targets);
            batchsize=batchsize, shuffle=true, rng=rng
        )
            y = onehotbatch(y, 0:9)
            loss, grads = Zygote.withgradient(ps) do ps
                ŷ, st = m(xs, ps, st)
                loss_fn(ŷ, y)
            end
            push!(losses, loss)
            Optimisers.update!(opt_st, ps, grads[1])
        end
    end
    @printf "Epoch %d: Test accuracy %.4f\n" n_epoch acc(m, ps, st)
    ps, st, opt_st, losses
end

ps, st, opt_st, losses = train()

i = rand(1:size(TEST.features, 3))
label = 1 + TEST.targets[i]
prediction = onecold(m(view(TEST.features, :, :, i:i), ps, st)[1])[]
println(lineplot(losses, title="Losses"))
println("Label: $(TEST.metadata["class_names"][label])\tPrediction: $(TEST.metadata["class_names"][prediction])")
colorview(Gray, TEST.features[:, :, i]')