using LinearAlgebra: triu
using NNlib: batched_mul, softmax, gelu
using Statistics: mean, var

struct EmbeddingLayer
    token_embedding::Matrix
    position_embedding::Matrix
end

function (m::EmbeddingLayer)(ids)
    pe = m.position_embedding[:, axes(ids, 1)]
    te = m.token_embedding[:, ids]
    pe .+ te
end

struct Dense{T}
    weight::AbstractMatrix
    bias::T
end

(m::Dense{Nothing})(x::Matrix) = m.weight * x
(m::Dense)(x::Matrix) = m.weight * x .+ m.bias
(m::Dense)(x) = reshape(m(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

struct LayerNorm
    weight::Vector
    bias::Vector
    eps::Float32
end

(m::LayerNorm)(x) = m.weight .* (x .- mean(x, dims=1)) ./ sqrt.(var(x, dims=1) .+ m.eps) .+ m.bias

struct Attention
    n_heads::Int
    qkv_proj::Dense
    o_proj::Dense
end

function (m::Attention)(x)
    D, S, B = size(x)
    qkv = m.qkv_proj(x)
    qkv = reshape(qkv, :, m.n_heads, 3, size(qkv, 2), size(qkv, 3))
    q, k, v = (@view(qkv[:, :, i, :, :]) for i in 1:3)

    q = permutedims(q, (1, 3, 2, 4))
    kᵀ = permutedims(k, (3, 1, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))

    mask = triu(ones(Bool, S, S))
    logits = batched_mul(kᵀ, q ./ sqrt(size(q, 1)))
    masked_logits = ifelse.(mask, logits, typemin(eltype(x)))
    scores = softmax(masked_logits)
    o = batched_mul(v, scores)
    o = reshape(permutedims(o, (1, 3, 2, 4)), :, S, B)

    m.o_proj(o)
end

struct FeedForward
    up_proj::Dense
    down_proj::Dense
end

(m::FeedForward)(x) = m.down_proj(gelu.(m.up_proj(x)))

struct TransformerBlock
    pre_attn_norm::LayerNorm
    self_attn::Attention
    pre_ffn_norm::LayerNorm
    ffn::FeedForward
end

function (m::TransformerBlock)(x)
    h = x |> m.pre_attn_norm |> m.self_attn
    h = x + h
    o = h |> m.pre_ffn_norm |> m.ffn
    h = o + h
end

struct Transformer
    embedding::EmbeddingLayer
    blocks::Vector{TransformerBlock}
    layer_norm::LayerNorm
    head::Dense
end

function (m::Transformer)(x)
    h = m.embedding(x)
    for block in m.blocks
        h = block(h)
    end
    h = m.layer_norm(h)
    h = m.head(h)
end

#####

using SafeTensors: load_safetensors
using JSON3
import HuggingFaceTokenizers as HFT

function generate(model, tokenizer, prompt; n_tokens=30)
    tokens = HFT.encode(tokenizer, prompt).ids .+ 1
    for _ in 1:n_tokens
        next_token = argmax(model(reshape(tokens, :, 1))[:, end, 1])
        push!(tokens, next_token)
    end
    HFT.decode(tokenizer, tokens .- 1)
end

function from_pretrained(MODEL=joinpath(@__DIR__, "..", "models", "openai-community", "gpt2"))
    c = JSON3.read(joinpath(MODEL, "config.json"))
    ps = load_safetensors(joinpath(MODEL, "model.safetensors"))
    tokenizer = HFT.from_file(HFT.Tokenizer, joinpath(MODEL, "tokenizer.json"))
    model = Transformer(
        EmbeddingLayer(ps["wte.weight"]', ps["wpe.weight"]'),
        [
            TransformerBlock(
                LayerNorm(ps["h.$i.ln_1.weight"], ps["h.$i.ln_1.bias"], c[:layer_norm_epsilon]),
                Attention(
                    c[:n_head],
                    Dense(ps["h.$i.attn.c_attn.weight"]', ps["h.$i.attn.c_attn.bias"]),
                    Dense(ps["h.$i.attn.c_proj.weight"]', ps["h.$i.attn.c_proj.bias"])
                ),
                LayerNorm(ps["h.$i.ln_2.weight"], ps["h.$i.ln_2.bias"], c[:layer_norm_epsilon]),
                FeedForward(
                    Dense(ps["h.$i.mlp.c_fc.weight"]', ps["h.$i.mlp.c_fc.bias"]),
                    Dense(ps["h.$i.mlp.c_proj.weight"]', ps["h.$i.mlp.c_proj.bias"])
                ),
            ) for i in 0:c[:n_layer]-1
        ],
        LayerNorm(ps["ln_f.weight"], ps["ln_f.bias"], c[:layer_norm_epsilon]),
        Dense(ps["wte.weight"], nothing)
    )
    model, tokenizer
end

function main()
    m, t = from_pretrained()
    s = generate(m, t, "The Julia programming language")
    println(s)
end