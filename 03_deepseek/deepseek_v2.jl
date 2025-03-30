using SafeTensors: load_sharded_safetensors
using Statistics: mean
using NNlib: batched_mul, softmax, swish
using LinearAlgebra: triu
using JSON3

struct EmbeddingLayer
    weight::AbstractMatrix
end

(m::EmbeddingLayer)(ids) = m.weight[:, ids]

struct Dense
    weight::AbstractMatrix
end

(m::Dense)(x::AbstractMatrix) = m.weight * x
(m::Dense)(x) = reshape(m(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

struct RMSNorm
    eps::Float32
    weight::AbstractVector
end

function (m::RMSNorm)(x::AbstractArray{T}) where {T}
    m.weight .* T.(Float32.(x) ./ sqrt.(mean(Float32.(x) .^ 2, dims=1) .+ m.eps))
end

#####
# YaRN
#####

# Eq 17 in YaRN paper, given r of α or β, find the corresponding d
r2d(r, D, b, L) = (D * log(L / (2π * r))) / (2log(b))

# (Modified) Eq 22 in YaRN paper
yarn_mscale(scale, mscale) = scale <= 1.0 ? 1.0 : 0.1 * mscale * log(scale) + 1.0

struct RopeCache
    scaling_factor::Float32
    mscale::Float32
    mscale_all_dim::Float32
    cos::Array
    sin::Array
end

function RopeCache(; dim, rope_theta, max_position_embeddings, scaling_factor, beta_slow, beta_fast, original_max_position_embeddings, mscale, mscale_all_dim)
    freq_extra = 1 ./ (rope_theta .^ ((0:2:(dim-1)) ./ dim))
    freq_inter = 1 ./ (scaling_factor .* rope_theta .^ ((0:2:(dim-1)) ./ dim))

    low = floor(Int, r2d(beta_fast, dim, rope_theta, original_max_position_embeddings))
    high = ceil(Int, r2d(beta_slow, dim, rope_theta, original_max_position_embeddings))
    mask = vcat(zeros(low), range(0, 1, length=high - low + 1), ones(dim ÷ 2 - high - 1))

    inv_freq = freq_inter .* mask .+ freq_extra .* (1 .- mask)
    t = 0:(max_position_embeddings-1)
    freqs = inv_freq * t'

    # !!! mscal is skipped here.
    # see https://github.com/ggml-org/llama.cpp/discussions/7416
    RopeCache(scaling_factor, mscale, mscale_all_dim, cos.(freqs), sin.(freqs))
end


#####
# MLA
#####

struct Attention
    num_heads::Int
    qk_nope_head_dim::Int
    kv_lora_rank::Int

    q_proj::Dense
    kv_proj::Dense
    kv_layernorm::RMSNorm
    k_v_proj::Dense
    o_proj::Dense

    rope_cache::RopeCache
end

function apply_rotary_pos_emb(x, cache)
    D, H, T, B = size(x)
    freqs_cos = reshape(cache.cos[:, 1:T], :, 1, T)
    freqs_sin = reshape(cache.sin[:, 1:T], :, 1, T)

    # !!! different from the llama implementation
    x_r = selectdim(reshape(x, 2, :, size(x)[2:end]...), 1, 1)
    x_i = selectdim(reshape(x, 2, :, size(x)[2:end]...), 1, 2)

    x_pos_r = freqs_cos .* x_r .- freqs_sin .* x_i
    x_pos_i = freqs_sin .* x_r .+ freqs_cos .* x_i
    vcat(x_pos_r, x_pos_i)
end

function (m::Attention)(h)
    D, T, B = size(h)

    cQ = m.q_proj(h)  # in inference mode, the weight is alreayd merged
    cQ = reshape(cQ, :, m.num_heads, T, B)
    qᶜ = @view cQ[1:m.qk_nope_head_dim, :, :, :]
    qᴿ = @view cQ[m.qk_nope_head_dim+1:end, :, :, :]
    qᴿ = apply_rotary_pos_emb(qᴿ, m.rope_cache)
    q = vcat(qᶜ, qᴿ)

    cKV = m.kv_proj(h)
    kᴿ = @view cKV[m.kv_lora_rank+1:end, :, :]
    kᴿ = reshape(kᴿ, :, 1, T, B)
    kᴿ = apply_rotary_pos_emb(kᴿ, m.rope_cache)

    cᴷⱽ = @view cKV[1:m.kv_lora_rank, :, :]
    cᴷⱽ = cᴷⱽ |> m.kv_layernorm |> m.k_v_proj
    cᴷⱽ = reshape(cᴷⱽ, :, m.num_heads, T, B)
    kᶜ = @view cᴷⱽ[1:m.qk_nope_head_dim, :, :, :]
    kᴿ = repeat(kᴿ, inner=(1, m.num_heads, 1, 1))
    k = vcat(kᶜ, kᴿ)

    v = @view cᴷⱽ[m.qk_nope_head_dim+1:end, :, :, :]

    # scaled dot product attention
    q = permutedims(q, (1, 3, 2, 4))
    kᵗ = permutedims(k, (3, 1, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))

    scale = size(q, 1)^(-0.5)
    # !!! mscale applied here instead
    mscale = yarn_mscale(m.rope_cache.scaling_factor, m.rope_cache.mscale_all_dim)
    scale = scale * mscale^2
    logits = batched_mul(kᵗ, q .* scale)

    mask = triu(ones(Bool, T, T))

    masked_logits = ifelse.(mask, logits, typemin(eltype(logits)))
    scores = softmax(masked_logits)
    out = batched_mul(v, scores)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, :, size(out, 3), size(out, 4))
    o = m.o_proj(out)

    return o
end

#####

struct FeedForwardLayer
    up_proj::Dense
    gate_proj::Dense
    down_proj::Dense
end

function (m::FeedForwardLayer)(x)
    h = m.up_proj(x)
    a = m.gate_proj(x)
    h = swish.(a) .* h
    o = m.down_proj(h)
end

struct MoELayer
    topk::Int
    routed_scaling_factor::Float32

    gate::Dense
    experts::Vector{FeedForwardLayer}
    shared_experts::FeedForwardLayer
end

function (m::MoELayer)(h)
    hh = h
    h = reshape(h, size(h, 1), :)

    scores = softmax(m.gate(h))
    topk_linear_idx = sortperm(scores, dims=1, rev=true)[1:m.topk, :]
    topk_weight = scores[topk_linear_idx] .* m.routed_scaling_factor
    topk_cartesian_idx = CartesianIndices(size(scores))[topk_linear_idx]

    expert_out = similar(h, size(h, 1), m.topk, size(h, 2))
    for i in 1:length(m.experts)
        expert = m.experts[i]
        # !!! original implementation use the argsort trick
        # the implementation here is less efficient but more clear 
        loc = findall(x -> x[1] == i, topk_cartesian_idx)
        token_ids = map(x -> x[2], topk_cartesian_idx[loc])
        expert_input = h[:, token_ids]
        expert_output = expert(expert_input)
        expert_out[:, loc] .= expert_output
    end

    o = expert_out .* reshape(topk_weight, 1, size(topk_weight)...)
    o = sum(o, dims=2)

    reshape(o, size(hh)...) .+ m.shared_experts(hh)
end

struct TransformerBlock
    pre_norm::RMSNorm
    attn::Attention
    post_norm::RMSNorm
    ffn::Union{FeedForwardLayer,MoELayer}
end

function (m::TransformerBlock)(x)
    h = x |> m.pre_norm |> m.attn
    o = (x + h) |> m.post_norm |> m.ffn
    x + h + o
end

struct DeepSeekV2
    embedding_layer::EmbeddingLayer
    transformer_blocks::Vector{TransformerBlock}
    norm_layer::RMSNorm
    head::Dense
end

function (m::DeepSeekV2)(x)
    h = m.embedding_layer(x)
    for (i, block) in enumerate(m.transformer_blocks)
        h = block(h)
    end
    h = m.norm_layer(h)
    o = m.head(h)
end

#####

import HuggingFaceTokenizers as HFT
using ProgressMeter

function generate(model, tokenizer, prompt; n_tokens=30, bos_token_id=100000)
    tokens = HFT.encode(tokenizer, prompt).ids .+ 1
    pushfirst!(tokens, bos_token_id + 1)
    @showprogress for _ in 1:n_tokens
        next_token = argmax(model(reshape(tokens, :, 1))[:, end, 1])
        push!(tokens, next_token)
    end
    HFT.decode(tokenizer, tokens .- 1)
end

function from_pretrained(MODEL=joinpath(@__DIR__, "..", "models", "deepseek-ai", "DeepSeek-V2-Lite-Chat"))
    ps = load_sharded_safetensors(MODEL)
    c = JSON3.read(joinpath(MODEL, "config.json"))
    tokenizer = HFT.from_file(HFT.Tokenizer, joinpath(MODEL, "tokenizer.json"))
    rope_cache = RopeCache(
        dim=c["qk_rope_head_dim"],
        rope_theta=c["rope_theta"],
        max_position_embeddings=c["max_position_embeddings"],
        scaling_factor=c["rope_scaling"]["factor"],
        beta_slow=c["rope_scaling"]["beta_slow"],
        beta_fast=c["rope_scaling"]["beta_fast"],
        original_max_position_embeddings=c["rope_scaling"]["original_max_position_embeddings"],
        mscale=c["rope_scaling"]["mscale"],
        mscale_all_dim=c["rope_scaling"]["mscale_all_dim"],
    )
    model = DeepSeekV2(
        EmbeddingLayer(ps["model.embed_tokens.weight"]'),
        [
            TransformerBlock(
                RMSNorm(c["rms_norm_eps"], ps["model.layers.$i.input_layernorm.weight"]),
                Attention(
                    c["num_attention_heads"],
                    c["qk_nope_head_dim"],
                    c["kv_lora_rank"],
                    Dense(ps["model.layers.$i.self_attn.q_proj.weight"]),
                    Dense(ps["model.layers.$i.self_attn.kv_a_proj_with_mqa.weight"]),
                    RMSNorm(c["rms_norm_eps"], ps["model.layers.$i.self_attn.kv_a_layernorm.weight"]),
                    Dense(ps["model.layers.$i.self_attn.kv_b_proj.weight"]),
                    Dense(ps["model.layers.$i.self_attn.o_proj.weight"]),
                    rope_cache
                ),
                RMSNorm(c["rms_norm_eps"], ps["model.layers.$i.post_attention_layernorm.weight"]),
                i < c["first_k_dense_replace"] ? FeedForwardLayer(
                    Dense(ps["model.layers.$i.mlp.up_proj.weight"]),
                    Dense(ps["model.layers.$i.mlp.gate_proj.weight"]),
                    Dense(ps["model.layers.$i.mlp.down_proj.weight"])
                ) : MoELayer(
                    c["num_experts_per_tok"],
                    c["routed_scaling_factor"],
                    Dense(ps["model.layers.$i.mlp.gate.weight"]),
                    [
                        FeedForwardLayer(
                            Dense(ps["model.layers.$i.mlp.experts.$j.up_proj.weight"]),
                            Dense(ps["model.layers.$i.mlp.experts.$j.gate_proj.weight"]),
                            Dense(ps["model.layers.$i.mlp.experts.$j.down_proj.weight"])
                        )
                        for j in 0:c["n_routed_experts"]-1
                    ],
                    FeedForwardLayer(
                        Dense(ps["model.layers.$i.mlp.shared_experts.up_proj.weight"]),
                        Dense(ps["model.layers.$i.mlp.shared_experts.gate_proj.weight"]),
                        Dense(ps["model.layers.$i.mlp.shared_experts.down_proj.weight"])
                    )
                )
            )
            for i in 0:c["num_hidden_layers"]-1
        ],
        RMSNorm(c["rms_norm_eps"], ps["model.norm.weight"]),
        Dense(ps["lm_head.weight"])
    )
    model, tokenizer, c
end

function main()
    m, t, c = from_pretrained()
    p = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    s = generate(m, t, p; bos_token_id=c["bos_token_id"])
    println(s)
end