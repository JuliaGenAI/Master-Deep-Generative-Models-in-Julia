using FileIO
using ImageCore: channelview
using SafeTensors: load_sharded_safetensors

import HuggingFaceTokenizers as HFT
using NNlib: batched_mul, gelu_tanh, conv, DenseConvDims, softmax, meanpool, PoolDims
using Statistics: mean, var

#####
# Processor
#####
struct Processor
    boi_token::String
    eoi_token::String
    image_token::String
    mm_tokens_per_image::Int

    height::Int
    width::Int
    mean::Vector{Float32}
    std::Vector{Float32}

    tokenizer::Any
end


(p::Processor)(text::String, images::String) = p(text, [images])
function (p::Processor)(text::String, images::Vector{String})
    # !!! Pan-and-Scan crops are ignored here for simplicity
    imgs = cat([channelview(load(img)) for img in images]...; dims=4) # (C, H, W, B)
    # The default BICUBIC interpolation in PIL is not found in Julia, so here we assume the image input is already resized for reproducibility
    @assert size(imgs) == (3, p.height, p.width, length(images))
    imgs = Float32.((imgs .- reshape(p.mean, 3, 1, 1, 1)) ./ reshape(p.std, 3, 1, 1, 1))

    # Here we assume only one image is present in the text for simplicity
    @assert length(findall(p.boi_token, text)) == 1
    expanded_text = replace(text, p.boi_token => "\n\n$(p.boi_token)$(repeat(p.image_token, p.mm_tokens_per_image))$(p.eoi_token)\n\n")
    tids = HFT.encode(p.tokenizer, expanded_text).ids .+ 1
    tids, imgs
end

#####
# SigLIP
#####

struct CrossCor
    weight::AbstractArray
    bias::AbstractVector
    stride::Int
end

function (c::CrossCor)(x)
    x = permutedims(x, (3, 2, 1, 4))
    y = conv(x, c.weight, DenseConvDims(x, c.weight; stride=c.stride, flipkernel=true))
    y = permutedims(y, (3, 2, 1, 4))
    y .+ c.bias
end

struct SiglipVisionEmbedding
    patch_embedding::CrossCor
    pos_embedding::Matrix
end

function (m::SiglipVisionEmbedding)(x)
    patch_emb = m.patch_embedding(x)
    patch_emb = permutedims(patch_emb, (1, 3, 2, 4)) # !!! align with the order of `pos_emb``
    patch_emb = reshape(patch_emb, size(patch_emb, 1), :, size(patch_emb, 4))
    patch_emb .+ m.pos_embedding
end

struct LayerNorm
    γ::Vector{Float32}
    β::Vector{Float32}
    ϵ::Float32
end

(m::LayerNorm)(x) = m.γ .* (x .- mean(x, dims=1)) ./ sqrt.(var(x, dims=1) .+ m.ϵ) .+ m.β

struct Dense{T}
    weight::AbstractMatrix
    bias::T
end

(m::Dense{Nothing})(x::Matrix) = m.weight * x
(m::Dense)(x::Matrix) = m.weight * x .+ m.bias
(m::Dense)(x) = reshape(m(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

struct SiglipAttention
    q_proj::Dense
    k_proj::Dense
    v_proj::Dense
    o_proj::Dense

    n_heads::Int
end

function (m::SiglipAttention)(x)
    q, k, v = m.q_proj(x), m.k_proj(x), m.v_proj(x)
    q = reshape(q, :, m.n_heads, size(q, 2), size(q, 3))
    k = reshape(k, :, m.n_heads, size(k, 2), size(k, 3))
    v = reshape(v, :, m.n_heads, size(v, 2), size(v, 3))

    q = permutedims(q, (1, 3, 2, 4))
    kᵗ = permutedims(k, (3, 1, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))

    scale = convert(eltype(q), 1 / sqrt(size(q, 1)))
    logits = batched_mul(kᵗ, q) .* scale
    scores = softmax(logits)
    out = batched_mul(v, scores)
    o = permutedims(out, (1, 3, 2, 4))
    o = reshape(o, :, size(o, 3), size(o, 4))
    m.o_proj(o)
end

struct SiglipFFN
    up_proj::Dense
    down_proj::Dense
end

function (m::SiglipFFN)(x)
    h = m.up_proj(x)
    h = gelu_tanh.(h)
    h = m.down_proj(h)
end

struct SiglipBlock
    self_attn::SiglipAttention
    layer_norm1::LayerNorm
    mlp::SiglipFFN
    layer_norm2::LayerNorm
end

function (m::SiglipBlock)(x)
    h = x
    o = x |> m.layer_norm1 |> m.self_attn
    h = h + o
    o = h |> m.layer_norm2 |> m.mlp
    h + o
end

struct SiglipEncoder
    embedding::SiglipVisionEmbedding
    blocks::Vector{SiglipBlock}
    post_layernorm::LayerNorm
end

function (m::SiglipEncoder)(x)
    h = m.embedding(x)
    for block in m.blocks
        h = block(h)
    end
    m.post_layernorm(h)
end

#####
# MultiModalProjector
#####

struct MeanPool
    k
end

(m::MeanPool)(x) = meanpool(x, PoolDims(x, m.k; stride=m.k))

struct RMSNorm
    eps::Float32
    weight::AbstractVector
end

function (m::RMSNorm)(x::AbstractArray{T}) where {T}
    (1 .+ m.weight) .* T.(Float32.(x) ./ sqrt.(mean(Float32.(x) .^ 2, dims=1) .+ m.eps))
end

struct MultiModalProjector
    patches_per_image::Int

    pooling::MeanPool
    rms_norm::RMSNorm
    vision2text::Dense{Nothing}
end

function (m::MultiModalProjector)(x)
    P, D, B = size(x)
    x = reshape(x, m.patches_per_image, m.patches_per_image, D, B)
    x = m.pooling(x)
    x = reshape(x, :, D, B)
    x = permutedims(x, (2, 1, 3))
    x = m.rms_norm(x)
    m.vision2text(x)
end

#####

struct VisionModel
    encoder::SiglipEncoder
    projector::MultiModalProjector
end


#####

using JSON3

function from_pretrained(MODEL=joinpath(@__DIR__, "..", "models", "google", "gemma-3-4b-it"))
    ps = load_sharded_safetensors(MODEL)
    c = JSON3.read(joinpath(MODEL, "config.json"))
    pc = JSON3.read(joinpath(MODEL, "preprocessor_config.json"))
    stm = JSON3.read(joinpath(MODEL, "special_tokens_map.json"))
    tokenizer = HFT.from_file(HFT.Tokenizer, joinpath(MODEL, "tokenizer.json"))

    p = Processor(
        stm["boi_token"],
        stm["eoi_token"],
        stm["image_token"],
        c["mm_tokens_per_image"],
        pc["size"]["height"],
        pc["size"]["width"],
        pc["image_mean"],
        pc["image_std"],
        tokenizer
    )

    vc = c["vision_config"]

    m = VisionModel(
        SiglipEncoder(
            SiglipVisionEmbedding(
                CrossCor(
                    permutedims(ps["vision_tower.vision_model.embeddings.patch_embedding.weight"], (4, 3, 2, 1)),
                    ps["vision_tower.vision_model.embeddings.patch_embedding.bias"],
                    c["vision_config"]["patch_size"]
                ),
                ps["vision_tower.vision_model.embeddings.position_embedding.weight"]'
            ),
            [
                SiglipBlock(
                    SiglipAttention(
                        Dense(
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.q_proj.weight"],
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.q_proj.bias"],
                        ),
                        Dense(
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.k_proj.weight"],
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.k_proj.bias"],
                        ),
                        Dense(
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.v_proj.weight"],
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.v_proj.bias"],
                        ),
                        Dense(
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.out_proj.weight"],
                            ps["vision_tower.vision_model.encoder.layers.$i.self_attn.out_proj.bias"],
                        ),
                        vc["num_attention_heads"]
                    ),
                    LayerNorm(
                        ps["vision_tower.vision_model.encoder.layers.$i.layer_norm1.weight"],
                        ps["vision_tower.vision_model.encoder.layers.$i.layer_norm1.bias"],
                        vc["layer_norm_eps"]
                    ),
                    SiglipFFN(
                        Dense(
                            ps["vision_tower.vision_model.encoder.layers.$i.mlp.fc1.weight"],
                            ps["vision_tower.vision_model.encoder.layers.$i.mlp.fc1.bias"],
                        ),
                        Dense(
                            ps["vision_tower.vision_model.encoder.layers.$i.mlp.fc2.weight"],
                            ps["vision_tower.vision_model.encoder.layers.$i.mlp.fc2.bias"],
                        ),
                    ),
                    LayerNorm(
                        ps["vision_tower.vision_model.encoder.layers.$i.layer_norm2.weight"],
                        ps["vision_tower.vision_model.encoder.layers.$i.layer_norm2.bias"],
                        vc["layer_norm_eps"]
                    )
                )
                for i in 0:vc["num_hidden_layers"]-1
            ],
            LayerNorm(
                ps["vision_tower.vision_model.post_layernorm.weight"],
                ps["vision_tower.vision_model.post_layernorm.bias"],
                vc["layer_norm_eps"]
            )
        ),
        let n_patches = vc["image_size"] ÷ vc["patch_size"], kernel_size = n_patches ÷ sqrt(c["mm_tokens_per_image"])
            MultiModalProjector(
                n_patches,
                MeanPool((kernel_size, kernel_size)),
                RMSNorm(vc["layer_norm_eps"], ps["multi_modal_projector.mm_soft_emb_norm.weight"]),
                Dense(ps["multi_modal_projector.mm_input_projection_weight"]', nothing)
            )
        end
    )
    ps, p, m
end