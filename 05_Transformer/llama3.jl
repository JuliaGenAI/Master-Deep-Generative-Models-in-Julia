using Lux
using Pickle.Torch: THload
import HuggingFaceHub as HF
using NNlib: batched_mul, softmax
using LinearAlgebra: triu
using Statistics: mean
using Random

MODEL_ID = "karpathy/tinyllamas"
HF_TINYLLAMAS = HF.info(HF.Model(id=MODEL_ID))
CHECKPOINT = THload(HF.file_download(HF_TINYLLAMAS, "stories260K/stories260K.pt"))
# CHECKPOINT = THload(HF.file_download(HF_TINYLLAMAS, "stories15M.pt"))
# CHECKPOINT = THload(HF.file_download(HF_TINYLLAMAS, "stories110M.pt"))
ARGS = CHECKPOINT["model_args"]

using PythonCall
using CondaPkg
CondaPkg.add("sentencepiece")
sentencepiece = pyimport("sentencepiece")
SP_MODEL = sentencepiece.SentencePieceProcessor(model_file=HF.file_download(HF_TINYLLAMAS, "stories260K/tok512.model"))
BOS_ID = pyconvert(Int, SP_MODEL.bos_id()) + 1
EOS_ID = pyconvert(Int, SP_MODEL.eos_id()) + 1
PAD_ID = pyconvert(Int, SP_MODEL.pad_id()) + 1

function encode(s, bos=true, eos=false)
    t = pyconvert(Vector{Int}, SP_MODEL.encode(s))
    bos && pushfirst!(t, BOS_ID)
    eos && push!(t, EOS_ID)
    t
end

decode(xs) = SP_MODEL.decode(pylist([x - 1 for x in xs]))

# TODO: dispatch to CUDA.sqrt ?
rsqrt(x::T) where {T} = one(T) / sqrt(x)

function cos_sin_cache(head_dim, max_seq_len, base=10_000)
    inv_freq = 1.0f0 ./ (base .^ ((0:2:head_dim-1) .* 1.0f0 ./ head_dim))
    freqs = reshape(inv_freq, :, 1) * reshape(0:max_seq_len-1, 1, :)
    cos.(freqs), sin.(freqs)
end

function apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    freqs_cos = reshape(freqs_cos, size(freqs_cos, 1), 1, size(freqs_cos, 2), 1)
    freqs_sin = reshape(freqs_sin, size(freqs_sin, 1), 1, size(freqs_sin, 2), 1)

    H = size(xq, 1)
    xq_r = selectdim(reshape(xq, 2, :, size(xq)[2:end]...), 1, 1)
    xq_i = selectdim(reshape(xq, 2, :, size(xq)[2:end]...), 1, 2)

    xk_r = selectdim(reshape(xk, 2, :, size(xk)[2:end]...), 1, 1)
    xk_i = selectdim(reshape(xk, 2, :, size(xk)[2:end]...), 1, 2)

    xq_out_r = freqs_cos .* xq_r .- freqs_sin .* xq_i
    xq_out_i = freqs_sin .* xq_r .+ freqs_cos .* xq_i

    xk_out_r = freqs_cos .* xk_r .- freqs_sin .* xk_i
    xk_out_i = freqs_sin .* xk_r .+ freqs_cos .* xk_i

    vcat(xq_out_r, xq_out_i), vcat(xk_out_r, xk_out_i)
end

struct RMSNorm <: LuxCore.AbstractLuxLayer
    n::Int
    eps::Float32
end

RMSNorm(n::Int; eps=1.0e-5) = RMSNorm(n, eps)

LuxCore.initialparameters(rng::AbstractRNG, l::RMSNorm) = (; weight=ones(Float32, l.n))
LuxCore.initialstates(rng::AbstractRNG, l::RMSNorm) = (;)

function (m::RMSNorm)(x::X, ps, st) where {X}
    out = ps.weight .* rsqrt.(mean(x .^ 2; dims=1) .+ m.eps) .* x
    out, st
end

struct Attention <: LuxCore.AbstractLuxContainerLayer{(:wq, :wk, :wv, :wo, :attn_dropout, :resid_dropout)}
    wq::Dense
    wk::Dense
    wv::Dense
    wo::Dense
    attn_dropout::Dropout
    resid_dropout::Dropout
    head_dim::Int
    n_heads::Int
    n_kv_heads::Int
end

function (m::Attention)((x, freqs_cos, freqs_sin), ps, st)
    xq, st_wq = m.wq(x, ps.wq, st.wq)
    xk, st_wk = m.wk(x, ps.wk, st.wk)
    xv, st_wv = m.wv(x, ps.wv, st.wv)

    xq = reshape(xq, m.head_dim, m.n_heads, size(xq, 2), size(xq, 3))
    xk = reshape(xk, m.head_dim, m.n_kv_heads, size(xk, 2), size(xk, 3))
    xv = reshape(xv, m.head_dim, m.n_kv_heads, size(xv, 2), size(xv, 3))

    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    xk = repeat(xk, inner=(1, m.n_heads ÷ m.n_kv_heads, 1, 1))
    xv = repeat(xv, inner=(1, m.n_heads ÷ m.n_kv_heads, 1, 1))

    xq = permutedims(xq, (1, 3, 2, 4))
    xkt = permutedims(xk, (3, 1, 2, 4))
    xv = permutedims(xv, (1, 3, 2, 4))

    seq_len = size(xq, 2)
    mask = triu(ones(Bool, seq_len, seq_len))
    logits = batched_mul(xkt, xq ./ convert(eltype(xq), sqrt(m.head_dim)))
    masked_logits = ifelse.(mask, logits, typemin(eltype(logits)))
    scores = softmax(masked_logits)

    scores, st_attn_dropout = m.attn_dropout(scores, ps.attn_dropout, st.attn_dropout)
    out = batched_mul(xv, scores)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, :, size(out, 3), size(out, 4))

    xo, st_wo = m.wo(out, ps.wo, st.wo)
    xo, st_resid_dropout = m.resid_dropout(xo, ps.resid_dropout, st.resid_dropout)

    xo, (wq=st_wq, wk=st_wk, wv=st_wv, wo=st_wo, attn_dropout=st_attn_dropout, resid_dropout=st_resid_dropout)
end

struct FeedForward <: LuxCore.AbstractLuxContainerLayer{(:w1, :w2, :w3, :dropout)}
    w1::Dense
    w2::Dense
    w3::Dense
    dropout::Dropout
end

function (m::FeedForward)(x, ps, st)
    h, st_w3 = m.w3(x, ps.w3, st.w3)
    a, st_w1 = m.w1(x, ps.w1, st.w1)
    h = swish.(a) .* h
    h, st_w2 = m.w2(h, ps.w2, st.w2)
    out, st_dropout = m.dropout(h, ps.dropout, st.dropout)
    out, (w1=st_w1, w2=st_w2, w3=st_w3, dropout=st_dropout)
end

struct TransformerBlock <: LuxCore.AbstractLuxContainerLayer{(:attention_norm, :attention, :ffn_norm, :feed_forward)}
    attention_norm::RMSNorm
    attention::Attention
    ffn_norm::RMSNorm
    feed_forward::FeedForward
end

function (t::TransformerBlock)((x, freqs_cos, freqs_sin), ps, st)
    h, st_attention_norm = t.attention_norm(x, ps.attention_norm, st.attention_norm)
    h, st_attention = t.attention((h, freqs_cos, freqs_sin), ps.attention, st.attention)
    out = x + h
    h, st_ffn_norm = t.ffn_norm(out, ps.ffn_norm, st.ffn_norm)
    h, st_feed_forward = t.feed_forward(h, ps.feed_forward, st.feed_forward)
    (out + h, freqs_cos, freqs_sin), (attention_norm=st_attention_norm, attention=st_attention, ffn_norm=st_ffn_norm, feed_forward=st_feed_forward)
end

struct Transformer <: LuxCore.AbstractLuxContainerLayer{(:token_embeddings, :dropout, :transformer_blocks, :norm, :output)}
    token_embeddings::Embedding
    dropout::Dropout
    transformer_blocks::Chain
    norm::RMSNorm
    output::Dense
    max_seq_len::Int
end

function LuxCore.initialstates(rng::AbstractRNG, m::Transformer)
    freqs_cos, freqs_sin = cos_sin_cache(m.transformer_blocks.layer_1.attention.head_dim, m.max_seq_len)
    (;
        token_embeddings=LuxCore.initialstates(rng, m.token_embeddings),
        dropout=LuxCore.initialstates(rng, m.dropout),
        transformer_blocks=LuxCore.initialstates(rng, m.transformer_blocks),
        norm=LuxCore.initialstates(rng, m.norm),
        output=LuxCore.initialstates(rng, m.output),
        freqs_cos,
        freqs_sin
    )
end

function (m::Transformer)(tokens, ps, st)
    seqlen, batchsize = size(tokens)
    h, st_token_embeddings = m.token_embeddings(tokens, ps.token_embeddings, st.token_embeddings)
    h, st_dropout = m.dropout(h, ps.dropout, st.dropout)
    freqs_cos, freqs_sin = st.freqs_cos[:, 1:seqlen], st.freqs_sin[:, 1:seqlen]
    (h, _, _), st_transformer_blocks = m.transformer_blocks((h, freqs_cos, freqs_sin), ps.transformer_blocks, st.transformer_blocks)
    h, st_norm = m.norm(h, ps.norm, st.norm)
    h, st_output = m.output(h, ps.output, st.output)
    h, (token_embeddings=st_token_embeddings, dropout=st_dropout, transformer_blocks=st_transformer_blocks, norm=st_norm, output=st_output)
end

function convert_checkpoint(rng, checkpoint)
    ARGS = checkpoint["model_args"]
    model = Transformer(
        Embedding(ARGS["vocab_size"] => ARGS["dim"]),
        Dropout(ARGS["dropout"]),
        Chain(
            (TransformerBlock(
                RMSNorm(ARGS["dim"]),
                let head_dim = ARGS["dim"] ÷ ARGS["n_heads"]
                    Attention(
                        Dense(ARGS["dim"] => head_dim * ARGS["n_heads"], use_bias=false),
                        Dense(ARGS["dim"] => head_dim * ARGS["n_kv_heads"], use_bias=false),
                        Dense(ARGS["dim"] => head_dim * ARGS["n_kv_heads"], use_bias=false),
                        Dense(head_dim * ARGS["n_heads"] => ARGS["dim"], use_bias=false),
                        Dropout(ARGS["dropout"]),
                        Dropout(ARGS["dropout"]),
                        head_dim,
                        ARGS["n_heads"],
                        ARGS["n_kv_heads"]
                    )
                end,
                RMSNorm(ARGS["dim"]),
                let hidden_dim = ARGS["multiple_of"] * ceil(Int, (ARGS["dim"] * 4) * 2 ÷ 3 / ARGS["multiple_of"])
                    FeedForward(
                        Dense(ARGS["dim"] => hidden_dim, use_bias=false),
                        Dense(hidden_dim => ARGS["dim"], use_bias=false),
                        Dense(ARGS["dim"] => hidden_dim, use_bias=false),
                        Dropout(ARGS["dropout"])
                    )
                end
            )
             for _ in 1:ARGS["n_layers"])...
        ),
        RMSNorm(ARGS["dim"]),
        Dense(ARGS["dim"] => ARGS["vocab_size"]; use_bias=false),
        ARGS["max_seq_len"]
    )

    params = checkpoint["model"]
    ps = (;
        token_embeddings=(; weight=params["_orig_mod.tok_embeddings.weight"]'),
        dropout=(;),
        transformer_blocks=NamedTuple(
            Symbol("layer_$(i+1)") => (;
                attention_norm=(; weight=params["_orig_mod.layers.$i.attention_norm.weight"]),
                attention=(;
                    wq=(; weight=params["_orig_mod.layers.$i.attention.wq.weight"]),
                    wk=(; weight=params["_orig_mod.layers.$i.attention.wk.weight"]),
                    wv=(; weight=params["_orig_mod.layers.$i.attention.wv.weight"]),
                    wo=(; weight=params["_orig_mod.layers.$i.attention.wo.weight"]),
                    attn_dropout=(;),
                    resid_dropout=(;)
                ),
                ffn_norm=(; weight=params["_orig_mod.layers.$i.ffn_norm.weight"]),
                feed_forward=(;
                    w1=(; weight=params["_orig_mod.layers.$i.feed_forward.w1.weight"]),
                    w2=(; weight=params["_orig_mod.layers.$i.feed_forward.w2.weight"]),
                    w3=(; weight=params["_orig_mod.layers.$i.feed_forward.w3.weight"]),
                    dropout=(;)
                )
            )
            for i in 0:checkpoint["model_args"]["n_layers"]-1
        ),
        norm=(; weight=params["_orig_mod.norm.weight"]),
        output=(; weight=params["_orig_mod.output.weight"]),
    )
    model, ps, LuxCore.initialstates(rng, model)
end

function generate(m, ps, st; s="", max_new_tokens=200)
    tokens = reshape(encode(s), :, 1)
    st = Lux.testmode(st)
    for _ in 1:max_new_tokens
        logits, _ = m(tokens, ps, st)
        new_token = argmax(logits[:, end, 1])
        tokens = vcat(tokens, new_token)
    end
    pyconvert(String, decode(tokens))
end

# "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"\nLily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"\nLily didn't want to help her mom, so she"