# adapted from https://github.com/jaymody/picoGPT
import HuggingFaceHub as HF
using SafeTensors: load_safetensors
using BytePairEncoding: load_gpt2
using JSON3
using LinearAlgebra: triu
using NNlib: batched_mul, softmax
using Lux: gelu

using Statistics: mean, std

DATA_DIR = "data"
MODEL_ID = "openai-community/gpt2"
HF_GPT2 = HF.info(HF.Model(id=MODEL_ID))
PARAMS = load_safetensors(HF.file_download(HF_GPT2, "model.safetensors"))
TOKENIZER = load_gpt2()
VOCAB = JSON3.read(HF.file_download(HF_GPT2, "vocab.json"))
CONFIG = JSON3.read(HF.file_download(HF_GPT2, "config.json"))


encode(s) = [VOCAB[t] + 1 for t in TOKENIZER(s)] # !!! +1 for 1-indexing

function gpt2(tokens)
    xs = view(PARAMS["wte.weight"]', :, tokens) .+ view(PARAMS["wpe.weight"]', :, 1:length(tokens))

    for i in 1:CONFIG["n_layer"]
        xs = apply_transformer_layer(xs, i - 1)
    end

    xs = layer_norm(xs, PARAMS["ln_f.weight"], PARAMS["ln_f.bias"])
    PARAMS["wte.weight"] * xs
end

function apply_transformer_layer(xs, i)
    xs_norm = layer_norm(xs, PARAMS["h.$i.ln_1.weight"], PARAMS["h.$i.ln_1.bias"])
    xs = xs + mha(
        xs_norm,
        PARAMS["h.$i.attn.c_attn.weight"]',
        PARAMS["h.$i.attn.c_attn.bias"]',
        PARAMS["h.$i.attn.c_proj.weight"]',
        PARAMS["h.$i.attn.c_proj.bias"]',
        CONFIG["n_head"]
    )
    xs_norm = layer_norm(xs, PARAMS["h.$i.ln_2.weight"], PARAMS["h.$i.ln_2.bias"])
    xs + ffn(xs_norm, PARAMS["h.$i.mlp.c_fc.weight"]', PARAMS["h.$i.mlp.c_fc.bias"]', PARAMS["h.$i.mlp.c_proj.weight"]', PARAMS["h.$i.mlp.c_proj.bias"]')
end

function layer_norm(xs, w, b)
    m = mean(xs, dims=1)
    s = std(xs, dims=1)
    xs = (xs .- m) ./ s
    w .* xs .+ b
end

function mha(xs, c_attn_w, c_attn_b, c_proj_w, c_proj_b, n_heads)
    h_dim, seq_len = size(xs)
    # Q,K,V projection
    qkv = c_attn_w * xs .+ vec(c_attn_b)
    q, k, v = (reshape(selectdim(qkv, 1, (h_dim*(i-1)+1):h_dim*i), :, n_heads, seq_len) for i in 1:3)
    # move head dimension to the last
    q = permutedims(q, (1, 3, 2))
    kᵀ = permutedims(k, (3, 1, 2))
    v = permutedims(v, (1, 3, 2))

    mask = triu(ones(Bool, seq_len, seq_len))
    logits = ifelse.(mask, batched_mul(kᵀ, q ./ sqrt(size(q, 1))), typemin(eltype(xs)))
    scores = softmax(logits)
    out = batched_mul(v, scores)
    # recover original shape
    out = reshape(permutedims(out, (1, 3, 2)), :, seq_len)

    c_proj_w * out .+ vec(c_proj_b)
end

function ffn(xs, c_fc_w, c_fc_b, c_proj_w, c_proj_b)
    xs = gelu.(c_fc_w * xs .+ vec(c_fc_b))
    c_proj_w * xs .+ vec(c_proj_b)
end

function generate(prompt, n_tokens=40)
    tokens = encode(prompt)
    for _ in 1:n_tokens
        next_token = argmax(gpt2(tokens)[:, end])
        push!(tokens, next_token)
    end
    tokens
end