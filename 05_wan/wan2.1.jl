#=
Tips:

1. The dims of video input in the python world is (B, C, T, H, W), while in the julia world is (W, H, T, C, B).

=#
using LinearAlgebra
using SafeTensors
import Pickle
using NNlib: conv, DenseConvDims, swish
import FileIO
using ImageCore: channelview
using Statistics: mean


#####
# VAE
#####
@kwdef struct RMSNorm
    weight
    bias = nothing
    channel_dim::Int = 1
    eps::Float32 = 1e-12
end

function (m::RMSNorm)(x)
    scale = Float32(size(x, m.channel_dim)^0.5)
    # !!! actually it is `sum` not `mean` in the original implementation
    y = x ./ (sqrt.(sum(x .^ 2, dims=m.channel_dim) .+ m.eps)) .* scale .* m.weight
    isnothing(m.bias) ? y : y .+ m.bias
end

@kwdef struct Conv2d
    weight
    bias
    stride
    padding
end

#  size(x): (W, H, C, B)
function (m::Conv2d)(x)
    y = conv(x, m.weight, DenseConvDims(x, m.weight; stride=m.stride, padding=m.padding, flipkernel=true))
    y .+ reshape(m.bias, 1, 1, :, 1)
end

struct DownSample2d
    resample::Conv2d
end

function (m::DownSample2d)(x)
    y = permutedims(x, (1, 2, 4, 3, 5)) # (W, H, T, C, B) -> (W, H, C, T, B)
    y = reshape(y, size(y)[1:3]..., :)
    m.resample(y)
end

@kwdef struct CausalConv3d
    weight::AbstractArray
    bias::AbstractVector
    stride
    padding
end

#  size(x): (W, H, T, C, B)
function (c::CausalConv3d)(x)
    p = c.padding  # (W, H, T)
    casual_padding = p isa Int ? (p, p, p, p, 2p, 0) : (p[1], p[1], p[2], p[2], 2 * p[3], 0)
    y = conv(x, c.weight, DenseConvDims(x, c.weight; stride=c.stride, padding=casual_padding, flipkernel=true))
    y .+ reshape(c.bias, 1, 1, 1, :, 1)
end

@kwdef struct ResidualBlock
    up_norm::RMSNorm
    up_proj::CausalConv3d
    down_norm::RMSNorm
    down_proj::CausalConv3d
    shortcut = identity
end

function (m::ResidualBlock)(x)
    h = m.shortcut(x)
    o = m.down_proj(swish.(m.down_norm(m.up_proj(swish.(m.up_norm(x))))))
    h + o
end

struct AttentionBlock
    norm::RMSNorm
    qkv_proj::Conv2d
    o_proj::Conv2d
end

function (m::AttentionBlock)(x)
    W, H, C, D, B = size(x)
    h = reshape(x, W, H, C, D * B)
    h = m.norm(h)
    h = m.qkv_proj(h)
    h = reshape(h, W * H, :, 3, D * B)
    q = @view(h[:, :, 1, :])
    k = @view(h[:, :, 2, :])
    v = @view(h[:, :, 3, :])

    q = permutedims(q, (2, 1, 3))
    kᵀ = k
    v = permutedims(v, (2, 1, 3))
end

struct Encoder
    blocks
end

struct Decoder
end

struct VAE
    encoder::Encoder
    decoder::Decoder
end

#####

function from_pretrained(MODEL=joinpath(@__DIR__, "..", "models", "Wan-AI", "Wan2.1-T2V-1.3B"))
    ps_vae = Pickle.Torch.THload(joinpath(MODEL, "Wan2.1_VAE.pth"))
    ps_t5 = Pickle.Torch.THload(joinpath(MODEL, "models_t5_umt5-xxl-enc-fp32.pth"))
    ps_diff = load_safetensors(joinpath(MODEL, "diffusion_pytorch_model.safetensors"))
    (ps_vae, ps_t5, ps_diff)
end

function main()
    img = channelview(FileIO.load("example.jpg"))  # (C,H,W)
    img = (img .- 0.5f0) ./ 0.5f0
    img = permutedims(img, (3, 2, 1)) # (W,H,C)
    img = reshape(img, size(img, 1), size(img, 2), 1, size(img, 3), 1) # (W,H,T,C,B)

    ps_vae, ps_t5, ps_diff = from_pretrained()

    # m = CausalConv3d(weight=permutedims(ps_vae["encoder.conv1.weight"], (5, 4, 3, 2, 1)), bias=ps_vae["encoder.conv1.bias"], stride=1, padding=1)
    # m = RMSNorm(weight=permutedims(ps_vae["encoder.downsamples.0.residual.0.gamma"], (4, 3, 2, 1)), channel_dim=4)
    # m = ResidualBlock(
    #     up_norm=RMSNorm(weight=permutedims(ps_vae["encoder.downsamples.0.residual.0.gamma"], (4, 3, 2, 1)), channel_dim=4),
    #     up_proj=CausalConv3d(weight=permutedims(ps_vae["encoder.downsamples.0.residual.2.weight"], (5, 4, 3, 2, 1)), bias=ps_vae["encoder.downsamples.0.residual.2.bias"], stride=1, padding=1),
    #     down_norm=RMSNorm(weight=permutedims(ps_vae["encoder.downsamples.0.residual.3.gamma"], (4, 3, 2, 1)), channel_dim=4),
    #     down_proj=CausalConv3d(weight=permutedims(ps_vae["encoder.downsamples.0.residual.6.weight"], (5, 4, 3, 2, 1)), bias=ps_vae["encoder.downsamples.0.residual.6.bias"], stride=1, padding=1),
    # )
end