using SafeTensors
using Pickle

#####
# VAE
#####

function from_pretrained(MODEL=joinpath(@__DIR__, "..", "models", "Wan-AI", "Wan2.1-T2V-1.3B"))
    ps_vae = Pickle.Torch.THload(joinpath(MODEL, "Wan2.1_VAE.pth"))
    ps_t5 = Pickle.Torch.THload(joinpath(MODEL, "models_t5_umt5-xxl-enc-fp32.pth"))
    ps_diff = load_safetensors(joinpath(MODEL, "diffusion_pytorch_model.safetensors"))
    (ps_vae, ps_t5, ps_diff)
end