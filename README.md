# Master Deep Generative Models in Julia

The goal of this repo is to provide a list of deep generative models implemented in Julia.

Note that each implementation targets only a very specific case[^1]. Generally we'll
provide a simple version for the education purpose and a performant version
which demonstrates how to reach the state-of-the-art performance in Julia[^2].

## Model List

- MLP
- VAE
- VQ-VAE
- GAN
- GPT2
- LLAMA
- DDPM
- MoE
- [ ] VQGAN
- [ ] CLIP
- [ ] MaskGIT

## Benchmark

| Model | Environment | Performance | Description |
| ----- | ----------- | ----------- | ----------- |
| MLP   |             |             |             |

## Notable Implementations in Julia

- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl)
- [AutoEncoderToolkit.jl](https://github.com/mrazomej/AutoEncoderToolkit.jl)
- [DenoisingDiffusion.jl](https://github.com/LiorSinai/DenoisingDiffusion.jl)
- [Jjama3.jl](https://github.com/MurrellGroup/Jjama3.jl)

Feel free to add your work here.

## Q&A

1. Lux.jl vs Flux.jl?

   -  There have been some discussions on this topic (see [Deep learning in
      Julia](https://discourse.julialang.org/t/deep-learning-in-julia/112844)).
      Since each algorithm is implemented independently under separate folder,
      just choose the one you are most comfortable with.

## References

Following are some important blogs, papers, and codes that helped me a lot to understand the deep generative models implemented here.

- [6.S978 Deep Generative Models MIT EECS, Fall 2024](https://mit-6s978.github.io/schedule.html)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [A Beginner's Guide to Variational Methods: Mean-Field Approximation](https://blog.evjang.com/2016/08/variational-bayes.html)
- [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [A must-have training trick for VAE(variational autoencoder)](https://medium.com/@chengjing/a-must-have-training-trick-for-vae-variational-autoencoder-d28ff53b0023)
- [http://www.openias.org/variational-coin-toss](http://www.openias.org/variational-coin-toss)
- [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937v2)
- [Understanding Vector Quantization in VQ-VAE](https://huggingface.co/blog/ariG23498/understand-vq)
- [What is Residual Vector Quantization?](https://www.assemblyai.com/blog/what-is-residual-vector-quantization/)
- [Vector Quantization Pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [jax-vqvae-vqgan](https://github.com/kvfrans/jax-vqvae-vqgan)
- [pytorch-vqgan](https://github.com/Shubhamai/pytorch-vqgan)
- [The Illustarated VQGAN](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/)
  - ![VQGAN](https://ljvmiranda921.github.io/assets/png/vqgan/tree_of_knowledge.png)
- [CLIP](https://openai.com/index/clip/)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [Diffusion Models — DDPMs, DDIMs, and Classifier Free Guidance](https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion#resnet-block)

[^1]: This is to follow the [Tips & tricks](https://karpathy.github.io/2019/04/25/recipe/)
[^2]: See the discussions [here](https://discourse.julialang.org/t/community-interest-check-llms-from-scratch-in-pure-julia/121796/36?u=findmyway). Ideally we'd like to keep updating this repo and demonstrate how to reach the state-of-the-art performance in Julia.