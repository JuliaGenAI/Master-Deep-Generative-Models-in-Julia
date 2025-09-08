# Master Deep Generative Models in Julia

The goal of this repo is to provide a list of deep generative models implemented in Julia.

- ✅ **Clear** and **concise** implementation to help you understand model architecture and algorithm details.
- 🤔 Runtime efficiency and memory usage are not optimized.
- ❌ Continuous training or finetuning are not supported.

## User guide
Step 1. Initialize projects

```bash
make init  # `make update` to update packages
```

Step 2. Download models

```bash
model=openai-community/gpt2 make download  # GPT2 model
HF_TOKEN=... model=meta-llama/Llama-3.2-1B-Instruct make download  # Llama model (requires HuggingFace token and accepting the license agreement: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
model=deepseek-ai/DeepSeek-V2-Lite-Chat make download  # DeepSeek model
model=google/gemma-3-4b-it make download  # Gemma model
```

Step 3. Run examples

```bash
case=01_gpt2 make run  # GPT2 example
case=02_Llama make run  # Llama example
case=03_deepseek make run  # DeepSeek example
case=04_gemma make run  # Gemma example
```