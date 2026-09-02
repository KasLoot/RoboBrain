workspace=/workspace
mkdir -p $workspace/vllm
cd $workspace/vllm
uv venv -p 3.12
source .venv/bin/activate

# CUDA 13.0

uv pip install vllm==0.27.0 --extra-index-url https://wheels.vllm.ai/0.25.1/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

uv pip install transformers==5.14

mkdir -p $workspace/models/gemma-4-26B-A4B-it

hf download google/gemma-4-26B-A4B-it --local-dir $workspace/models/gemma-4-26B-A4B-it

VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /workspace/models/gemma-4-26B-A4B-it \
 --max-model-len 32768 \
 --enable-per-request-metrics \
 --gpu-memory-utilization 0.95 \
 --enable-auto-tool-choice \
 --tool-call-parser gemma4 \
 --reasoning-parser gemma4 \
 --mm-processor-kwargs '{"max_soft_tokens": 560}' \
 --moe-backend triton \
 --trust-remote-code \
 --port 9000

- Supported values: 70, 140, 280 (default), 560, 1120 tokens per image.

ssh -N -L 9000:127.0.0.1:9000 \
 -p 14315 -i ~/.ssh/id_ed25519 \
 root@157.157.221.177
