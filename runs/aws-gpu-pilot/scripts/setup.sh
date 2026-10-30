#!/bin/bash
# GPU instance setup (README "AWS GPU"), logged to ~/setup.log.
set -euo pipefail
step() { echo "=== [$(date "+%T")] $*"; }
cd ~/memory-aware-agent-edge-serving
step uv
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
step venv
[ -x .venv/bin/python ] || uv venv --python 3.13 .venv
[ -d third_party/tau2-bench ] || git clone -q --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench
uv pip install --python .venv/bin/python -r requirements-lock.txt 2>&1 | tail -2
step model
mkdir -p models
[ -s models/Qwen3-4B-Q4_K_M.gguf ] || curl -fsSL --retry 5 -C - -o models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf
(cd models && sha256sum -c ../runs/final/model.sha256)
step llama.cpp
[ -d third_party/llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout -q 620d2225c
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF > ~/cmake.log 2>&1
cmake --build third_party/llama.cpp/build -j 4 --target llama-server llama-bench >> ~/cmake.log 2>&1
third_party/llama.cpp/build/bin/llama-server --version 2>&1 | tail -2
step tests
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
step "SETUP OK"
