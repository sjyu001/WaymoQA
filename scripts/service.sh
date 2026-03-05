export CUDA_VISIBLE_DEVICES=0,1
GPU_NUM=2

vllm serve "Qwen/Qwen2.5-VL-7B-Instruct" \
  --port 8001 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --allowed-local-media-path /
