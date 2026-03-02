export CUDA_VISIBLE_DEVICES=MIG-e9c20b15-3db0-52c6-bc66-bb3905aeaa7b
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$CUDA_VISIBLE_DEVICES
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$CUDA_VISIBLE_DEVICES

nvidia-cuda-mps-control -d

