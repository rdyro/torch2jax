#!/usr/bin/env zsh

set -x

#local_cache_dir="$(realpath $(pwd))/.cache/uv"
local_cache_dir="$HOME/.cache/uv"

python_ver=("3.9" "3.10" "3.11" "3.12")
jax_ver=("0.4.26" "0.4.27" "0.4.29" "0.4.31" "0.4.33")

# cpu tests ####################################################################
image_name="t2j:cpu"
docker build -t $image_name --build-arg=IMAGE_NAME="python:3.9" .

# run actual tests
for python_ver in ${python_ver[@]}; do
  for jax_ver in ${jax_ver[@]}; do
    docker run --rm \
      -v "$local_cache_dir:/root/.cache/uv" \
      "$image_name" ./scripts/_install_and_test.sh $python_ver $jax_ver "cpu"
  done
done

# cuda tests ###################################################################
if [[ -z "$(command -v nvidia-smi)" ]]; then
  echo "No nvidia-smi found. Skipping CUDA tests."
  exit 0
fi

jax_ver=("0.4.30" "0.4.31" "0.4.33")

image_name="t2j:cuda"
docker build -t $image_name --build-arg=IMAGE_NAME="ghcr.io/nvidia/jax:base" .

# run actual tests
for python_ver in ${python_ver[@]}; do
  for jax_ver in ${jax_ver[@]}; do
    docker run --rm \
      -v "$local_cache_dir:/root/.cache/uv" \
      --gpus all --shm-size=1g \
      "$image_name" ./scripts/_install_and_test.sh $python_ver $jax_ver "cuda"
  done
done
