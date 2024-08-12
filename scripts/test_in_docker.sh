#!/usr/bin/env zsh

set -xe

root_dir="$(realpath $(pwd))"

python_ver=("3.9" "3.10" "3.11" "3.12")
jax_ver=("0.4.26" "0.4.27" "0.4.28" "0.4.29" "0.4.30" "0.4.31")


# cpu tests ####################################################################
#image_name="t2j:cpu"
#if [[ -z "$(docker images -q $image_name 2> /dev/null)" ]]; then
#  docker build -t $image_name \
#    --build-arg=IMAGE_NAME="python:3.9" .
#fi
#
## run actual tests
#for python_ver in ${python_ver[@]}; do
#  for jax_ver in ${jax_ver[@]}; do
#    docker run --rm \
#      -v "$root_dir/.cache/uv:/root/.cache/uv" \
#      "$image_name" ./scripts/_install_and_test.sh $python_ver $jax_ver "cpu"
#  done
#done

# cuda tests ###################################################################
if [[ -z "$(command -v nvidia-smi)" ]]; then
  echo "No nvidia-smi found. Skipping CUDA tests."
  exit 0
fi

python_ver=("3.9" "3.10" "3.11" "3.12")
jax_ver=("0.4.30" "0.4.31")

image_name="t2j:cuda"
if [[ -z "$(docker images -q $image_name 2> /dev/null)" ]]; then
  docker build -t $image_name \
    --build-arg=IMAGE_NAME="ghcr.io/nvidia/jax:base" .
fi

# run actual tests
for python_ver in ${python_ver[@]}; do
  for jax_ver in ${jax_ver[@]}; do
    docker run --rm \
      -v "$root_dir/.cache/uv:/root/.cache/uv" \
      --gpus all --shm-size=1g \
      "$image_name" ./scripts/_install_and_test.sh $python_ver $jax_ver "cuda"
  done
done