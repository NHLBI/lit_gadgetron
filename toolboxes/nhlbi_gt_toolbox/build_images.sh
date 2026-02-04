#!/bin/bash

# -----------------------------------------------------------------------------
# build_images.sh
#
# This script builds and pushes Docker images for Gadgetron using a specified
# base image name and tag. It supports an optional '--no-cache' argument to
# force Docker to build images without using cache.
#
# Usage:
#   ./build_images.sh [TAG] [BASE_IMAGE_NAME] [--no-cache]
#
# Arguments:
#   TAG               (required) The tag to append to the image names.
#   BASE_IMAGE_NAME   (optional) The base Docker image name to use.
#   --no-cache        (optional) Build Docker images without cache.
#
# Behavior:
#   - If no BASE_IMAGE_NAME is provided, a default image name is used.
#   - Builds two images: one for development (dev) and one for runtime (rt).
#   - Pushes both images to the Docker registry.
#
# Example:
#   # Build and push images using default base image and tag 'v1.0'
#   ./build_images.sh v1.0
#
#   # Build and push images using a custom base image and tag 'v2.1'
#   ./build_images.sh v2.1 myregistry/myimage
#
#   # Build and push images without using cache
#   ./build_images.sh v2.1 myregistry/myimage --no-cache
# -----------------------------------------------------------------------------

# Check for --no-cache argument
NO_CACHE=""
for arg in "$@"; do
    if [ "$arg" == "--no-cache" ]; then
        NO_CACHE="--no-cache"
        set -- "${@/--no-cache/}"
        break
    fi
done

TAG="$1"
BASE_IMAGE_NAME="$2"

if [ -z "$TAG" ]; then
    echo "Error: TAG argument is required."
    echo "Usage: ./build_images.sh [TAG] [BASE_IMAGE_NAME] [--no-cache]"
    exit 1
fi

if [ -z "$BASE_IMAGE_NAME" ]; then
    BASE_IMAGE_NAME="gtdocker-d0a2dpbsdpdzh6cz.azurecr.io/ubuntu_2204_cuda124_lit"
fi

DEV_NAME="${BASE_IMAGE_NAME}:${TAG}_dev"
RT_NAME="${BASE_IMAGE_NAME}:${TAG}_rt"

echo "image prefix: $DEV_NAME"
echo "image prefix: $RT_NAME"

docker buildx build $NO_CACHE --build-arg BUILDKIT_INLINE_CACHE=1 --network=host --target gadgetron_cudabuild -t "${DEV_NAME}" -f ../../Dockerfile ../../ --push
docker buildx build $NO_CACHE --build-arg BUILDKIT_INLINE_CACHE=1 --network=host --target gadgetron_rt_cuda -t "${RT_NAME}" -f ../../Dockerfile ../../ --push

# docker push "${DEV_NAME}"
# docker push "${RT_NAME}"
