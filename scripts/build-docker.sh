#!/usr/bin/env bash
set -e
set -o pipefail

echo "Starting GPU Docker build..."

# Optional defaults
IMAGE_NAME=${IMAGE_NAME:-gadgetron_lit}
ACR_LOGIN_SERVER=${ACR_LOGIN_SERVER:-gtdocker-d0a2dpbsdpdzh6cz.azurecr.io}

echo "Building RT image..."
docker build --target gadgetron_rt_cuda -t $ACR_LOGIN_SERVER/${IMAGE_NAME}_rt:latest .

echo "Building DEV image..."
docker build --target gadgetron_cudabuild -t $ACR_LOGIN_SERVER/${IMAGE_NAME}_dev:latest .

echo "Docker build completed."
