#!/bin/bash
# Build a Docker image with PyTorch 2.12 from source on top of the therock-main base.
# Usage: ./build_pytorch212.sh [PYTORCH_TAG]
#
# The build reuses the existing PyTorch git clone in the base image and
# fetches only the delta for the target tag, saving clone time.

set -e

PYTORCH_BRANCH="${1:-release/2.12}"
IMAGE_TAG="therock-main:gfx94X_pytorch2.12_rocm7.13"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building PyTorch (ROCm/pytorch ${PYTORCH_BRANCH}) image..."
echo "  Output tag: ${IMAGE_TAG}"
echo ""

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.pytorch212" \
    --build-arg PYTORCH_BRANCH="${PYTORCH_BRANCH}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "Done. Image tagged as: ${IMAGE_TAG}"
echo ""
echo "Verify with:"
echo "  docker run --rm ${IMAGE_TAG} python3 -c \"import torch; print(torch.__version__)\""
