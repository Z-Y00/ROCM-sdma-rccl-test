#!/bin/bash
# Build a docker image with PyTorch 2.12 from source on top of the
# TheRock Ubuntu 24.04 / ROCm 7.14 / py3.14 base.
#
# Usage:
#   ./build_pytorch212.sh                  # release/2.12 branch tip
#   ./build_pytorch212.sh release/2.12     # explicit branch
#   ./build_pytorch212.sh v2.12.0          # specific tag
#
# Reuses the existing /var/lib/jenkins/pytorch clone in the base image and
# fetches only the delta for the target ref, so the build doesn't have to
# re-download ~5 GB of git history.
set -e

PYTORCH_BRANCH="${1:-release/2.12}"
BASE_IMAGE="${BASE_IMAGE:-registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_ubuntu24.04_py3.14_pytorch_release-2.11_96bfee1}"
IMAGE_TAG="${IMAGE_TAG:-therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PyTorch source ref : ${PYTORCH_BRANCH}"
echo "=== Base image         : ${BASE_IMAGE}"
echo "=== Output tag         : ${IMAGE_TAG}"
echo ""

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.pytorch212" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PYTORCH_BRANCH="${PYTORCH_BRANCH}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "Done. Image tagged as: ${IMAGE_TAG}"
echo ""
echo "Verify (needs GPU):"
echo "  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\"
echo "      ${IMAGE_TAG} python3 -c 'import torch; print(torch.__version__)'"
echo ""
echo "Run the overlap benches against this image:"
echo "  ROCM_BUG_TEST_IMAGE=${IMAGE_TAG} bench/run_bench.sh both"
echo "  ROCM_BUG_TEST_IMAGE=${IMAGE_TAG} bench/run_trace.sh"
