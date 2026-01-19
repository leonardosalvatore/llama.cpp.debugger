#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$SCRIPT_DIR/os_images"
IMAGE_NAME="debian-12-generic-amd64.qcow2"
IMAGE_PATH="$IMAGE_DIR/$IMAGE_NAME"
IMAGE_URL="https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2"

mkdir -p "$IMAGE_DIR"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Image $IMAGE_NAME not found; downloading..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --show-error "$IMAGE_URL" -o "$IMAGE_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$IMAGE_PATH" "$IMAGE_URL"
  else
    echo "Neither curl nor wget is available; cannot download image." >&2
    exit 1
  fi
fi

if ! command -v qemu-system-x86_64 >/dev/null 2>&1; then
  echo "qemu-system-x86_64 not found in PATH." >&2
  exit 1
fi

exec qemu-system-x86_64 \
  -m 2048 \
  -smp 2 \
  -drive file="$IMAGE_PATH",if=virtio \
  -nic user,hostfwd=tcp::2222-:22 \
  -nographic