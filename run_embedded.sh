#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$SCRIPT_DIR/os_images"
IMAGE_NAME="debian-12-generic-amd64.qcow2"
IMAGE_PATH="$IMAGE_DIR/$IMAGE_NAME"
IMAGE_URL="https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2"
CLOUD_INIT_DIR="$IMAGE_DIR/cloud-init"
USER_DATA="$CLOUD_INIT_DIR/user-data"
META_DATA="$CLOUD_INIT_DIR/meta-data"
DEBIAN_PASSWORD="${DEBIAN_PASSWORD:-debian}"

mkdir -p "$IMAGE_DIR" "$CLOUD_INIT_DIR"

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

cat > "$USER_DATA" <<EOF
#cloud-config
ssh_pwauth: true
users:
  - default
chpasswd:
  expire: false
  list: |
    debian:${DEBIAN_PASSWORD}
preserve_hostname: false
hostname: debian-vm
fqdn: debian-vm.local

write_files:
  - path: /lib/systemd/system/test.service
    owner: root:root
    permissions: '0644'
    content: |
      [Unit]
      Description=Test Service

      [Service]
      Type=oneshot
      ExecStart=/bin/bash -c "sleep 1 && exit 1"
      RemainAfterExit=yes

      [Install]
      WantedBy=multi-user.target

runcmd:
  - systemctl daemon-reload
  - systemctl enable test.service
  - systemctl restart test.service || true

EOF

if command -v uuidgen >/dev/null 2>&1; then
  INSTANCE_ID="iid-$(uuidgen)"
else
  INSTANCE_ID="iid-$(date +%s)"
fi

cat > "$META_DATA" <<EOF
instance-id: $INSTANCE_ID
local-hostname: debian-vm
EOF

declare -a CLOUD_INIT_ARGS=()

if command -v cloud-localds >/dev/null 2>&1; then
  SEED_IMG="$CLOUD_INIT_DIR/debian-seed.img"
  cloud-localds "$SEED_IMG" "$USER_DATA" "$META_DATA"
  CLOUD_INIT_ARGS=(-drive file="$SEED_IMG",if=virtio,format=raw)
elif command -v genisoimage >/dev/null 2>&1; then
  SEED_IMG="$CLOUD_INIT_DIR/debian-seed.iso"
  genisoimage -output "$SEED_IMG" -volid cidata -joliet -rock "$USER_DATA" "$META_DATA"
  CLOUD_INIT_ARGS=(-drive file="$SEED_IMG",if=virtio,media=cdrom)
elif command -v mkisofs >/dev/null 2>&1; then
  SEED_IMG="$CLOUD_INIT_DIR/debian-seed.iso"
  mkisofs -output "$SEED_IMG" -volid cidata -joliet -rock "$USER_DATA" "$META_DATA"
  CLOUD_INIT_ARGS=(-drive file="$SEED_IMG",if=virtio,media=cdrom)
else
  echo "No cloud-init seed generator found (cloud-localds/genisoimage/mkisofs)." >&2
  exit 1
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
  -nographic \
  "${CLOUD_INIT_ARGS[@]}"