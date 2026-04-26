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
# Target virtual disk size. The pristine Debian cloud image is only ~3 GB,
# which fills up after installing the toolchain + a single gcc invocation.
# We grow the qcow2 (cheap - it's sparse) and let cloud-init's growpart +
# resize2fs modules extend the partition / filesystem on first boot.
QEMU_DISK_SIZE="${QEMU_DISK_SIZE:-16G}"

mkdir -p "$IMAGE_DIR" "$CLOUD_INIT_DIR"

# RESET=1 ./run_linux_in_qemu.sh wipes the boot disk so the next run starts from a
# freshly downloaded pristine image (handy when the qcow2 has accumulated bad
# state - e.g. interrupted dpkg, stale enabled units - from prior runs).
if [ "${RESET:-0}" = "1" ] && [ -f "$IMAGE_PATH" ]; then
  echo "RESET=1 -> deleting existing $IMAGE_PATH so it is re-downloaded fresh."
  rm -f "$IMAGE_PATH"
fi

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

# Grow the qcow2 to QEMU_DISK_SIZE if it is currently smaller. qcow2 is
# sparse so this only reserves virtual space - actual disk usage on the
# host stays roughly the same. cloud-init growpart + resize2fs (enabled
# by default in the Debian cloud image) extend the partition + ext4 on
# the next boot.
if command -v qemu-img >/dev/null 2>&1; then
  CURRENT_BYTES="$(qemu-img info --output=json "$IMAGE_PATH" 2>/dev/null \
                   | sed -n 's/.*"virtual-size":[[:space:]]*\([0-9]*\).*/\1/p' \
                   | head -n1)"
  TARGET_BYTES="$(numfmt --from=iec "$QEMU_DISK_SIZE" 2>/dev/null || echo 0)"
  if [ -n "$CURRENT_BYTES" ] && [ "$TARGET_BYTES" -gt 0 ] \
     && [ "$CURRENT_BYTES" -lt "$TARGET_BYTES" ]; then
    echo "Resizing $IMAGE_NAME from $(numfmt --to=iec --suffix=B "$CURRENT_BYTES") to $QEMU_DISK_SIZE..."
    qemu-img resize "$IMAGE_PATH" "$QEMU_DISK_SIZE"
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

# We deliberately do NOT use cloud-init's \`packages:\` directive: if a
# previous boot left dpkg half-configured (interrupted apt, ungraceful
# shutdown, etc.) cloud-init's package phase aborts cloud-final and the
# tools never get installed. Doing it from runcmd lets us run
# \`dpkg --configure -a\` first to recover, and survive partial state.
runcmd:
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive dpkg --configure -a 2>&1 || true" ]
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive apt-get update -y" ]
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux gdb gcc g++ make cmake git" ]
  - [ bash, -c, "apt-get clean && rm -rf /var/lib/apt/lists/*" ]

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