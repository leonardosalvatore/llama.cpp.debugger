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

# GUI=1 boots the VM into a real graphical session (Xorg or Wayland) inside
# a QEMU window, instead of the default headless serial console.
#   DESKTOP=xfce  (default) -> Xfce on Xorg via LightDM. Light, fast in QEMU.
#   DESKTOP=gnome           -> GNOME, Wayland session by default on Debian 12.
# The desktop packages are installed by cloud-init on the FIRST boot, which
# pulls a few hundred MB and takes several minutes. Subsequent boots are fast.
#
# GUI_ACCEL=1 switches the GPU from the conservative "virtio-vga + gtk"
# combo (no GL, works everywhere) to "virtio-vga-gl + gtk,gl=on", which
# uses host OpenGL via virglrenderer. Faster, but on some QEMU/host
# combinations triggers a 'Blocked re-entrant IO on vga-lowmem' warning
# that ends in a black GUI window - in that case leave GUI_ACCEL unset.
GUI="${GUI:-0}"
GUI_ACCEL="${GUI_ACCEL:-0}"
DESKTOP="${DESKTOP:-xfce}"
QEMU_RAM_MB="${QEMU_RAM_MB:-$([ "$GUI" = "1" ] && echo 4096 || echo 2048)}"
QEMU_SMP="${QEMU_SMP:-4}"

# Initial QEMU display resolution (only meaningful with GUI=1). Format is
# WIDTHxHEIGHT, parsed into the virtio-vga 'xres=' / 'yres=' properties.
# Once spice-vdagent comes up inside the guest, the desktop will auto-resize
# to match the QEMU window, so this mainly governs GRUB + early kernel + the
# login screen geometry on first boot.
QEMU_RES="${QEMU_RES:-1024x768}"
if ! [[ "$QEMU_RES" =~ ^([0-9]+)x([0-9]+)$ ]]; then
  echo "QEMU_RES must be WIDTHxHEIGHT (e.g. 1024x768), got '$QEMU_RES'" >&2
  exit 1
fi
QEMU_XRES="${BASH_REMATCH[1]}"
QEMU_YRES="${BASH_REMATCH[2]}"

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

# Build the desktop-related runcmd lines (if any). With recommends ON, since
# a desktop without recommends is missing fonts, drivers, network manager, etc.
# Lean package set on purpose - skip firefox/xfce4-goodies/etc; the user can
# `apt install` extras after first boot if they want them.
DESKTOP_RUNCMD=""
DESKTOP_GETTY_RUNCMD=""
if [ "$GUI" = "1" ]; then
  case "$DESKTOP" in
    xfce)
      DESKTOP_PKGS="xfce4 lightdm xserver-xorg dbus-x11 network-manager \
fonts-dejavu spice-vdagent qemu-guest-agent"
      ;;
    gnome)
      # gnome-core gives a Wayland session by default on Debian 12 (gdm3).
      DESKTOP_PKGS="gnome-core gdm3 dbus-x11 network-manager \
fonts-dejavu spice-vdagent qemu-guest-agent"
      ;;
    *)
      echo "Unknown DESKTOP='$DESKTOP' (expected: xfce | gnome)" >&2
      exit 1
      ;;
  esac
  # Enable a text getty on tty1 IMMEDIATELY so the QEMU GUI window has a
  # login prompt within seconds, instead of staring at a black screen for
  # 10+ minutes while the desktop install grinds. Also lets the user log
  # in and \`journalctl -fu cloud-final\` to watch progress.
  DESKTOP_GETTY_RUNCMD='  - [ bash, -c, "systemctl enable --now getty@tty1.service" ]'
  DESKTOP_RUNCMD="$(cat <<RUNCMD
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive apt-get install -y $DESKTOP_PKGS" ]
  - [ bash, -c, "systemctl set-default graphical.target" ]
  - [ bash, -c, "systemctl isolate graphical.target || true" ]
RUNCMD
)"
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
${DESKTOP_GETTY_RUNCMD}
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive dpkg --configure -a 2>&1 || true" ]
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive apt-get update -y" ]
  - [ bash, -c, "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux gdb gcc g++ make cmake git linux-perf" ]
  # Relax perf paranoia so the unprivileged 'debian' user can record
  # userspace + kernel samples without sudo (the perf_* MCP tools still
  # accept a sudo flag for anything that needs it). Persisted via a
  # sysctl drop-in so it survives reboots.
  - [ bash, -c, "printf 'kernel.perf_event_paranoid=-1\\nkernel.kptr_restrict=0\\n' > /etc/sysctl.d/99-perf.conf" ]
  - [ bash, -c, "sysctl --system 2>&1 || true" ]
${DESKTOP_RUNCMD}
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

# KVM gives the desktop usable performance. Without it the GUI is a slideshow.
declare -a ACCEL_ARGS=()
if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
  ACCEL_ARGS=(-enable-kvm -cpu host)
fi

declare -a DISPLAY_ARGS=()
if [ "$GUI" = "1" ]; then
  if [ "$GUI_ACCEL" = "1" ]; then
    # virgl-accelerated path. Faster, but on some QEMU versions emits
    # 'Blocked re-entrant IO on vga-lowmem' and ends in a black window;
    # if that happens, drop GUI_ACCEL.
    DISPLAY_ARGS=(
      -device "virtio-vga-gl,xres=$QEMU_XRES,yres=$QEMU_YRES"
      -display gtk,gl=on
    )
  else
    # Conservative path: plain virtio-vga (no GL). Works on every host,
    # no virglrenderer needed, no re-entrant warnings. Software-rendered
    # but with KVM still perfectly usable for a debugger desktop.
    DISPLAY_ARGS=(
      -device "virtio-vga,xres=$QEMU_XRES,yres=$QEMU_YRES"
      -display gtk
    )
  fi
  DISPLAY_ARGS+=(
    -device qemu-xhci
    -device usb-tablet
    -device usb-kbd
    -device virtio-serial-pci
    -chardev spicevmc,id=spicechannel0,name=vdagent
    -device virtserialport,chardev=spicechannel0,name=com.redhat.spice.0
  )
  cat <<MSG
Booting with GUI (DESKTOP=$DESKTOP, GUI_ACCEL=$GUI_ACCEL, RES=${QEMU_XRES}x${QEMU_YRES}).
- A text login on tty1 (debian / $DEBIAN_PASSWORD) appears within seconds.
- The desktop is installed by cloud-init in the background; the graphical
  login screen will replace tty1 in ~5-15 minutes on first boot (next
  boots are instant). Watch progress from inside the VM with:
      journalctl -fu cloud-final
- If the QEMU window stays blank after the kernel handoff, your host most
  likely tripped the virtio-vga-gl re-entrancy bug; rerun without
  GUI_ACCEL=1 (the default).
- spice-vdagent will auto-resize the desktop to the QEMU window once the
  desktop is up; the QEMU_RES setting governs the boot-time / login-screen
  resolution.
MSG
else
  DISPLAY_ARGS=(-nographic)
fi

exec qemu-system-x86_64 \
  "${ACCEL_ARGS[@]}" \
  -m "$QEMU_RAM_MB" \
  -smp "$QEMU_SMP" \
  -drive file="$IMAGE_PATH",if=virtio \
  -nic user,hostfwd=tcp::2222-:22 \
  "${DISPLAY_ARGS[@]}" \
  "${CLOUD_INIT_ARGS[@]}"