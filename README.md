# llama.cpp.debugger

A safe playground for driving an embedded Linux SUT (System Under Test) from a
local LLM. The model runs on **llama.cpp** (`llama-server`) and acts on a Debian
VM brought up by QEMU through an SSH-driven tool surface.

It ships in two equivalent flavors:

- **CLI** (`systemd_mcp/cli.py`) - talks to the local `llama-server` via its
  OpenAI-compatible `/v1/chat/completions` endpoint and executes tools
  in-process.
- **MCP server** (`systemd_mcp/server.py`) - exposes the exact same tool
  surface over MCP for any MCP-compatible client.

Both share one tool registry; there is no duplicated logic.

## Components

**`start-llama-server.sh`** - launches `llama-server` (ROCm build) on
`http://0.0.0.0:53425` with `--jinja --reasoning-format deepseek` so the
thinking trace is split out of the visible content. Edit the `DEFAULT_MODEL`
line to pick a different GGUF.

**`run_linux_in_qemu.sh`** - boots a Debian 12 cloud image inside QEMU
(NAT'd, with SSH host-forwarded to `127.0.0.1:2222`). Cloud-init injects the
`debian` / `debian` credentials and pre-installs the toolchain the agent
needs (`tmux`, `gdb`, `gcc`, `g++`, `make`, `cmake`). Image source:
<https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2>.
To use the Debian shell, press Enter and log in as `debian` / `debian`.

By default the VM is headless (serial console attached to the launching
terminal via `-nographic`). Set `GUI=1` to instead boot into a real
graphical session in a QEMU window:

```bash
GUI=1 ./run_linux_in_qemu.sh                  # Xfce on Xorg via LightDM (default)
GUI=1 DESKTOP=gnome ./run_linux_in_qemu.sh    # GNOME, Wayland session via gdm3
GUI=1 GUI_ACCEL=1 ./run_linux_in_qemu.sh      # opt in to virgl GL acceleration
```

When `GUI=1`:

- Cloud-init **first enables a text getty on tty1**, then installs the chosen
  desktop in the background. So a few seconds after boot the QEMU window
  shows a `debian-vm login:` prompt - log in as `debian` / `debian` and run
  `journalctl -fu cloud-final` to watch the desktop install live (don't
  panic if the screen stays blank past the kernel boot, that's just tty1
  before getty has spawned).
- The desktop install pulls ~400-800 MB and takes 5-15 minutes on the first
  boot; subsequent boots go straight to the graphical login.
- RAM is bumped to 4096 MB (override with `QEMU_RAM_MB=...`) and KVM
  (`-enable-kvm -cpu host`) is auto-enabled when `/dev/kvm` is accessible.
  Without KVM the desktop is unusable.
- The boot resolution defaults to **1024x768**; override with e.g.
  `QEMU_RES=1920x1080`. This sets the virtio-vga `xres` / `yres` properties,
  which controls GRUB / early kernel / login-screen geometry. After login,
  `spice-vdagent` auto-resizes the desktop to match the QEMU window.
- The default GPU setup is `-device virtio-vga -display gtk` (software
  rendered, but rock-solid, no virgl needed). Set `GUI_ACCEL=1` to switch
  to `-device virtio-vga-gl -display gtk,gl=on` for virgl host-OpenGL
  passthrough - faster, but on some QEMU versions emits `Blocked re-entrant
  IO on vga-lowmem` and ends in a black QEMU window. If that happens to
  you, drop `GUI_ACCEL`.
- `spice-vdagent` + `usb-tablet` are wired in either way for absolute mouse
  / clipboard / resolution integration.

Switching desktops on an already-provisioned disk requires `RESET=1` (the
desktop install lives in cloud-init `runcmd`, which only runs on a fresh
instance), or just `apt install task-gnome-desktop` from inside the VM.

**`systemd_mcp/server.py`** - FastMCP server. All tools route through one
`_run_ssh_cmd` helper that reads its target from a single module-level dict;
the agent can repoint it at runtime via `configuration_setTargetHost`.

**`systemd_mcp/cli.py`** - OpenAI-protocol streaming client for `llama-server`,
re-using the same callables registered in `server.py`. Streams the
`reasoning_content` (thinking) channel separately from the visible answer.

## Tool namespaces

| Prefix              | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `configuration_*`   | `setTargetHost(host, port, username, password)`, `getTargetHost`, `setRemoteEnv(name, value)`, `unsetRemoteEnv(name)`, `getRemoteEnv` |
| `systemd_*`         | service status, journal, start/stop/restart/enable/disable, daemon-reload, list, uptime |
| `linux_*`           | list/read/write/append/remove files, mkdir, cp, mv, find, grep, ps, df, which |
| `compiler_*`        | `gcc`, `make`, `cmake_configure`, `cmake_build`                |
| `gdb_*`             | persistent tmux session: attach/run/core, send_command, break/continue/step/next/finish, print, backtrace, info_registers, info_threads, list_breakpoints, read_output, quit |

`gdb_*` keeps its state in a single tmux session (`llamadbg`) on the SUT, so
breakpoints, stepping, and inspecting locals all work across MCP calls.

## Demo

Open three terminals.

1. **Start the LLM**

   ```bash
   ./start-llama-server.sh
   ```

2. **Boot the SUT** (only if it isn't already up; check `ss -ltn | grep 2222`)

   ```bash
   ./run_linux_in_qemu.sh
   ```

   First boot installs `tmux gdb gcc g++ make cmake` via cloud-init `runcmd`,
   which takes a minute. The launcher also grows the qcow2 to 16 GB
   (overridable via `QEMU_DISK_SIZE=...`) so cloud-init's `growpart` can
   extend `/` enough for the toolchain plus debug builds. If a prior run
   left the qcow2 in a bad state (interrupted dpkg, stale enabled units,
   `/` already full) wipe it and re-download:

   ```bash
   RESET=1 ./run_linux_in_qemu.sh
   QEMU_DISK_SIZE=32G ./run_linux_in_qemu.sh    # override default disk size
   GUI=1 ./run_linux_in_qemu.sh                 # boot into Xfce in a QEMU window
   GUI=1 DESKTOP=gnome ./run_linux_in_qemu.sh   # boot into GNOME (Wayland)
   GUI=1 GUI_ACCEL=1 ./run_linux_in_qemu.sh     # add virgl GPU acceleration
   GUI=1 QEMU_RES=1920x1080 ./run_linux_in_qemu.sh  # custom boot resolution
   ```

3. **Run the agent**

   ```bash
   poetry sync
   poetry run llama_debugger_mcp_cli "List the running services and report any in failed state."
   ```

   Or just chat freely without an initial prompt:

   ```bash
   poetry run llama_debugger_mcp_cli
   ```

   Useful flags: `--llama-host`, `--llama-port` (defaults `127.0.0.1:53425`),
   `--single` (one-shot), `--no-tools` (chat only).

   `--split-screen` (alias `--tui`) opens a full-screen TUI: top frame is the
   chat with the model (reasoning, answer, tool calls), bottom frame mirrors
   the live SSH wire to the SUT (every command + raw output, timestamped),
   one-line input at the bottom. `Ctrl-C` / `Ctrl-Q` exits.

   ```bash
   poetry run llama_debugger_mcp_cli --split-screen
   ```

To stop QEMU:

```bash
killall qemu-system-x86_64
```

## Pointing at a different host

By default everything talks to `debian@127.0.0.1:2222`. To redirect against a
real board (or a second VM):

- From inside the agent chat, just ask it: *"Use 192.168.1.42 port 22 with
  user root password ..."* - the model will call
  `configuration_setTargetHost`.
- Or call the tool directly from any MCP client.

## Running GUI programs on the SUT

paramiko's `exec_command` runs a non-interactive, non-login shell, so
`~/.bashrc` / `~/.profile` are skipped and `sshd` strips most env vars
(`SendEnv` / `AcceptEnv` are off). To make GUI programs (lvglsim, glxgears,
GTK apps, ...) reach the desktop session running inside the SUT, the server
unconditionally prepends a set of `export K=V; ...` to every command run
through `_run_ssh_cmd`. The defaults are:

```
DISPLAY=:0
```

That covers Xfce / LightDM (real Xorg on `:0`) and GNOME / GDM (XWayland
also exposes `:0`). If you need an X cookie or a different display, mutate
the dict at runtime:

- *"Set the remote env XAUTHORITY to /home/debian/.Xauthority"* -> the
  model calls `configuration_setRemoteEnv`.
- `configuration_unsetRemoteEnv("DISPLAY")` strips a key.
- `configuration_getRemoteEnv()` shows what's currently exported.

The exports apply to **every** SSH-driven tool, including
`linux_run_in_background` (so a backgrounded `lvglsim` inherits `DISPLAY`)
and the persistent gdb tmux session.

## Requirements

**Host:** `poetry`, `genisoimage` (or `cloud-localds` / `mkisofs`),
`qemu-system-x86_64`, the ROCm-built `llama-server` referenced by
`start-llama-server.sh`. For `GUI=1` mode also: KVM access (your user in
the `kvm` group, or `/dev/kvm` otherwise readable+writable). For
`GUI_ACCEL=1` additionally: `libvirglrenderer1` for virgl host-OpenGL
passthrough.

**SUT:** `tmux` and `gdb` are required for `gdb_*` tools; `gcc`, `g++`, `make`
and `cmake` for `compiler_*`. On Debian/Ubuntu:

```bash
sudo apt-get install -y tmux gdb gcc g++ make cmake
```

## MCP notes

MCP is only needed if you want to plug a different client into the tool
server; `cli.py` calls llama.cpp directly and executes tools locally. To run
the MCP server stand-alone:

```bash
poetry run systemd_mcp_server
```
