# llama.cpp.debugger

A safe playground for driving an embedded Linux SUT (System Under Test) from a
local LLM. The model runs on **llama.cpp** (`llama-server`) and acts on a Debian
VM brought up by QEMU through an SSH-driven tool surface.

Demo video: <https://www.youtube.com/watch?v=i8Lcic8HxLQ>

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
needs (`tmux`, `gdb`, `gcc`, `g++`, `make`, `cmake`, plus `linux-perf` and
`bzip2` for the `perf_*` tools). It also relaxes `kernel.perf_event_paranoid`
so the unprivileged `debian` user can record profiles without `sudo`. Image
source:
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

**`start-llama-embedding-server.sh`** - launches a *second* `llama-server`
instance on `http://0.0.0.0:53426` with `--embeddings --pooling mean` and a
dedicated embedding GGUF (default `nomic-embed-text-v1.5.Q8_0`). The chat
server on 53425 is left untouched - chat models make poor embedders, and a
server in `--embeddings` mode is no good for generation.

**`systemd_mcp/vectordb/`** - self-contained vector-DB demo. Embeds the
[LVGL docs](https://github.com/lvgl/lvgl/tree/master/docs) (Markdown +
MDX) through the embedding server above and stores the vectors in a
single `sqlite-vec` file. Driven by the `llama_debugger_vectordb` CLI;
see *Vector DB demo* below.

## Tool namespaces

| Prefix              | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `configuration_*`   | `setTargetHost(host, port, username, password)`, `getTargetHost`, `setRemoteEnv(name, value)`, `unsetRemoteEnv(name)`, `getRemoteEnv` |
| `systemd_*`         | service status, journal, start/stop/restart/enable/disable, daemon-reload, list, uptime |
| `linux_*`           | list/read/write/append/remove files, mkdir, cp, mv, find, grep, ps, df, which. Mutating tools (`write_file`, `append_file`, `remove`, `make_directory`, `copy`, `move`) take an optional `sudo: bool = False` flag - left off for the common user-space case (`/home/debian`, `/tmp`, build trees), set true only for root-owned trees (`/etc`, `/usr`, `/var`, `/opt`). |
| `compiler_*`        | `gcc`, `make`, `cmake_configure`, `cmake_build`                |
| `gdb_*`             | persistent tmux session: attach/run/core (each takes optional `sudo`; `attach` defaults to `sudo=true` for ptrace across UIDs, `run` and `core` default off), send_command, break/continue/step/next/finish, print, backtrace, info_registers, info_threads, list_breakpoints, read_output, quit |
| `perf_*`            | Linux `perf` profiling: `stat` (cycles/IPC/cache summary), `record` (time-boxed sampling, works for GUI/long-running programs), `report`, `top_functions` (JSON), `heatmap` (renders a function-overhead treemap PNG on the host), `open_hotspot` (pulls the recording to the host and opens it in the interactive [Hotspot](https://github.com/KDAB/hotspot) GUI with cross-machine symbols). See *Profiling demo* below. |
| `rag_*`             | `search(query, k)` against the LVGL docs vector DB built by `llama_debugger_vectordb`. Soft-fails when the embedding server (port 53426) or the DB are missing - the chat continues, the model is told to advise rebuilding. |

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

## Profiling demo (perf → heatmap / Hotspot)

The `perf_*` tools let the agent CPU-profile a program on the SUT and bring
the result back to the host for visualization. `perf_record` is time-boxed
(via `timeout`), so it works even for GUI / never-returning programs like the
LVGL simulator: the agent just says "record this for N seconds".

**`run_mcp_demo.sh`** wires the whole flow end-to-end against
`lvglsim -b GLFW`. It does the model-unfriendly setup *itself*
(deterministically) so even a small model only has to make the two tool
calls that matter:

```bash
GUI=1 GUI_ACCEL=1 ./run_linux_in_qemu.sh   # SUT with the LVGL sim built
./start-llama-server.sh                    # chat model
./run_mcp_demo.sh                          # profile + visualize
```

What it does:

- Copies the live display-manager X cookie to a `debian`-readable path and
  exports `XAUTHORITY` (via `LLAMA_DEBUGGER_REMOTE_ENV`) so the GUI program
  can open its window - the model never touches SSH creds or X cookies.
- Prompts the agent to `perf_record` the sim, then `perf_heatmap` it.
- When the chat finishes, opens the rendered heatmap PNG (`xdg-open`) and,
  by default, the raw profile in **Hotspot** (`OPEN_HOTSPOT=0` to skip).

Tunables (env): `LVGLSIM_BIN`, `LVGLSIM_ARGS`, `PERF_DURATION`,
`HEATMAP_PNG`, `SUT_PERF_DATA`, `OPEN_HOTSPOT`.

### Two ways to view a profile

- **Heatmap PNG** (`perf_heatmap`) - pulls the hottest functions over SSH and
  renders a squarified treemap on the *host* with `matplotlib` (tile area =
  self-overhead, color = blue-cold → red-hot). Headless (Agg backend), so it
  works over SSH with no display. Returns `{png, functions, count}`.
- **Hotspot GUI** (`perf_open_hotspot`) - Hotspot runs on the *host* but
  `perf.data` lives on the SUT, so the tool copies the recording over SFTP
  and, by default, runs `perf archive` on the SUT to bundle the build-id'd
  binaries/libraries, extracting them into the host build-id cache
  (`~/.debug`) so SUT symbols resolve **across machines**. It launches
  Hotspot with debuginfod disabled (`DEBUGINFOD_URLS=""`) - otherwise the
  parser stalls on "Loading Results..." querying a debuginfod server for the
  SUT's distro libs; the archive already supplies the symbols. Pass
  `use_debuginfod=true` to opt back in.

Both soft-fail cleanly: `perf_heatmap` if `matplotlib` is missing on the
host, `perf_open_hotspot` if `hotspot` isn't on the host `PATH` or there's
no recording yet.

## Vector DB demo (LVGL docs + source)

llama.cpp itself is an inference engine - it computes embeddings via
`llama-server --embeddings` (and the one-shot `llama-embedding` CLI), but
it does **not** ship a vector database. The `systemd_mcp.vectordb`
subpackage closes that gap with a thin local store on top of
[`sqlite-vec`](https://github.com/asg017/sqlite-vec) and a CLI that can
build, query, inspect, and wipe it.

The demo corpus is **both halves** of the
[`lvgl/lvgl`](https://github.com/lvgl/lvgl) repo:

- **Documentation**: `docs/**/*.{md,mdx}` (~415 files / ~2.8k chunks).
- **C source**: `src/**/*.{c,h}` and `examples/**/*.{c,h}` (~1285 files /
  ~9k chunks). Each chunk is a ~50-line line-window; the chunk's
  "heading" is the *enclosing* function name as captured by a
  `^TYPE name(args) {` regex, so the agent can cite
  `src/widgets/button/lv_button.c :: lv_button_create` or
  `src/misc/lv_anim.c :: lv_anim_start`.

`examples/assets/` (pre-encoded image/animation pixel arrays) and
`src/font/` (bitmap font glyph tables) are excluded by default in
[`systemd_mcp/vectordb/ingest.py`](systemd_mcp/vectordb/ingest.py)'s
`DEFAULT_EXCLUDE_PREFIXES` - they're mechanical data, not retrievable
prose, and would inflate the embedding cost by ~30% with no demo
value. Re-enable by passing `exclude_prefixes=()` to `iter_source_files`.

That gives the chat agent ~12k retrievable chunks covering both *what
the API is supposed to do* (docs prose) and *what it actually does at
the C level* (function bodies, structs, X-macro tables). It pairs
naturally with this project since the SUT side is already wired to run
`lvglsim` and other LVGL programs (see *Running GUI programs on the
SUT*).

### One-time setup

Drop an embedding GGUF next to the chat model (path inside
`start-llama-embedding-server.sh`). Recommended:

```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF \
  nomic-embed-text-v1.5.Q8_0.gguf \
  --local-dir ../llama.cpp/llama-b9940-bin-ubuntu-rocm-7.2-x64/llama-b9940
```

`bge-small-en-v1.5` (smaller) and `bge-m3` (multilingual) are listed as
alternatives in the launcher; if you switch, also pass
`--embed-family generic` to the CLI so it stops adding nomic's
`search_document:` / `search_query:` prefixes.

### Demo flow

```bash
./start-llama-server.sh                  # terminal 1: chat (port 53425)
./start-llama-embedding-server.sh        # terminal 2: embeddings (port 53426)

poetry sync                              # picks up sqlite-vec + numpy

poetry run llama_debugger_vectordb build    # clones lvgl, embeds docs + src + examples
poetry run llama_debugger_vectordb info     # backend, dim, chunk count, source commit
poetry run llama_debugger_vectordb query "lv_anim opacity fade ready_cb" --k 5
poetry run llama_debugger_vectordb delete --yes
```

**`build` is destructive and not fast.** It runs a sparse + partial +
shallow `git clone` of `lvgl/lvgl` into `./.cache/lvgl/` (only `docs/`,
`src/` and `examples/` materialize on disk; ~30 MB), then dispatches per
file extension:

- `.md` / `.mdx`: strip YAML frontmatter and MDX JSX components
  (`<Callout>`, `<LvglExample>`, `<Figure>`, ...), split by `##` / `###`
  headings, then ~800-char windows with 120-char overlap. Heading
  breadcrumb is `H1 > H2 > H3`.
- `.c` / `.h`: ~50-line windows with 8-line overlap. Heading is the
  *enclosing function name* matched against
  `^TYPE name(args) { ... }` (best-effort - misses are tagged with an
  empty heading).

Everything goes into `./systemd_mcp/vectordb/vector-database.db`
(~50-80 MB; gitignored). Embedding
takes ~5-15 minutes on the iGPU embedding server depending on its
throughput - it's a one-shot cost; `info` lets you watch chunk_count
climb in another terminal while it runs. Override the source with
`--source` (git URL, path to a clone, or path directly at a leaf
subdir) and the output with `--db`.

`query` embeds your text via the same embedding server, runs an exact
KNN against `sqlite-vec`, and prints the top-k chunks with the file
path, heading (function name for code chunks), and a snippet.

For a graphical alternative, `poetry run llama_debugger_vectordb_ui`
opens a native (Toga/GTK) panel that streams `journalctl` over SSH into
the store, runs the same searches, and manages the `.db` file - see
[`systemd_mcp/vectordb_ui/README.md`](systemd_mcp/vectordb_ui/README.md).

### Using the store from the chat agent

Once the DB exists, the chat agent gets it for free: a `rag_search`
tool is registered in [`systemd_mcp/server.py`](systemd_mcp/server.py)
and surfaced through `llama_debugger_mcp_cli` (and the FastMCP server)
under the `rag_*` namespace. The model's system prompt says the
*hard* version: for any LVGL/`lv_*`/widget/animation/style task it
**must** call `rag_search` at least once, **separately** for each
distinct concept, and **must** cite the returned paths in its answer.
With both the docs and the source code embedded, an answer typically
mixes citations from `docs/src/...mdx` (prose) and `src/...c` (the
actual function body), e.g.:

```text
> Add a button to my LVGL screen that, when clicked, fades its background
> opacity from 100% to 0% over 800 ms then deletes itself. Use rag_search
> separately for: button creation, click event, lv_anim opacity setup,
> lv_anim ready_cb, lv_obj_delete. Cite the paths for every API.

[tool_call rag_search query="lv_button_create parent" k=4]
[tool_result hits=[{score:0.78, path:"src/widgets/button/lv_button.c",
                    heading:"lv_button_create", text:"..."}, ...]]
[tool_call rag_search query="lv_obj_add_event_cb LV_EVENT_CLICKED" k=4]
[tool_call rag_search query="lv_anim_t opacity exec_cb lv_obj_set_style_bg_opa" k=4]
[tool_call rag_search query="lv_anim_t ready_cb lv_obj_delete callback" k=4]
[tool_call rag_search query="lv_obj_delete" k=3]
... model writes the C code citing each src/... path ...
```

That's **5 distinct calls to `/v1/embeddings`** for one question. You'll
see them stream past in the embedding-server terminal as
`POST /v1/embeddings` lines, with the chat-server terminal idling at
"all slots are idle" between them - i.e. the embedding GPU genuinely
participates in every LVGL turn.

`rag_search` **soft-fails** (returns `{"hits": [], "error": "..."}`) when
either the embedding server or the DB is missing, so a chat session
still works without `./start-llama-embedding-server.sh` or
`llama_debugger_vectordb build`; the model just won't have grounding
and will tell the user to start the missing piece.

Override paths via env vars (handy if you ship a different corpus):

```bash
export LLAMA_DEBUGGER_VECTORDB=/path/to/my-corpus.db
export LLAMA_DEBUGGER_EMBED_HOST=192.168.1.42
export LLAMA_DEBUGGER_EMBED_PORT=53426
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
`start-llama-server.sh`. `matplotlib` (pulled in by `poetry sync`) for
`perf_heatmap`, and optionally `hotspot` for `perf_open_hotspot`. For
`GUI=1` mode also: KVM access (your user in the `kvm` group, or `/dev/kvm`
otherwise readable+writable). For `GUI_ACCEL=1` additionally:
`libvirglrenderer1` for virgl host-OpenGL passthrough.

**SUT:** `tmux` and `gdb` are required for `gdb_*` tools; `gcc`, `g++`, `make`
and `cmake` for `compiler_*`; `linux-perf` (and `bzip2`, used by
`perf archive`) for `perf_*`. On Debian/Ubuntu:

```bash
sudo apt-get install -y tmux gdb gcc g++ make cmake linux-perf bzip2
```

## MCP notes

MCP is only needed if you want to plug a different client into the tool
server; `cli.py` calls llama.cpp directly and executes tools locally. To run
the MCP server stand-alone:

```bash
poetry run systemd_mcp_server
```
