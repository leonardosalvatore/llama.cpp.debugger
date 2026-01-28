Here a safe playground where you can play with a linux system without braking yours.
This is an Ollama tools demo that shows how to control an embedded Linux, systemd-based system.
It also includes an MCP server that exposes the same tools for any MCP-compatible client.

**cli.py**
The Ollama terminal client that exposes the tools directly to the Ollama model.
It talks to Ollama only, so it does not require MCP.
It can use any model that offer tools and thinking capability.


**server.py**
The MCP server that exposes basic Linux/systemd tools over the MCP protocol.
Use this only if you want to connect another MCP-capable client (instead of `cli.py`).


**run_embedded.sh**
The embedded system runs inside a QEMU emulator; this script starts it.
This is the image that will run:
https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2
It injects SSH configuration and creates a failing service for the demo.
To use the Debian shell, press Enter and log in with user: debian / password: debian.


**demo**
1. Init
It's a Poetry project, so run: `poetry sync`.
To run this project, open 2 terminals.
2. Run:  `poetry run systemd_mcp_cli --model qwen3:latest "Is ths SUT running fine without any failure`
3. Try with smaller model
`poetry run systemd_mcp_cli --model qwen3:0.6b "Is ths SUT running fine without any failure`

Or just chat with the agent without a prompt: `poetry run systemd_mcp_cli`

To stop QEMU, run:
`killall qemu-system-x86_64`

**Requirements**
poetry, genisoimage, qemu

**MCP notes**
- MCP is only needed if you want to connect a different client to the tool server.
- The `cli.py` path does not use MCP; it calls Ollama directly and executes tools locally.
