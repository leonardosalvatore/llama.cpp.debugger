This is a sinmple MCP demo that show how to control some embedded linux systemd based system.

**server.py**
The MCP server that expose some basic linux and systemd functions.

**cli.py**
The Ollama terminal client that will reach the MCP server.

**run_embedded.sh**
The embedded system runs inside a qemu emulator, this script is to start it.
This is the image that will run.
https://www.google.com/search?q=https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2
It also inject some configuration for ssh and create a service file that is failing. This is for the demo.


To run this project just open 2 terminals.
1. run_embedded.sh
for tiny GPU low on ram
2. poetry run systemd_mcp_cli --max-tool-steps 50 --model qwen3:4b "get all the service, find which one are failing and report here the reason of the failure"
or 
2. poetry run systemd_mcp_cli --max-tool-steps 50 --model qwen2.5-coder:14b "get all the service, find which one are failing and report here the reason of the failure"


poetry run systemd_mcp_cli --max-tool-steps 50 --model qwen2.5-coder:14b "get all the service. Inspect the log to investigate which one is failing."

Or just chat with the agent without proposing a prompt:
poetry run systemd_mcp_cli

**Requirements**
poetry,  genisoimage , qemu
