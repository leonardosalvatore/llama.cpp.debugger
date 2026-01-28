This is a sinmple MCP demo that show how to control some embedded linux systemd based system.

**cli.py**
The Ollama terminal client that will expose the functions to the Ollama LLM model.
It does use the Ollama directly so does not require anything else. 


**server.py**
The MCP server that expose some basic linux and systemd functions.
Can be used with any "AI" tool that talk MCP protocol. 


**run_embedded.sh**
The embedded system runs inside a qemu emulator, this script is to start it.
This is the image that will run.
https://www.google.com/search?q=https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2
It also inject some configuration for ssh and create a service file that is failing. This is for the demo.
To use this Debian bash just press enter and enter with user:debian psw:debian


**demo**
1. Init
it's a poetry project so run -poetry sync-
To run this project just open 2 terminals.
2. run_empoetry run systemd_mcp_cli --model qwen3:latest "get all the service. Inspect the log to investigate which one is failing."

Or just chat with the agent without proposing a prompt:
`poetry run systemd_mcp_cli`

To stop the qemu just 
`killall qemu-system-x86_64`

**Requirements**
poetry,  genisoimage , qemu
