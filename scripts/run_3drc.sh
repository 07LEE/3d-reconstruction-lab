#!/bin/bash

# 3DRC Command Line Interface
# Arguments are parsed by src/cli/main.py (ADR 0022); this script and 3drc.py only forward to it.
# Run it from the project root: stage scripts and relative dataset paths resolve from there.

exec python3 "$(dirname "$0")/../3drc.py" "$@"
