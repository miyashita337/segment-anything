#!/bin/bash

# Claude Code Screenshot Integration - Main Entry Point
# Provides seamless /ss command functionality for Claude Code
# Author: Claude AI Assistant
# Version: 1.0
# Date: 2025-07-09

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_INTEGRATION="$SCRIPT_DIR/claude_integration.sh"

# Simple wrapper to get the latest screenshot path for Claude Code
# This is the main entry point for the /ss command

if [ -f "$CLAUDE_INTEGRATION" ]; then
    exec "$CLAUDE_INTEGRATION" latest
else
    echo "Error: Claude integration script not found at $CLAUDE_INTEGRATION"
    exit 1
fi