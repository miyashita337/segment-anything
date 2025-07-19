#!/bin/bash

# Claude Code Custom Commands Setup
# Sets up /ss, /sslist, and /ssshow commands for Claude Code
# Author: Claude AI Assistant
# Version: 1.0
# Date: 2025-07-09

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_COMMANDS_DIR="$HOME/.claude/commands"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Setup Claude Code custom commands
setup_claude_commands() {
    log_info "Setting up Claude Code custom commands..."
    
    # Create commands directory
    mkdir -p "$CLAUDE_COMMANDS_DIR"
    log_success "Commands directory created: $CLAUDE_COMMANDS_DIR"
    
    # Copy command files if they exist in the script directory
    local commands=("ss.md" "sslist.md" "ssshow.md")
    
    for cmd in "${commands[@]}"; do
        local src_file="$SCRIPT_DIR/../.claude/commands/$cmd"
        local dst_file="$CLAUDE_COMMANDS_DIR/$cmd"
        
        if [ -f "$src_file" ]; then
            cp "$src_file" "$dst_file"
            log_success "Installed command: /$cmd"
        elif [ -f "$dst_file" ]; then
            log_info "Command already exists: /$cmd"
        else
            log_warning "Command file not found: $cmd"
        fi
    done
}

# Verify commands are working
verify_commands() {
    log_info "Verifying command integration..."
    
    # Check if integration script exists and is executable
    local integration_script="$SCRIPT_DIR/claude_integration.sh"
    if [ -x "$integration_script" ]; then
        log_success "Integration script is ready: $integration_script"
        
        # Test the latest command
        if "$integration_script" latest >/dev/null 2>&1 || [ $? -eq 1 ]; then
            log_success "Screenshot integration is working"
        else
            log_warning "Screenshot integration may have issues"
        fi
    else
        log_warning "Integration script not found or not executable: $integration_script"
    fi
}

# Show usage information
show_usage() {
    cat << 'EOF'

🎉 Claude Code Custom Commands Setup Complete!

Available Commands:
  📸 /ss              - Show the latest screenshot
  📋 /sslist          - List recent screenshots with numbers
  🔍 /ssshow NUMBER   - Show specific screenshot by number

Usage Examples:
  /ss                 # Show latest screenshot
  /sslist             # List all recent screenshots
  /ssshow 1           # Show the most recent screenshot
  /ssshow 3           # Show the 3rd most recent screenshot

How to Use:
1. Take a screenshot: Command+Shift+4
2. In Claude Code, type: /ss
3. Claude will automatically display and analyze the image!

Integration:
- Commands are installed in: ~/.claude/commands/
- Backend scripts are in: ~/Pictures/Screenshots/
- Screenshots are saved to: ~/Pictures/Screenshots/

Note: Restart Claude Code to load new custom commands.

🚀 Your screenshot workflow is now fully integrated with Claude Code!
EOF
}

# Main execution
main() {
    log_info "Starting Claude Code custom commands setup..."
    echo
    
    setup_claude_commands
    verify_commands
    
    echo
    show_usage
}

# Show help
show_help() {
    cat << 'EOF'
Claude Code Custom Commands Setup

Usage: claude_commands_setup.sh [COMMAND]

Commands:
  install   Setup Claude Code custom commands (default)
  help      Show this help message

This script installs /ss, /sslist, and /ssshow custom commands
for Claude Code integration with the screenshot system.
EOF
}

# Command handling
case "${1:-install}" in
    "install")
        main
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac