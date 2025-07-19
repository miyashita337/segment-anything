#!/bin/bash

# Claude Code Screenshot Integration
# Provides /ss command functionality for Claude Code
# Author: Claude AI Assistant
# Version: 1.0
# Date: 2025-07-09

set -euo pipefail

# Configuration
SCREENSHOT_DIR="/mnt/c/Users/shakufuku/Pictures/Screenshots"
VAULT_DIR="$HOME/my-vault/test"
REFERENCE_FILE="$VAULT_DIR/Screenshots.md"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Get the most recent screenshot
get_latest_screenshot() {
    ls -t "$SCREENSHOT_DIR"/*.{png,jpg,jpeg,gif,bmp} 2>/dev/null | head -1
}

# Get all recent screenshots (last 10)
get_recent_screenshots() {
    ls -t "$SCREENSHOT_DIR"/*.{png,jpg,jpeg,gif,bmp} 2>/dev/null | head -10
}

# List screenshots with numbers for selection
list_screenshots() {
    local screenshots=()
    while IFS= read -r line; do
        screenshots+=("$line")
    done < <(get_recent_screenshots)
    
    if [ ${#screenshots[@]} -eq 0 ]; then
        echo "No screenshots found in $SCREENSHOT_DIR"
        return 1
    fi
    
    echo "Recent screenshots:"
    for i in "${!screenshots[@]}"; do
        local file="${screenshots[$i]}"
        local filename=$(basename "$file")
        local date=$(date -r "$file" '+%Y-%m-%d %H:%M:%S')
        local size=$(du -h "$file" | cut -f1)
        printf "%2d. %s (%s, %s)\n" $((i+1)) "$filename" "$date" "$size"
    done
}

# Show the latest screenshot for Claude Code /ss command
show_latest() {
    local latest=$(get_latest_screenshot)
    
    if [ -z "$latest" ]; then
        echo "No screenshots found in $SCREENSHOT_DIR"
        echo "Use Command+Shift+4 to take a screenshot first."
        return 1
    fi
    
    local filename=$(basename "$latest")
    local date=$(date -r "$latest" '+%Y-%m-%d %H:%M:%S')
    local size=$(du -h "$latest" | cut -f1)
    
    log "Latest screenshot: $filename ($date, $size)"
    
    # Output the path for Claude Code to read
    echo "$latest"
    
    # Update reference file if not already done
    if [ -f "$HOME/Pictures/Screenshots/screenshot_manager.sh" ]; then
        "$HOME/Pictures/Screenshots/screenshot_manager.sh" process >/dev/null 2>&1
    fi
}

# Show a specific screenshot by number
show_screenshot() {
    local number="$1"
    local screenshots=()
    while IFS= read -r line; do
        screenshots+=("$line")
    done < <(get_recent_screenshots)
    
    if [ ${#screenshots[@]} -eq 0 ]; then
        echo "No screenshots found in $SCREENSHOT_DIR"
        return 1
    fi
    
    if [[ ! "$number" =~ ^[0-9]+$ ]] || [ "$number" -lt 1 ] || [ "$number" -gt ${#screenshots[@]} ]; then
        echo "Error: Invalid screenshot number. Use 1-${#screenshots[@]}"
        return 1
    fi
    
    local index=$((number - 1))
    local selected="${screenshots[$index]}"
    local filename=$(basename "$selected")
    local date=$(date -r "$selected" '+%Y-%m-%d %H:%M:%S')
    local size=$(du -h "$selected" | cut -f1)
    
    log "Selected screenshot #$number: $filename ($date, $size)"
    
    # Output the path for Claude Code to read
    echo "$selected"
}

# Interactive mode for screenshot selection
interactive_select() {
    list_screenshots
    
    local screenshots=()
    while IFS= read -r line; do
        screenshots+=("$line")
    done < <(get_recent_screenshots)
    
    if [ ${#screenshots[@]} -eq 0 ]; then
        return 1
    fi
    
    echo
    read -p "Select screenshot number (1-${#screenshots[@]}): " selection
    
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#screenshots[@]} ]; then
        show_screenshot "$selection"
    else
        echo "Error: Invalid selection"
        return 1
    fi
}

# Setup alias for easy access
setup_alias() {
    local shell_rc=""
    
    if [ -n "${ZSH_VERSION:-}" ]; then
        shell_rc="$HOME/.zshrc"
    elif [ -n "${BASH_VERSION:-}" ]; then
        shell_rc="$HOME/.bashrc"
    fi
    
    if [ -n "$shell_rc" ] && [ -f "$shell_rc" ]; then
        if ! grep -q "alias ss=" "$shell_rc"; then
            echo "" >> "$shell_rc"
            echo "# Claude screenshot integration" >> "$shell_rc"
            echo "alias ss='$HOME/Pictures/Screenshots/claude_integration.sh latest'" >> "$shell_rc"
            echo "alias sslist='$HOME/Pictures/Screenshots/claude_integration.sh list'" >> "$shell_rc"
            echo "alias ssselect='$HOME/Pictures/Screenshots/claude_integration.sh select'" >> "$shell_rc"
            echo "" >> "$shell_rc"
            
            log "Added aliases to $shell_rc"
            log "Restart your shell or run 'source $shell_rc' to use the aliases"
        else
            log "Aliases already exist in $shell_rc"
        fi
    fi
}

# Show usage information
show_help() {
    cat << 'EOF'
Claude Code Screenshot Integration - /ss command functionality

Usage: claude_integration.sh [COMMAND] [OPTIONS]

Commands:
  latest        Show the most recent screenshot path (default)
  list          List recent screenshots with numbers
  select        Interactive screenshot selection
  show NUMBER   Show specific screenshot by number
  setup         Setup shell aliases
  help          Show this help message

Aliases (after running 'setup'):
  ss            Show latest screenshot
  sslist        List recent screenshots
  ssselect      Interactive selection

Claude Code Integration:
  This script provides the /ss functionality for Claude Code.
  The /ss command should read the output path and display the image.

Examples:
  claude_integration.sh latest      # Get latest screenshot path
  claude_integration.sh list        # List all recent screenshots
  claude_integration.sh show 3      # Show screenshot #3
  claude_integration.sh select      # Interactive selection
  claude_integration.sh setup       # Setup shell aliases

Workflow:
  1. Take screenshot with Command+Shift+4
  2. Use /ss or run this script to get the path
  3. Claude Code reads and displays the image
EOF
}

# Main execution
main() {
    case "${1:-latest}" in
        "latest")
            show_latest
            ;;
        "list")
            list_screenshots
            ;;
        "select")
            interactive_select
            ;;
        "show")
            if [ $# -lt 2 ]; then
                echo "Error: Screenshot number required"
                echo "Usage: $0 show NUMBER"
                exit 1
            fi
            show_screenshot "$2"
            ;;
        "setup")
            setup_alias
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            log "Error: Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi