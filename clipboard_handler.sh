#!/bin/bash

# Clipboard Screenshot Handler
# Handles Command+Shift+Control+4 clipboard screenshots
# Author: Claude AI Assistant
# Version: 1.0
# Date: 2025-07-09

set -euo pipefail

# Configuration
SCREENSHOT_DIR="$HOME/Pictures/Screenshots"
VAULT_DIR="$HOME/my-vault/test"
TEMP_DIR="$HOME/Library/Caches/claude-screenshots"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Check if clipboard contains an image
has_clipboard_image() {
    osascript -e 'clipboard info' 2>/dev/null | grep -q 'picture'
}

# Save clipboard image to file
save_clipboard_image() {
    local timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    local filename="clipboard_screenshot_${timestamp}.png"
    local filepath="$SCREENSHOT_DIR/$filename"
    
    # Create directories if they don't exist
    mkdir -p "$SCREENSHOT_DIR"
    mkdir -p "$TEMP_DIR"
    
    # Save clipboard to file using osascript
    if osascript << EOF
tell application "System Events"
    set theClipboard to the clipboard as «class PNGf»
    set theFile to open for access POSIX file "$filepath" with write permission
    write theClipboard to theFile
    close access theFile
end tell
EOF
    then
        log "Saved clipboard screenshot: $filename"
        echo "$filepath"
        return 0
    else
        log "Error: Failed to save clipboard image"
        return 1
    fi
}

# Watch clipboard for new images
monitor_clipboard() {
    log "Monitoring clipboard for screenshots..."
    log "Use Command+Shift+Control+4 to capture to clipboard"
    log "Press Ctrl+C to stop monitoring"
    
    local last_clipboard_hash=""
    
    while true; do
        if has_clipboard_image; then
            # Get a hash of the current clipboard content to detect changes
            local current_hash=$(osascript -e 'clipboard info' 2>/dev/null | md5)
            
            if [ "$current_hash" != "$last_clipboard_hash" ]; then
                log "New clipboard image detected"
                
                if filepath=$(save_clipboard_image); then
                    # Process the saved screenshot with the main manager
                    if [ -f "$HOME/Pictures/Screenshots/screenshot_manager.sh" ]; then
                        log "Processing with screenshot manager..."
                        "$HOME/Pictures/Screenshots/screenshot_manager.sh" process >/dev/null 2>&1
                    fi
                    
                    log "Screenshot saved and processed: $(basename "$filepath")"
                fi
                
                last_clipboard_hash="$current_hash"
            fi
        fi
        
        sleep 2  # Check every 2 seconds
    done
}

# Save current clipboard image if available
save_current() {
    if has_clipboard_image; then
        if filepath=$(save_clipboard_image); then
            # Process with screenshot manager
            if [ -f "$HOME/Pictures/Screenshots/screenshot_manager.sh" ]; then
                "$HOME/Pictures/Screenshots/screenshot_manager.sh" process >/dev/null 2>&1
            fi
            echo "Saved: $(basename "$filepath")"
        else
            echo "Error: Failed to save clipboard image"
            exit 1
        fi
    else
        echo "No image found in clipboard"
        echo "Use Command+Shift+Control+4 to capture a screenshot to clipboard first"
        exit 1
    fi
}

# Show usage information
show_help() {
    cat << 'EOF'
Clipboard Screenshot Handler - Handle Command+Shift+Control+4 screenshots

Usage: clipboard_handler.sh [COMMAND]

Commands:
  monitor   Monitor clipboard for new screenshots (default)
  save      Save current clipboard image if available
  help      Show this help message

Workflow:
  1. Use Command+Shift+Control+4 to capture to clipboard
  2. Run 'clipboard_handler.sh save' to save the current clipboard image
  3. Or use 'clipboard_handler.sh monitor' to automatically save new clipboard images

Integration:
  This script works with screenshot_manager.sh to automatically
  add references to the Screenshots.md file.

Examples:
  clipboard_handler.sh monitor    # Start monitoring clipboard
  clipboard_handler.sh save       # Save current clipboard image
  clipboard_handler.sh help       # Show help
EOF
}

# Main execution
main() {
    case "${1:-monitor}" in
        "monitor")
            monitor_clipboard
            ;;
        "save")
            save_current
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