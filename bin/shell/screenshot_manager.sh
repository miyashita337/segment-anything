#!/bin/bash

# Screenshot Manager Script
# Automatically organizes and references screenshots
# Author: Claude AI Assistant
# Version: 1.0
# Date: 2025-07-09

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCREENSHOT_DIR="$HOME/Pictures/Screenshots"
VAULT_DIR="$HOME/my-vault/test"
REFERENCE_FILE="$VAULT_DIR/Screenshots.md"
SUPPORTED_FORMATS=("*.png" "*.jpg" "*.jpeg" "*.gif" "*.bmp")

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Initialize reference file if it doesn't exist
init_reference_file() {
    if [ ! -f "$REFERENCE_FILE" ]; then
        mkdir -p "$(dirname "$REFERENCE_FILE")"
        cat > "$REFERENCE_FILE" << 'EOF'
# Screenshots

Automatically generated screenshot references.

EOF
        log "Created reference file: $REFERENCE_FILE"
    fi
}

# Check if file is already referenced
is_already_referenced() {
    local file="$1"
    local filename=$(basename "$file")
    grep -q "$filename" "$REFERENCE_FILE" 2>/dev/null
}

# Function to add screenshot reference
add_screenshot_reference() {
    local screenshot_file="$1"
    
    if [ ! -f "$screenshot_file" ]; then
        log "Error: Screenshot file not found: $screenshot_file"
        return 1
    fi
    
    if is_already_referenced "$screenshot_file"; then
        log "Screenshot already referenced: $(basename "$screenshot_file")"
        return 0
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local filename=$(basename "$screenshot_file")
    local filesize=$(du -h "$screenshot_file" | cut -f1)
    
    {
        echo "## Screenshot - $timestamp"
        echo ""
        echo "![Screenshot]($screenshot_file)"
        echo ""
        echo "**File:** \`$filename\`"
        echo "**Path:** \`$screenshot_file\`"
        echo "**Size:** $filesize"
        echo ""
        echo "---"
        echo ""
    } >> "$REFERENCE_FILE"
    
    log "Added reference for: $filename"
}

# Monitor for new screenshots
monitor_screenshots() {
    init_reference_file
    log "Monitoring screenshots in $SCREENSHOT_DIR"
    
    if ! command -v fswatch >/dev/null 2>&1; then
        log "Error: fswatch not found. Install with: brew install fswatch"
        return 1
    fi
    
    # Create screenshot directory if it doesn't exist
    mkdir -p "$SCREENSHOT_DIR"
    
    log "Press Ctrl+C to stop monitoring"
    fswatch -o "$SCREENSHOT_DIR" | while read -r f; do
        sleep 1  # Small delay to ensure file is completely written
        
        for format in "${SUPPORTED_FORMATS[@]}"; do
            for file in "$SCREENSHOT_DIR"/$format; do
                if [ -f "$file" ] && [ "$file" -nt "$REFERENCE_FILE" ]; then
                    add_screenshot_reference "$file"
                fi
            done
        done
    done
}

# Manual mode - process existing screenshots
process_existing() {
    init_reference_file
    log "Processing existing screenshots in $SCREENSHOT_DIR"
    
    local count=0
    for format in "${SUPPORTED_FORMATS[@]}"; do
        for file in "$SCREENSHOT_DIR"/$format; do
            if [ -f "$file" ]; then
                if add_screenshot_reference "$file"; then
                    ((count++))
                fi
            fi
        done
    done
    
    if [ $count -eq 0 ]; then
        log "No new screenshots to process"
    else
        log "Processed $count screenshot(s)"
    fi
}

# Show usage information
show_help() {
    cat << 'EOF'
Screenshot Manager - Automatically organize and reference screenshots

Usage: screenshot_manager.sh [COMMAND]

Commands:
  monitor   Watch for new screenshots and auto-reference them (default)
  process   Process existing screenshots and add references
  help      Show this help message

Configuration:
  Screenshot Dir: ~/Pictures/Screenshots
  Reference File: ~/my-vault/test/Screenshots.md
  Supported Formats: PNG, JPG, JPEG, GIF, BMP

Examples:
  screenshot_manager.sh monitor    # Start monitoring
  screenshot_manager.sh process    # Process existing files
  screenshot_manager.sh help       # Show help
EOF
}

# Main execution
main() {
    case "${1:-monitor}" in
        "monitor")
            monitor_screenshots
            ;;
        "process")
            process_existing
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