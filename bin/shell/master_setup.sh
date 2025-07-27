#!/bin/bash

# Master Setup Script - Complete Screenshot System Setup
# Sets up all components for the Claude Code screenshot integration
# Author: Claude AI Assistant
# Version: 1.0
# Date: 2025-07-09

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREENSHOT_DIR="$HOME/Pictures/Screenshots"
VAULT_DIR="$HOME/my-vault/test"

# Logging with colors
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This script is designed for macOS only"
        exit 1
    fi
    
    # Check if Homebrew is installed
    if ! command -v brew >/dev/null 2>&1; then
        log_warning "Homebrew not found. Some features may not work."
    fi
    
    # Check if fswatch is installed
    if ! command -v fswatch >/dev/null 2>&1; then
        log_info "Installing fswatch..."
        if command -v brew >/dev/null 2>&1; then
            brew install fswatch
            log_success "fswatch installed"
        else
            log_error "Cannot install fswatch without Homebrew"
            exit 1
        fi
    else
        log_success "fswatch is already installed"
    fi
}

# Setup macOS screenshot configuration
setup_macos_screenshots() {
    log_info "Configuring macOS screenshot settings..."
    
    # Create screenshot directory
    mkdir -p "$SCREENSHOT_DIR"
    log_success "Screenshot directory created: $SCREENSHOT_DIR"
    
    # Set screenshot location
    defaults write com.apple.screencapture location "$SCREENSHOT_DIR"
    log_success "Screenshot location set to: $SCREENSHOT_DIR"
    
    # Restart SystemUIServer to apply changes
    killall SystemUIServer 2>/dev/null || true
    log_success "SystemUIServer restarted"
}

# Setup Obsidian vault integration
setup_vault_integration() {
    log_info "Setting up Obsidian vault integration..."
    
    # Create vault directory if it doesn't exist
    mkdir -p "$VAULT_DIR"
    log_success "Vault directory ensured: $VAULT_DIR"
    
    # Initialize Screenshots.md if it doesn't exist
    local reference_file="$VAULT_DIR/Screenshots.md"
    if [ ! -f "$reference_file" ]; then
        cat > "$reference_file" << 'EOF'
# Screenshots

Automatically generated screenshot references.

EOF
        log_success "Created reference file: $reference_file"
    else
        log_info "Reference file already exists: $reference_file"
    fi
}

# Setup shell aliases
setup_shell_aliases() {
    log_info "Setting up shell aliases..."
    
    local shell_rc=""
    if [ -n "${ZSH_VERSION:-}" ]; then
        shell_rc="$HOME/.zshrc"
    elif [ -n "${BASH_VERSION:-}" ]; then
        shell_rc="$HOME/.bashrc"
    fi
    
    if [ -n "$shell_rc" ] && [ -f "$shell_rc" ]; then
        if ! grep -q "screenshot-monitor" "$shell_rc"; then
            {
                echo ""
                echo "# Claude Screenshot System Aliases"
                echo "alias screenshot-monitor='$SCRIPT_DIR/screenshot_manager.sh monitor'"
                echo "alias screenshot-process='$SCRIPT_DIR/screenshot_manager.sh process'"
                echo "alias clipboard-monitor='$SCRIPT_DIR/clipboard_handler.sh monitor'"
                echo "alias clipboard-save='$SCRIPT_DIR/clipboard_handler.sh save'"
                echo "alias ss='$SCRIPT_DIR/ss'"
                echo "alias sslist='$SCRIPT_DIR/claude_integration.sh list'"
                echo "alias ssselect='$SCRIPT_DIR/claude_integration.sh select'"
                echo ""
            } >> "$shell_rc"
            log_success "Added aliases to $shell_rc"
            log_info "Restart your shell or run 'source $shell_rc' to use aliases"
        else
            log_info "Aliases already exist in $shell_rc"
        fi
    else
        log_warning "Could not determine shell configuration file"
    fi
}

# Setup LaunchAgent for automatic monitoring
setup_launch_agent() {
    log_info "Setting up LaunchAgent for automatic screenshot monitoring..."
    
    local launch_agent_dir="$HOME/Library/LaunchAgents"
    local launch_agent_file="$launch_agent_dir/com.claude.screenshot-monitor.plist"
    
    mkdir -p "$launch_agent_dir"
    
    cat > "$launch_agent_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claude.screenshot-monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_DIR/screenshot_manager.sh</string>
        <string>monitor</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/claude-screenshot-monitor.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/claude-screenshot-monitor.error.log</string>
</dict>
</plist>
EOF
    
    log_success "LaunchAgent created: $launch_agent_file"
    log_info "To enable automatic startup: launchctl load $launch_agent_file"
    log_info "To disable automatic startup: launchctl unload $launch_agent_file"
}

# Verify all components
verify_installation() {
    log_info "Verifying installation..."
    
    local errors=0
    
    # Check if all scripts are executable
    for script in "screenshot_manager.sh" "clipboard_handler.sh" "claude_integration.sh" "ss"; do
        if [ -x "$SCRIPT_DIR/$script" ]; then
            log_success "$script is executable"
        else
            log_error "$script is not executable"
            ((errors++))
        fi
    done
    
    # Test screenshot manager
    if "$SCRIPT_DIR/screenshot_manager.sh" help >/dev/null 2>&1; then
        log_success "screenshot_manager.sh is working"
    else
        log_error "screenshot_manager.sh is not working"
        ((errors++))
    fi
    
    # Test Claude integration
    if "$SCRIPT_DIR/claude_integration.sh" help >/dev/null 2>&1; then
        log_success "claude_integration.sh is working"
    else
        log_error "claude_integration.sh is not working"
        ((errors++))
    fi
    
    # Test /ss command
    if "$SCRIPT_DIR/ss" >/dev/null 2>&1 || [ $? -eq 1 ]; then  # Exit code 1 is expected when no screenshots
        log_success "/ss command is working"
    else
        log_error "/ss command is not working"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "All components verified successfully!"
        return 0
    else
        log_error "Verification failed with $errors error(s)"
        return 1
    fi
}

# Show usage summary
show_usage() {
    cat << 'EOF'

🎉 Claude Screenshot System Setup Complete!

Usage:
  📸 Screenshot Commands:
    Command+Shift+4           - Save screenshot to file
    Command+Shift+Control+4   - Copy screenshot to clipboard

  🔧 Management Commands:
    screenshot-monitor        - Monitor for new file screenshots
    screenshot-process        - Process existing screenshots
    clipboard-monitor         - Monitor clipboard for screenshots
    clipboard-save           - Save current clipboard screenshot

  🔍 Claude Code Integration:
    ss                       - Get latest screenshot for Claude (/ss command)
    sslist                   - List recent screenshots
    ssselect                 - Interactive screenshot selection

  📁 Files:
    Screenshots Dir:         ~/Pictures/Screenshots
    Reference File:          ~/my-vault/test/Screenshots.md
    Log Files:               ~/Library/Logs/claude-screenshot-*

  ⚙️  Optional Auto-start:
    launchctl load ~/Library/LaunchAgents/com.claude.screenshot-monitor.plist

Enjoy your automated screenshot workflow! 🚀
EOF
}

# Main execution
main() {
    log_info "Starting Claude Screenshot System Setup..."
    echo
    
    check_prerequisites
    setup_macos_screenshots
    setup_vault_integration
    setup_shell_aliases
    setup_launch_agent
    
    echo
    if verify_installation; then
        show_usage
    else
        log_error "Setup completed with errors. Please check the logs above."
        exit 1
    fi
}

# Show help
show_help() {
    cat << 'EOF'
Claude Screenshot System Master Setup

Usage: master_setup.sh [COMMAND]

Commands:
  install   Complete system setup (default)
  verify    Verify installation only
  help      Show this help message

This script sets up a complete screenshot automation system for macOS
that integrates with Claude Code and Obsidian vaults.
EOF
}

# Command handling
case "${1:-install}" in
    "install")
        main
        ;;
    "verify")
        verify_installation
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac