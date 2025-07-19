#!/bin/bash

# Screenshot Automation Setup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREENSHOT_MANAGER="$SCRIPT_DIR/screenshot_manager.sh"

echo "Setting up screenshot automation..."

# Install fswatch if not present
if ! command -v fswatch >/dev/null 2>&1; then
    echo "Installing fswatch for file monitoring..."
    if command -v brew >/dev/null 2>&1; then
        brew install fswatch
    else
        echo "Homebrew not found. Please install fswatch manually:"
        echo "brew install fswatch"
        exit 1
    fi
fi

# Add shell alias
SHELL_RC=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
    if ! grep -q "screenshot_manager" "$SHELL_RC"; then
        echo "" >> "$SHELL_RC"
        echo "# Screenshot automation" >> "$SHELL_RC"
        echo "alias screenshot-monitor='$SCREENSHOT_MANAGER monitor'" >> "$SHELL_RC"
        echo "alias screenshot-process='$SCREENSHOT_MANAGER process'" >> "$SHELL_RC"
        echo "" >> "$SHELL_RC"
        echo "Added aliases to $SHELL_RC"
    else
        echo "Aliases already exist in $SHELL_RC"
    fi
fi

# Create LaunchAgent for automatic startup (optional)
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCH_AGENT_FILE="$LAUNCH_AGENT_DIR/com.user.screenshot-monitor.plist"

mkdir -p "$LAUNCH_AGENT_DIR"

cat > "$LAUNCH_AGENT_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.screenshot-monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCREENSHOT_MANAGER</string>
        <string>monitor</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/screenshot-monitor.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/screenshot-monitor.error.log</string>
</dict>
</plist>
EOF

echo "Setup complete!"
echo ""
echo "Usage:"
echo "  Manual processing: screenshot-process"
echo "  Start monitoring: screenshot-monitor"
echo "  Or run directly: $SCREENSHOT_MANAGER"
echo ""
echo "To enable automatic startup:"
echo "  launchctl load $LAUNCH_AGENT_FILE"
echo ""
echo "To disable automatic startup:"
echo "  launchctl unload $LAUNCH_AGENT_FILE"