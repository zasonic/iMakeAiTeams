#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  START HERE — iMakeAiTeams (macOS)
#  Double-click this file in Finder.
#  First time: right-click → Open  (to bypass Gatekeeper).
# ═══════════════════════════════════════════════════════════════════

# Ensure we're running from the right directory
cd "$(dirname "$0")"
APP="$(pwd)/app"

# Fix permissions on self (zip extraction strips +x)
chmod +x "$0" 2>/dev/null

# ── Find Python 3 ─────────────────────────────────────────────────
PYEXE=""

if command -v python3 &>/dev/null; then
    PYEXE="python3"
elif [ -f "/usr/local/bin/python3" ]; then
    PYEXE="/usr/local/bin/python3"
elif [ -f "/opt/homebrew/bin/python3" ]; then
    PYEXE="/opt/homebrew/bin/python3"
fi

if [ -z "$PYEXE" ]; then
    osascript -e '
    set theChoice to button returned of (display dialog "Python 3 is not installed.\n\nClick \"Install\" to set up the Xcode command line tools (includes Python 3), or download from python.org." buttons {"Open python.org", "Install"} default button "Install" with title "iMakeAiTeams" with icon caution)
    if theChoice is "Install" then
        do shell script "xcode-select --install"
    else
        open location "https://www.python.org/downloads/"
    end if'
    echo ""
    echo "  After installing Python 3, double-click this file again."
    exit 1
fi

# ── Verify minimum version (3.10) ─────────────────────────────────
VERSION_OK=$($PYEXE -c "import sys; print(1 if sys.version_info >= (3,10) else 0)" 2>/dev/null)
if [ "$VERSION_OK" != "1" ]; then
    osascript -e 'display dialog "Python 3.10 or newer is required.\n\nPlease update from python.org." buttons {"Open python.org"} default button 1 with title "iMakeAiTeams" with icon caution'
    open "https://www.python.org/downloads/"
    exit 1
fi

# ── Launch setup (detached from terminal) ──────────────────────────
$PYEXE "$APP/setup_launcher.pyw" &
disown

# Close this terminal window after a short delay
sleep 1
osascript -e 'tell application "Terminal" to close front window' 2>/dev/null &
exit 0
