#!/bin/bash

# A script to install EmbedServ, set it up as a service, and ensure
# the CLI is easily accessible.

# Stop the script if any command fails
set -e

# --- Configuration ---
PROJECT_NAME="embedserv"
GITHUB_REPO="TPO-Code/TPO-embedserv"
# --- NEW: Specify the branch to install from. Defaults to 'main'. ---
# You can override this by passing a branch name as an argument to the script.
# e.g., ./reinstall.sh develop
BRANCH="master" # <-- NEW
INSTALL_DIR="$HOME/embedserv"
VENV_DIR="$INSTALL_DIR/venv"
SERVICE_FILE="/etc/systemd/system/$PROJECT_NAME.service"
SYMLINK_DIR="$HOME/.local/bin"
SYMLINK_PATH="$SYMLINK_DIR/$PROJECT_NAME"

# --- Color Codes for Output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Helper Functions ---
function print_success { echo -e "${GREEN}$1${NC}"; }
function print_warning { echo -e "${YELLOW}$1${NC}"; }
function print_error { echo -e "${RED}$1${NC}"; }

# --- Main Script Logic ---
echo "🚀 Starting EmbedServ Installation..."

# --- NEW: Allow overriding the branch from the command line ---
if [ -n "$1" ]; then
    BRANCH="$1"
    print_success "✅ Overriding default branch. Will install from: $BRANCH"
else
    print_warning "ℹ️ Using default branch: $BRANCH. To install a different branch, run: $0 <branch_name>"
fi

# Steps 1-3: Prerequisites, Venv, and Installation
print_warning "\nStep 1 & 2: Checking tools and setting up directories..."
command -v git >/dev/null 2>&1 || { print_error "Error: git is not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { print_error "Error: python3 is not installed."; exit 1; }
USE_UV=false
if command -v uv >/dev/null 2>&1; then USE_UV=true; print_success "✅ 'uv' detected."; else print_warning "ℹ️ 'uv' not found, using 'pip'."; fi
if [ -d "$INSTALL_DIR" ]; then read -p "Existing '$INSTALL_DIR' found. Remove and start fresh? (y/N) " -n 1 -r; echo; if [[ $REPLY =~ ^[Yy]$ ]]; then rm -rf "$INSTALL_DIR"; else print_error "Installation cancelled."; exit 1; fi; fi
mkdir -p "$INSTALL_DIR"
if [ "$USE_UV" = true ]; then uv venv "$VENV_DIR"; else python3 -m venv "$VENV_DIR"; fi
print_success "✅ Virtual environment created."

print_warning "\nStep 3: Installing EmbedServ package from branch '$BRANCH'..." # <-- MODIFIED
INSTALL_URL="git+https://github.com/$GITHUB_REPO.git@$BRANCH" # <-- NEW (for clarity)
if [ "$USE_UV" = true ]; then
    uv pip install --python "$VENV_DIR/bin/python" "$INSTALL_URL" # <-- MODIFIED
else
    "$VENV_DIR/bin/pip" install "$INSTALL_URL" # <-- MODIFIED
fi
EXEC_PATH="$VENV_DIR/bin/$PROJECT_NAME"
if [ ! -f "$EXEC_PATH" ]; then print_error "Installation failed. Check if branch '$BRANCH' exists."; exit 1; fi # <-- MODIFIED
print_success "✅ Package installed successfully."

# Step 4: Systemd Service Setup
print_warning "\nStep 4: Setting up the systemd background service..."
read -p "Create a systemd service to run EmbedServ on boot? (Requires sudo) (y/N) " -n 1 -r; echo
SERVICE_INSTALLED=false
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CURRENT_USER=$(whoami)
    SERVICE_CONTENT="[Unit]\nDescription=EmbedServ\nAfter=network.target\n\n[Service]\nUser=$CURRENT_USER\nGroup=$CURRENT_USER\nWorkingDirectory=$INSTALL_DIR\nExecStart=$EXEC_PATH serve\nRestart=always\nRestartSec=3\n\n[Install]\nWantedBy=multi-user.target"
    echo "Creating systemd service file..."
    echo -e "$SERVICE_CONTENT" | sudo tee "$SERVICE_FILE" > /dev/null
    sudo systemctl daemon-reload && sudo systemctl enable "$PROJECT_NAME.service" && sudo systemctl start "$PROJECT_NAME.service"
    print_success "✅ Service created and started."
    SERVICE_INSTALLED=true
fi

# Step 5: Create Symlink for CLI access
print_warning "\nStep 5: Creating command-line shortcut..."
mkdir -p "$SYMLINK_DIR"
ln -sf "$EXEC_PATH" "$SYMLINK_PATH"
print_success "✅ Command shortcut created."

# --- Final Summary and Post-Install Check ---
print_success "\n🎉🎉🎉 EmbedServ has been successfully installed! 🎉🎉🎉"

if [ "$SERVICE_INSTALLED" = true ]; then
    echo
    echo "The service is running in the background. You can check its status with:"
    print_warning "  sudo systemctl status $PROJECT_NAME"
fi

# --- Step 6: Check if the symlink is in the user's PATH ---
print_warning "\nStep 6: Checking shell configuration..."
case ":$PATH:" in
  *":$SYMLINK_DIR:"*)
    print_success "✅ Your PATH is correctly configured. You can use the 'embedserv' command directly."
    echo "Try it: embedserv --help"
    ;;
  *)
    print_error "‼️ Action Required: Your shell PATH needs to be updated."
    echo "To use the 'embedserv' command directly, please do the following:"
    echo
    echo "1. Add this line to your shell's configuration file:"
    if [ -n "$ZSH_VERSION" ]; then
        print_warning "   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
        echo
        echo "2. Then, apply the changes by running:"
        print_warning "   source ~/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        print_warning "   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
        echo
        echo "2. Then, apply the changes by running:"
        print_warning "   source ~/.bashrc"
    else
        echo "   Add '$SYMLINK_DIR' to your PATH environment variable."
    fi
    echo
    echo "You only need to do this once. A new terminal will also have the correct PATH."
    ;;
esac

echo
echo "To change the server port (default: 11536), use:"
print_warning "  embedserv config set port 8000"
echo "Then, if you installed the service, restart it: sudo systemctl restart $PROJECT_NAME"