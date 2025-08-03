#!/bin/bash

# A script to uninstall the EmbedServ application, its systemd service, and symlink.

# --- Configuration ---
PROJECT_NAME="embedserv"
INSTALL_DIR="$HOME/embedserv"
SERVICE_FILE="/etc/systemd/system/$PROJECT_NAME.service"
SYMLINK_PATH="$HOME/.local/bin/$PROJECT_NAME"

# --- Color Codes ---
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Uninstalling EmbedServ...${NC}"

# 1. Stop and Disable the systemd Service
if [ -f "$SERVICE_FILE" ]; then
    echo "Stopping and disabling systemd service..."
    sudo systemctl stop "$PROJECT_NAME.service" || true
    sudo systemctl disable "$PROJECT_NAME.service" || true
    echo "Removing service file..."
    sudo rm -f "$SERVICE_FILE"
    sudo systemctl daemon-reload
    echo "Service has been removed."
else
    echo "No systemd service file found. Skipping."
fi

# 2. Remove the Symlink
if [ -L "$SYMLINK_PATH" ]; then
    echo "Removing command-line shortcut (symlink)..."
    rm -f "$SYMLINK_PATH"
    echo "Symlink removed."
else
    echo "No symlink found. Skipping."
fi

# 3. Remove the Installation Directory
if [ -d "$INSTALL_DIR" ]; then
    read -p "Do you want to permanently delete all application files in '$INSTALL_DIR'? (y/N) " -n 1 -r; echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing installation directory: $INSTALL_DIR..."
        rm -rf "$INSTALL_DIR"
        echo "Directory removed."
    else
        echo "Skipping removal of application files."
    fi
else
    echo "No installation directory found at $INSTALL_DIR. Skipping."
fi

echo -e "${YELLOW}✅ Uninstallation complete.${NC}"