#!/bin/bash
# Deploy LangGraph Multi-Agent MCTS to Hugging Face Spaces
#
# Prerequisites:
# 1. Install Hugging Face CLI: pip install huggingface_hub
# 2. Login to Hugging Face: huggingface-cli login
# 3. Set HF_SPACE_NAME environment variable

set -e

# Configuration
HF_SPACE_NAME="${HF_SPACE_NAME:-langgraph-multi-agent-mcts}"
HF_USERNAME="${HF_USERNAME:-}"
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if huggingface-cli is installed
    if ! command -v huggingface-cli &> /dev/null; then
        log_error "huggingface-cli not found. Install with: pip install huggingface_hub"
        exit 1
    fi

    # Check if logged in
    if ! huggingface-cli whoami &> /dev/null; then
        log_error "Not logged in to Hugging Face. Run: huggingface-cli login"
        exit 1
    fi

    # Get username if not set
    if [ -z "$HF_USERNAME" ]; then
        HF_USERNAME=$(huggingface-cli whoami | head -n 1)
        log_info "Using Hugging Face username: $HF_USERNAME"
    fi

    log_info "Prerequisites check passed"
}

# Create or update the Space
create_or_update_space() {
    local space_repo="$HF_USERNAME/$HF_SPACE_NAME"
    local temp_dir=$(mktemp -d)

    log_info "Deploying to Hugging Face Space: $space_repo"

    # Clone or create the space
    if huggingface-cli repo info spaces/$space_repo &> /dev/null 2>&1; then
        log_info "Space exists, cloning..."
        cd "$temp_dir"
        git clone "https://huggingface.co/spaces/$space_repo" space
        cd space
    else
        log_info "Creating new Space..."
        huggingface-cli repo create "$HF_SPACE_NAME" --type space --space_sdk docker
        cd "$temp_dir"
        git clone "https://huggingface.co/spaces/$space_repo" space
        cd space
    fi

    # Copy necessary files
    log_info "Copying application files..."

    # Copy Hugging Face specific files
    cp "$REPO_ROOT/deploy/huggingface/README.md" ./README.md
    cp "$REPO_ROOT/deploy/huggingface/Dockerfile" ./Dockerfile
    cp "$REPO_ROOT/deploy/huggingface/requirements.txt" ./requirements.txt

    # Create deploy directory structure for Dockerfile context
    mkdir -p deploy/huggingface
    cp "$REPO_ROOT/deploy/huggingface/requirements.txt" ./deploy/huggingface/requirements.txt

    # Copy source code
    cp -r "$REPO_ROOT/src" ./src

    # Copy models (if they exist)
    if [ -d "$REPO_ROOT/models" ]; then
        cp -r "$REPO_ROOT/models" ./models
    else
        mkdir -p models
    fi

    # Copy main app
    cp "$REPO_ROOT/app.py" ./app.py

    # Configure git-lfs for large files
    log_info "Configuring git-lfs for large files..."
    git lfs install
    git lfs track "*.pt"
    git lfs track "*.bin"
    git lfs track "*.safetensors"
    git lfs track "models/**/*"

    # Commit and push
    log_info "Committing changes..."
    git add .
    git commit -m "Deploy LangGraph Multi-Agent MCTS to Hugging Face Spaces" || true

    log_info "Pushing to Hugging Face..."
    git push origin main

    # Cleanup
    cd /
    rm -rf "$temp_dir"

    log_info "Deployment complete!"
    log_info "Visit your Space at: https://huggingface.co/spaces/$space_repo"
}

# Main
main() {
    log_info "LangGraph Multi-Agent MCTS - Hugging Face Spaces Deployment"
    log_info "============================================================"

    check_prerequisites
    create_or_update_space
}

main "$@"
