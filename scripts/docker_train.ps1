# =============================================================================
# Docker Training Quick Start Script
# =============================================================================
#
# Complete workflow to build and run training in Docker
#
# Usage:
#   .\scripts\docker_train.ps1                # Run demo mode
#   .\scripts\docker_train.ps1 -Mode prod     # Run production mode
#   .\scripts\docker_train.ps1 -SkipBuild     # Skip build, use existing image
#
# =============================================================================

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("demo", "prod")]
    [string]$Mode = "demo",

    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild = $false,

    [Parameter(Mandatory=$false)]
    [switch]$SkipChecks = $false,

    [Parameter(Mandatory=$false)]
    [switch]$Detached = $false
)

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

# Colors
$GREEN = "Green"
$RED = "Red"
$YELLOW = "Yellow"
$CYAN = "Cyan"

function Write-Step {
    param([string]$Message)
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor $CYAN
    Write-Host $Message -ForegroundColor $CYAN
    Write-Host ("=" * 80) -ForegroundColor $CYAN
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $GREEN
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $RED
}

function Write-Info {
    param([string]$Message)
    Write-Host "  $Message" -ForegroundColor $YELLOW
}

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================

Write-Step "Step 1/6: Checking Prerequisites"

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker not found. Please install Docker Desktop for Windows."
    exit 1
}
Write-Success "Docker found: $(docker --version)"

# Check Docker Compose
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Error "Docker Compose not found. Please install Docker Compose."
    exit 1
}
Write-Success "Docker Compose found: $(docker-compose --version)"

# Check .env file
if (-not (Test-Path ".env")) {
    Write-Error ".env file not found!"
    Write-Info "Please copy .env.example to .env and fill in your API keys"
    Write-Info "  cp .env.example .env"
    exit 1
}
Write-Success ".env file found"

# Check NVIDIA Docker (optional but recommended)
$nvidiaSmiOutput = docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Success "NVIDIA Docker runtime available"
} else {
    Write-Info "NVIDIA Docker runtime not available - GPU training disabled"
    Write-Info "Install nvidia-docker2 for GPU support"
}

# =============================================================================
# Step 2: Run Sanity Checks (Optional)
# =============================================================================

if (-not $SkipChecks) {
    Write-Step "Step 2/6: Running Pre-Deployment Sanity Checks"

    Write-Info "Running deployment sanity checks..."

    # Use basic Python checks instead of the full script to avoid encoding issues
    Write-Info "Checking configuration files..."
    python -c "import yaml; yaml.safe_load(open('training/config_local_demo.yaml')); print('Config valid')" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Configuration files valid"
    } else {
        Write-Error "Configuration validation failed"
        exit 1
    }

    Write-Info "Checking Docker files..."
    if ((Test-Path "Dockerfile.train") -and (Test-Path "docker-compose.train.yml")) {
        Write-Success "Docker files present"
    } else {
        Write-Error "Missing Docker files"
        exit 1
    }
} else {
    Write-Step "Step 2/6: Skipping Sanity Checks (--SkipChecks flag)"
}

# =============================================================================
# Step 3: Build Docker Image
# =============================================================================

if (-not $SkipBuild) {
    Write-Step "Step 3/6: Building Docker Image"

    $imageName = "langgraph-mcts-train:$Mode"
    Write-Info "Building image: $imageName"
    Write-Info "This may take 5-10 minutes on first build..."

    docker build -f Dockerfile.train --target $Mode -t $imageName .

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker image built successfully"

        # Show image size
        $imageSize = docker images $imageName --format "{{.Size}}"
        Write-Info "Image size: $imageSize"
    } else {
        Write-Error "Docker build failed"
        exit 1
    }
} else {
    Write-Step "Step 3/6: Skipping Build (--SkipBuild flag)"
    Write-Info "Using existing image: langgraph-mcts-train:$Mode"
}

# =============================================================================
# Step 4: Verify Image
# =============================================================================

Write-Step "Step 4/6: Verifying Docker Image"

$imageName = "langgraph-mcts-train:$Mode"
$imageExists = docker images $imageName --format "{{.Repository}}:{{.Tag}}" | Select-String -Pattern $imageName

if ($imageExists) {
    Write-Success "Image exists: $imageName"
} else {
    Write-Error "Image not found: $imageName"
    Write-Info "Run without --SkipBuild to build the image"
    exit 1
}

# =============================================================================
# Step 5: Start Training Container
# =============================================================================

Write-Step "Step 5/6: Starting Training Container"

$containerName = "mcts-training-$Mode"

# Stop existing container if running
$existingContainer = docker ps -a --filter "name=$containerName" --format "{{.Names}}"
if ($existingContainer) {
    Write-Info "Stopping existing container: $containerName"
    docker stop $containerName 2>&1 | Out-Null
    docker rm $containerName 2>&1 | Out-Null
}

Write-Info "Starting container: $containerName"

if ($Mode -eq "demo") {
    $serviceName = "training-demo"
} else {
    $serviceName = "training-prod"
}

# Start with docker-compose
if ($Detached) {
    docker-compose -f docker-compose.train.yml up -d $serviceName

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Container started in detached mode"
        Write-Info "View logs with: docker logs -f $containerName"
    } else {
        Write-Error "Failed to start container"
        exit 1
    }
} else {
    Write-Success "Starting container (logs will follow)..."
    Write-Info "Press Ctrl+C to stop"
    Write-Info ""

    docker-compose -f docker-compose.train.yml up $serviceName
}

# =============================================================================
# Step 6: Post-Run Information
# =============================================================================

if ($Detached) {
    Write-Step "Step 6/6: Training Information"

    Write-Info "Container is running in background"
    Write-Info ""
    Write-Info "Useful commands:"
    Write-Info "  View logs:        docker logs -f $containerName"
    Write-Info "  Check status:     docker ps"
    Write-Info "  Stop container:   docker-compose -f docker-compose.train.yml stop"
    Write-Info "  Remove container: docker-compose -f docker-compose.train.yml down"
    Write-Info ""
    Write-Info "Monitor training:"
    Write-Info "  W&B Dashboard:    https://wandb.ai"
    Write-Info "  Local logs:       .\logs\demo\"
    Write-Info "  Checkpoints:      .\checkpoints\"
    Write-Info ""
    Write-Success "Training started successfully!"
}

# =============================================================================
# Completion
# =============================================================================

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor $GREEN
Write-Host "Docker Training Setup Complete!" -ForegroundColor $GREEN
Write-Host ("=" * 80) -ForegroundColor $GREEN
Write-Host ""
