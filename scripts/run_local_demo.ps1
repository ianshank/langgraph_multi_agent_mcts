# ============================================================================
# Local Demo Training Pipeline Runner (PowerShell)
# ============================================================================
#
# This script runs the complete local verification training demo on a 16GB GPU.
# It performs the following:
#   1. Environment validation (Python, CUDA, packages)
#   2. External service verification (Pinecone, W&B, GitHub)
#   3. Training pipeline execution
#   4. Results reporting
#
# Usage:
#   .\scripts\run_local_demo.ps1 [-SkipVerification] [-Verbose]
#
# Requirements:
#   - Windows 10/11
#   - PowerShell 5.1+
#   - Python 3.10+
#   - NVIDIA GPU with CUDA
#   - Environment variables set (see below)
#
# ============================================================================

param(
    [Parameter(Mandatory=$false)]
    [switch]$SkipVerification = $false,

    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false,

    [Parameter(Mandatory=$false)]
    [switch]$CleanArtifacts = $false,

    [Parameter(Mandatory=$false)]
    [string]$LogFile = "logs\demo\run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
)

# ============================================================================
# Configuration
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

# Colors for output
$COLOR_GREEN = "Green"
$COLOR_RED = "Red"
$COLOR_YELLOW = "Yellow"
$COLOR_CYAN = "Cyan"
$COLOR_WHITE = "White"

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor $COLOR_CYAN
    Write-Host $Message -ForegroundColor $COLOR_CYAN
    Write-Host ("=" * 80) -ForegroundColor $COLOR_CYAN
    Write-Host ""
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host $Message -ForegroundColor $COLOR_YELLOW
    Write-Host ("-" * 80) -ForegroundColor $COLOR_YELLOW
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $COLOR_GREEN
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $COLOR_RED
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $COLOR_YELLOW
}

function Write-Info {
    param([string]$Message)
    Write-Host "  $Message" -ForegroundColor $COLOR_WHITE
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Get-ElapsedTime {
    param([DateTime]$StartTime)
    $elapsed = (Get-Date) - $StartTime
    return "{0:D2}m {1:D2}s" -f $elapsed.Minutes, $elapsed.Seconds
}

# ============================================================================
# Environment Validation
# ============================================================================

function Test-Environment {
    Write-Section "Step 1/5: Validating Environment"

    # Check Python
    if (-not (Test-Command "python")) {
        Write-Error "Python not found in PATH"
        Write-Info "Please install Python 3.10+ from https://www.python.org/"
        return $false
    }

    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"

    # Check Python version
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]

        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Error "Python 3.10+ required (found $pythonVersion)"
            return $false
        }
    }

    # Check if in virtual environment
    if ($env:VIRTUAL_ENV) {
        Write-Success "Virtual environment active: $env:VIRTUAL_ENV"
    } else {
        Write-Warning "No virtual environment detected"
        Write-Info "Consider activating a virtual environment first"
    }

    # Check Git
    if (Test-Command "git") {
        $gitVersion = git --version
        Write-Success "Git found: $gitVersion"
    } else {
        Write-Warning "Git not found (optional for development)"
    }

    # Check CUDA availability
    if (Test-Command "nvidia-smi") {
        Write-Success "NVIDIA GPU detected"
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($gpuInfo) {
            Write-Info "GPU: $gpuInfo"
        }
    } else {
        Write-Error "nvidia-smi not found - CUDA not available"
        Write-Info "Please install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
        return $false
    }

    return $true
}

# ============================================================================
# Package Installation
# ============================================================================

function Install-Requirements {
    Write-Section "Step 2/5: Installing/Verifying Python Packages"

    $requirementsFile = Join-Path $PROJECT_ROOT "requirements.txt"

    if (-not (Test-Path $requirementsFile)) {
        Write-Error "requirements.txt not found"
        return $false
    }

    Write-Info "Installing packages from requirements.txt..."

    try {
        # Use pip with upgrade flag
        python -m pip install --upgrade pip | Out-Null

        if ($Verbose) {
            python -m pip install -r $requirementsFile
        } else {
            python -m pip install -r $requirementsFile --quiet
        }

        Write-Success "Python packages installed successfully"
        return $true
    }
    catch {
        Write-Error "Failed to install packages: $_"
        return $false
    }
}

# ============================================================================
# Environment Variables Check
# ============================================================================

function Test-EnvironmentVariables {
    Write-Section "Step 3/5: Checking Environment Variables"

    $requiredVars = @(
        @{Name="PINECONE_API_KEY"; Required=$true; Description="Vector database"},
        @{Name="WANDB_API_KEY"; Required=$true; Description="Experiment tracking"},
        @{Name="GITHUB_TOKEN"; Required=$true; Description="Repository access"}
    )

    $optionalVars = @(
        @{Name="OPENAI_API_KEY"; Required=$false; Description="OpenAI API (optional)"},
        @{Name="NEO4J_PASSWORD"; Required=$false; Description="Knowledge graph (optional)"}
    )

    $allVarsValid = $true

    # Check required variables
    Write-Info "Required environment variables:"
    foreach ($var in $requiredVars) {
        $value = [System.Environment]::GetEnvironmentVariable($var.Name)

        if ($value) {
            $maskedValue = $value.Substring(0, [Math]::Min(8, $value.Length)) + "***"
            Write-Success "$($var.Name) is set ($maskedValue) - $($var.Description)"
        } else {
            Write-Error "$($var.Name) is NOT set - $($var.Description)"
            $allVarsValid = $false
        }
    }

    # Check optional variables
    Write-Info ""
    Write-Info "Optional environment variables:"
    foreach ($var in $optionalVars) {
        $value = [System.Environment]::GetEnvironmentVariable($var.Name)

        if ($value) {
            $maskedValue = $value.Substring(0, [Math]::Min(8, $value.Length)) + "***"
            Write-Success "$($var.Name) is set ($maskedValue) - $($var.Description)"
        } else {
            Write-Warning "$($var.Name) is NOT set - $($var.Description)"
        }
    }

    if (-not $allVarsValid) {
        Write-Info ""
        Write-Info "Please set missing environment variables:"
        Write-Info "  PowerShell: `$env:VARIABLE_NAME = 'your-api-key'"
        Write-Info "  CMD: set VARIABLE_NAME=your-api-key"
        Write-Info "  Or create a .env file in the project root"
        return $false
    }

    return $true
}

# ============================================================================
# External Services Verification
# ============================================================================

function Test-ExternalServices {
    Write-Section "Step 4/5: Verifying External Services"

    if ($SkipVerification) {
        Write-Warning "Skipping service verification (--SkipVerification flag set)"
        return $true
    }

    $verifyScript = Join-Path $PROJECT_ROOT "scripts\verify_external_services.py"

    if (-not (Test-Path $verifyScript)) {
        Write-Warning "Verification script not found, skipping..."
        return $true
    }

    try {
        Write-Info "Running service verification..."

        if ($Verbose) {
            python $verifyScript --config "training\config_local_demo.yaml" --verbose
        } else {
            python $verifyScript --config "training\config_local_demo.yaml"
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Success "All critical services verified successfully"
            return $true
        } else {
            Write-Error "Service verification failed"
            return $false
        }
    }
    catch {
        Write-Error "Service verification error: $_"
        return $false
    }
}

# ============================================================================
# Demo Training Execution
# ============================================================================

function Start-DemoTraining {
    Write-Section "Step 5/5: Starting Demo Training Pipeline"

    $startTime = Get-Date

    Write-Info "Configuration: training\config_local_demo.yaml"
    Write-Info "Expected duration: 30-45 minutes"
    Write-Info "Start time: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host ""

    try {
        # Build command
        $command = "python -m training.cli train --demo"

        if ($SkipVerification) {
            $command += " --skip-verification"
        }

        if ($Verbose) {
            $command += " --log-level DEBUG"
        }

        # Execute training
        Write-Info "Executing: $command"
        Write-Host ""

        Invoke-Expression $command

        if ($LASTEXITCODE -eq 0) {
            $elapsed = Get-ElapsedTime -StartTime $startTime
            Write-Success "Training completed successfully in $elapsed"
            return $true
        } else {
            Write-Error "Training failed with exit code $LASTEXITCODE"
            return $false
        }
    }
    catch {
        Write-Error "Training execution error: $_"
        return $false
    }
}

# ============================================================================
# Artifact Cleanup
# ============================================================================

function Remove-DemoArtifacts {
    Write-Section "Cleaning up demo artifacts..."

    $artifactPaths = @(
        "checkpoints\demo",
        "logs\demo",
        "cache\dabstep",
        "cache\primus_seed",
        "cache\primus_instruct",
        "cache\embeddings",
        "cache\rag_index_demo"
    )

    foreach ($path in $artifactPaths) {
        $fullPath = Join-Path $PROJECT_ROOT $path

        if (Test-Path $fullPath) {
            try {
                Remove-Item -Path $fullPath -Recurse -Force
                Write-Success "Removed: $path"
            }
            catch {
                Write-Warning "Failed to remove: $path - $_"
            }
        }
    }

    Write-Success "Cleanup complete"
}

# ============================================================================
# Results Summary
# ============================================================================

function Show-Results {
    param([DateTime]$StartTime)

    Write-Header "Demo Execution Summary"

    $elapsed = Get-ElapsedTime -StartTime $StartTime
    Write-Info "Total execution time: $elapsed"

    # Check for artifacts
    $checkpointsPath = Join-Path $PROJECT_ROOT "checkpoints\demo"
    $logsPath = Join-Path $PROJECT_ROOT "logs\demo"

    Write-Host ""
    Write-Info "Artifacts:"

    if (Test-Path $checkpointsPath) {
        $checkpoints = Get-ChildItem -Path $checkpointsPath -File -ErrorAction SilentlyContinue
        Write-Success "Checkpoints: $($checkpoints.Count) files in $checkpointsPath"
    } else {
        Write-Warning "No checkpoints found"
    }

    if (Test-Path $logsPath) {
        $logs = Get-ChildItem -Path $logsPath -File -ErrorAction SilentlyContinue
        Write-Success "Logs: $($logs.Count) files in $logsPath"
    } else {
        Write-Warning "No logs found"
    }

    Write-Host ""
    Write-Info "Next Steps:"
    Write-Info "  1. View training metrics at https://wandb.ai"
    Write-Info "  2. Review logs in $logsPath"
    Write-Info "  3. Check checkpoints in $checkpointsPath"
    Write-Info "  4. Scale to full training: python -m training.cli train"

    Write-Host ""
}

# ============================================================================
# Main Execution
# ============================================================================

function Main {
    $scriptStart = Get-Date

    Write-Header "Multi-Agent MCTS Local Demo Training Pipeline"

    Write-Info "Project root: $PROJECT_ROOT"
    Write-Info "Log file: $LogFile"

    if ($Verbose) {
        Write-Info "Verbose mode: ENABLED"
    }

    if ($SkipVerification) {
        Write-Warning "Service verification: DISABLED"
    }

    # Create log directory
    $logDir = Split-Path -Parent $LogFile
    if ($logDir -and -not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    # Step 1: Validate environment
    if (-not (Test-Environment)) {
        Write-Error "Environment validation failed"
        exit 1
    }

    # Step 2: Install packages
    if (-not (Install-Requirements)) {
        Write-Error "Package installation failed"
        exit 1
    }

    # Step 3: Check environment variables
    if (-not (Test-EnvironmentVariables)) {
        Write-Error "Environment variables check failed"
        exit 1
    }

    # Step 4: Verify external services
    if (-not (Test-ExternalServices)) {
        Write-Error "External services verification failed"
        Write-Info "Use -SkipVerification to bypass this check (not recommended)"
        exit 1
    }

    # Step 5: Run training
    if (-not (Start-DemoTraining)) {
        Write-Error "Training pipeline failed"
        exit 1
    }

    # Cleanup if requested
    if ($CleanArtifacts) {
        Remove-DemoArtifacts
    }

    # Show results
    Show-Results -StartTime $scriptStart

    Write-Header "Demo Completed Successfully!"

    exit 0
}

# ============================================================================
# Script Entry Point
# ============================================================================

try {
    Main
}
catch {
    Write-Error "Unexpected error: $_"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
