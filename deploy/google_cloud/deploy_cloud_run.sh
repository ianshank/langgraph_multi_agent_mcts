#!/bin/bash
# Deploy LangGraph Multi-Agent MCTS to Google Cloud Run
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Required APIs enabled (see README.md)
# 3. Environment variables set

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-langgraph-mcts}"
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Image configuration
AR_REPO="${AR_REPO:-langgraph-mcts}"
IMAGE_NAME="${IMAGE_NAME:-langgraph-mcts-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Cloud Run configuration
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
TIMEOUT="${TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."

    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check project ID
    if [ -z "$PROJECT_ID" ]; then
        log_error "GOOGLE_CLOUD_PROJECT not set. Export it or set it in gcloud config."
        exit 1
    fi

    # Verify authentication
    if ! gcloud auth print-access-token &> /dev/null; then
        log_error "Not authenticated. Run: gcloud auth login"
        exit 1
    fi

    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Service: $SERVICE_NAME"
}

# Enable required APIs
enable_apis() {
    log_step "Enabling required APIs..."

    apis=(
        "run.googleapis.com"
        "artifactregistry.googleapis.com"
        "cloudbuild.googleapis.com"
        "aiplatform.googleapis.com"
    )

    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID" --quiet || true
    done
}

# Create Artifact Registry repository
create_artifact_registry() {
    log_step "Creating Artifact Registry repository..."

    if ! gcloud artifacts repositories describe "$AR_REPO" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        gcloud artifacts repositories create "$AR_REPO" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --description="LangGraph MCTS container images"
        log_info "Created Artifact Registry repository: $AR_REPO"
    else
        log_info "Artifact Registry repository already exists: $AR_REPO"
    fi
}

# Build and push container image
build_and_push_image() {
    log_step "Building and pushing container image..."

    local image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

    # Configure docker for Artifact Registry
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

    # Build with Cloud Build
    log_info "Building image with Cloud Build..."
    cd "$REPO_ROOT"

    gcloud builds submit \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --tag="$image_uri" \
        --timeout=1800s \
        -f deploy/google_cloud/Dockerfile \
        .

    log_info "Image pushed: $image_uri"
    echo "$image_uri"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    local image_uri="$1"

    log_step "Deploying to Cloud Run..."

    gcloud run deploy "$SERVICE_NAME" \
        --image="$image_uri" \
        --platform=managed \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --memory="$MEMORY" \
        --cpu="$CPU" \
        --min-instances="$MIN_INSTANCES" \
        --max-instances="$MAX_INSTANCES" \
        --timeout="${TIMEOUT}s" \
        --port=8080 \
        --allow-unauthenticated \
        --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},ADK_BACKEND=vertex_ai" \
        --quiet

    # Get service URL
    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")

    log_info "Deployment complete!"
    log_info "Service URL: $service_url"
    log_info "Health check: ${service_url}/health"
    log_info "API docs: ${service_url}/docs"
}

# Run smoke test
smoke_test() {
    local service_url="$1"

    log_step "Running smoke test..."

    if curl -s "${service_url}/health" | grep -q "healthy"; then
        log_info "Health check passed!"
    else
        log_warn "Health check did not return expected response"
    fi
}

# Main
main() {
    log_info "LangGraph Multi-Agent MCTS - Cloud Run Deployment"
    log_info "=================================================="

    check_prerequisites
    enable_apis
    create_artifact_registry

    local image_uri=$(build_and_push_image)
    deploy_cloud_run "$image_uri"

    # Get final URL and run smoke test
    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.url)")

    smoke_test "$service_url"

    log_info ""
    log_info "Deployment Summary"
    log_info "=================="
    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Service: $SERVICE_NAME"
    log_info "URL: $service_url"
}

main "$@"
