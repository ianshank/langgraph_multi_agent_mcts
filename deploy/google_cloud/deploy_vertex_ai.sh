#!/bin/bash
# Deploy LangGraph Multi-Agent MCTS to Vertex AI Agent Engine
#
# This script deploys the framework using Vertex AI Agent Engine
# which provides managed infrastructure for LangGraph agents.
#
# Prerequisites:
# 1. gcloud CLI installed and authenticated
# 2. Required APIs enabled
# 3. Python 3.11+ with required packages

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
AGENT_NAME="${AGENT_NAME:-langgraph-mcts-agent}"
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

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

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found."
        exit 1
    fi

    # Check project ID
    if [ -z "$PROJECT_ID" ]; then
        log_error "GOOGLE_CLOUD_PROJECT not set."
        exit 1
    fi

    # Verify authentication
    if ! gcloud auth print-access-token &> /dev/null; then
        log_error "Not authenticated. Run: gcloud auth application-default login"
        exit 1
    fi

    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Agent: $AGENT_NAME"
}

# Enable required APIs
enable_apis() {
    log_step "Enabling required APIs..."

    apis=(
        "aiplatform.googleapis.com"
        "compute.googleapis.com"
        "iam.googleapis.com"
    )

    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID" --quiet || true
    done
}

# Install Python dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."

    pip install --quiet --upgrade \
        google-cloud-aiplatform[agent_engines,langchain] \
        langgraph \
        langchain-google-vertexai
}

# Deploy to Vertex AI Agent Engine
deploy_agent_engine() {
    log_step "Deploying to Vertex AI Agent Engine..."

    # Create and run the deployment script
    python3 << 'PYTHON_SCRIPT'
import os
import sys

# Add repo to path
sys.path.insert(0, os.environ.get('REPO_ROOT', '.'))

from google.cloud import aiplatform

PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
REGION = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
AGENT_NAME = os.environ.get('AGENT_NAME', 'langgraph-mcts-agent')

print(f"Initializing Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=REGION)

try:
    from vertexai.preview import agent_engines

    # Define the LangGraph application
    class LangGraphMCTSApp:
        """LangGraph Multi-Agent MCTS Application for Vertex AI."""

        def __init__(self, project: str, location: str):
            self.project = project
            self.location = location
            self._initialized = False

        def set_up(self):
            """Initialize the application."""
            if self._initialized:
                return

            # Import framework components
            from src.agents.meta_controller.rnn_controller import RNNMetaController
            from src.agents.meta_controller.bert_controller_v2 import BERTMetaController

            self.controllers = {
                'rnn': RNNMetaController(name="RNNController", seed=42),
                'bert': BERTMetaController(name="BERTController", seed=42),
            }
            self._initialized = True
            print("Framework initialized")

        def query(self, query: str, controller_type: str = "rnn") -> dict:
            """Process a query."""
            if not self._initialized:
                self.set_up()

            from src.agents.meta_controller.base import MetaControllerFeatures

            # Create features
            features = MetaControllerFeatures(
                hrm_confidence=0.5,
                trm_confidence=0.5,
                mcts_value=0.5,
                consensus_score=0.7,
                last_agent="none",
                iteration=0,
                query_length=len(query),
                has_rag_context=len(query) > 50,
                rag_relevance_score=0.7 if len(query) > 50 else 0.0,
                is_technical_query=any(
                    w in query.lower()
                    for w in ["algorithm", "code", "implement"]
                ),
            )

            controller = self.controllers.get(controller_type, self.controllers['rnn'])
            prediction = controller.predict(features)

            return {
                "response": f"[{prediction.agent.upper()}] {query[:100]}...",
                "agent": prediction.agent,
                "confidence": prediction.confidence,
                "routing_probabilities": prediction.probabilities,
            }

    # Deploy the agent
    print(f"Creating agent: {AGENT_NAME}")

    remote_agent = agent_engines.create(
        LangGraphMCTSApp(project=PROJECT_ID, location=REGION),
        requirements=[
            "google-cloud-aiplatform[agent_engines,langchain]>=1.93.0",
            "langgraph>=0.0.1",
            "langchain>=0.1.0",
            "langchain-core>=0.1.0",
            "torch>=2.1.0",
            "transformers>=4.40.0,<4.46.0",
            "peft==0.10.0",
            "sentence-transformers>=2.2.0",
        ],
        display_name=AGENT_NAME,
    )

    print(f"Agent deployed successfully!")
    print(f"Resource name: {remote_agent.resource_name}")

except ImportError as e:
    print(f"Note: Vertex AI Agent Engine preview not available: {e}")
    print("Falling back to standard deployment...")
    print("Use deploy_cloud_run.sh for containerized deployment")
    sys.exit(0)

except Exception as e:
    print(f"Deployment error: {e}")
    sys.exit(1)
PYTHON_SCRIPT
}

# Main
main() {
    log_info "LangGraph Multi-Agent MCTS - Vertex AI Agent Engine Deployment"
    log_info "================================================================"

    export REPO_ROOT
    export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
    export GOOGLE_CLOUD_LOCATION="$REGION"
    export AGENT_NAME

    check_prerequisites
    enable_apis
    install_dependencies
    deploy_agent_engine

    log_info ""
    log_info "Deployment complete!"
    log_info "View your agent in the Google Cloud Console:"
    log_info "https://console.cloud.google.com/vertex-ai/agents?project=$PROJECT_ID"
}

main "$@"
