# Deployment Guide

This guide covers deploying the LangGraph Multi-Agent MCTS framework to various platforms.

## Overview

The framework supports deployment to:

1. **Hugging Face Spaces** - Free/low-cost hosting with Gradio UI
2. **Google Cloud Run** - Scalable containerized deployment
3. **Vertex AI Agent Engine** - Managed LangGraph agent deployment
4. **Docker** - Self-hosted containerized deployment

## Quick Start

### Hugging Face Spaces (Easiest)

Deploy the interactive demo to Hugging Face Spaces:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Set your space name
export HF_SPACE_NAME="my-langgraph-mcts"

# Deploy
bash deploy/huggingface/deploy.sh
```

Your Space will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/my-langgraph-mcts`

### Google Cloud Run (Production)

Deploy the REST API to Cloud Run:

```bash
# Set your project
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Deploy
bash deploy/google_cloud/deploy_cloud_run.sh
```

---

## Hugging Face Spaces

### Prerequisites

- Hugging Face account
- `huggingface_hub` Python package
- Git LFS (for large model files)

### Configuration

The deployment uses Docker SDK for maximum flexibility. Key files:

| File | Purpose |
|------|---------|
| `deploy/huggingface/README.md` | Space configuration (YAML header) |
| `deploy/huggingface/Dockerfile` | Container build instructions |
| `deploy/huggingface/requirements.txt` | Python dependencies |

### Secrets Configuration

Configure secrets in your Space settings:

1. Go to your Space's Settings page
2. Navigate to "Variables and Secrets"
3. Add required secrets:

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENAI_API_KEY` | Optional | For OpenAI LLM features |
| `ANTHROPIC_API_KEY` | Optional | For Claude LLM features |
| `WANDB_API_KEY` | Optional | Experiment tracking |
| `PINECONE_API_KEY` | Optional | RAG features |

### Hardware Options

| Tier | Specs | Cost | Best For |
|------|-------|------|----------|
| CPU Basic | 2 vCPU, 16GB RAM | Free | Demo/testing |
| CPU Upgrade | 4 vCPU, 32GB RAM | $0.06/hr | Light production |
| GPU | T4/A10G/A100 | $0.50+/hr | Training/inference |

### Manual Deployment

If the automated script doesn't work:

1. Create a new Space on Hugging Face with Docker SDK
2. Clone the Space repository
3. Copy required files:
   ```bash
   cp deploy/huggingface/* your-space/
   cp -r src your-space/
   cp -r models your-space/
   cp app.py your-space/
   ```
4. Push to the Space repository

---

## Google Cloud Platform

### Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- Required APIs enabled

### Enable Required APIs

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com
```

### Authentication

#### For Development

```bash
# Login with your user account
gcloud auth login
gcloud auth application-default login
```

#### For Production (Service Account)

1. Create a service account:
   ```bash
   gcloud iam service-accounts create langgraph-mcts-sa \
       --display-name="LangGraph MCTS Service Account"
   ```

2. Grant required roles:
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT_ID \
       --member="serviceAccount:langgraph-mcts-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
       --role="roles/aiplatform.user"
   ```

3. Create and download key:
   ```bash
   gcloud iam service-accounts keys create key.json \
       --iam-account=langgraph-mcts-sa@${PROJECT_ID}.iam.gserviceaccount.com
   ```

### Cloud Run Deployment

Cloud Run provides serverless container deployment with automatic scaling.

```bash
# Configuration
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export SERVICE_NAME="langgraph-mcts"

# Deploy
bash deploy/google_cloud/deploy_cloud_run.sh
```

#### Configuration Options

Edit `deploy/google_cloud/config.yaml`:

```yaml
cloud_run:
  container:
    memory: 4Gi      # Memory limit
    cpu: 2           # CPU cores
  scaling:
    min_instances: 0  # Scale to zero
    max_instances: 10 # Max instances
```

#### Accessing the Service

After deployment:

```bash
# Get service URL
gcloud run services describe langgraph-mcts \
    --region=us-central1 \
    --format="value(status.url)"
```

### Vertex AI Agent Engine

For production LangGraph agent deployment with managed infrastructure.

```bash
# Deploy to Agent Engine
bash deploy/google_cloud/deploy_vertex_ai.sh
```

Agent Engine provides:
- Managed runtime for LangGraph agents
- Automatic scaling
- Built-in authentication
- VPC-SC compliance

### Using Google AI Studio (Development)

For quick prototyping with the Gemini API:

```python
from src.integrations.google_adk.ai_studio_client import AIStudioClient, AIStudioConfig

# Using API key (development)
config = AIStudioConfig.from_env()  # Reads GOOGLE_API_KEY
client = AIStudioClient(config)

response = await client.generate("What is MCTS?")
```

Get an API key from: https://aistudio.google.com/apikey

---

## Environment Variables

### Common Variables

```bash
# LLM Provider
LLM_PROVIDER=openai  # openai, anthropic, lmstudio, vertex_ai

# MCTS Configuration
MCTS_ENABLED=true
MCTS_ITERATIONS=100
MCTS_C=1.414

# Logging
LOG_LEVEL=INFO
```

### Hugging Face Variables

```bash
# Hugging Face Hub
HF_TOKEN=your-token
HF_SPACE_NAME=your-space-name
```

### Google Cloud Variables

```bash
# Required for Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# For AI Studio (development)
GOOGLE_API_KEY=your-api-key

# For Vertex AI (production)
GOOGLE_GENAI_USE_VERTEXAI=true

# ADK Configuration
ADK_BACKEND=vertex_ai  # local, ml_dev, vertex_ai
ROOT_AGENT_MODEL=gemini-2.0-flash-001
```

---

## Security Best Practices

### Secrets Management

1. **Never commit secrets** to version control
2. Use platform-specific secret management:
   - **Hugging Face**: Space Secrets
   - **Google Cloud**: Secret Manager
3. Rotate API keys regularly

### Network Security

1. Use HTTPS for all external communications
2. Enable VPC-SC for production Google Cloud deployments
3. Implement rate limiting

### IAM Best Practices

1. Use service accounts with minimal permissions
2. Avoid using owner/editor roles
3. Audit IAM policies regularly

---

## Monitoring

### Hugging Face

- View logs in Space settings
- Monitor uptime and usage

### Google Cloud

Enable Cloud Monitoring and Logging:

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision" --limit=100

# View metrics in Console
# https://console.cloud.google.com/monitoring
```

### Prometheus/Grafana

For self-hosted deployments, use the included monitoring stack:

```bash
docker-compose up -d prometheus grafana
```

---

## Troubleshooting

### Hugging Face

**Space not starting:**
- Check build logs in Space settings
- Verify Dockerfile syntax
- Ensure requirements.txt is valid

**Out of memory:**
- Upgrade hardware tier
- Reduce model size
- Enable model quantization

### Google Cloud

**Deployment fails:**
```bash
# Check Cloud Build logs
gcloud builds list --limit=5

# View specific build
gcloud builds log BUILD_ID
```

**Permission denied:**
```bash
# Verify IAM roles
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:"
```

---

## Cost Optimization

### Hugging Face
- Use CPU Basic (free) for demos
- Enable sleep mode for infrequently accessed Spaces

### Google Cloud
- Enable scale-to-zero on Cloud Run
- Use committed use discounts for sustained workloads
- Monitor with Cloud Billing budgets

---

## Next Steps

1. **Choose your platform** based on requirements
2. **Configure secrets** for your deployment
3. **Run the deployment script**
4. **Verify** the deployment is working
5. **Set up monitoring** for production use
