# Google Cloud Deployment Guide

This directory contains deployment configurations for deploying the LangGraph Multi-Agent MCTS framework to Google Cloud Platform.

## Deployment Options

### 1. Vertex AI Agent Engine (Recommended for Production)

Vertex AI Agent Engine is specifically designed for deploying LangGraph applications at scale with managed infrastructure.

**Features:**
- Managed runtime for agents
- Automatic scaling
- VPC-SC compliance
- IAM integration

### 2. Cloud Run

For containerized deployments with automatic scaling.

**Features:**
- Serverless container deployment
- Pay-per-use pricing
- Easy scaling

### 3. Google AI Studio Integration

For development and prototyping using the Gemini API.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **Required APIs enabled:**
   - Vertex AI API
   - Cloud Run API
   - Artifact Registry API
   - Cloud Build API

Enable APIs:
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## Environment Variables

Set these before deployment:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Quick Start

### Deploy to Vertex AI Agent Engine

```bash
cd deploy/google_cloud
./deploy_vertex_ai.sh
```

### Deploy to Cloud Run

```bash
cd deploy/google_cloud
./deploy_cloud_run.sh
```

## Configuration

See `config.yaml` for deployment configuration options.

## Security

- Always use service accounts with minimal required permissions
- Store secrets in Secret Manager
- Enable VPC-SC for production deployments
- Use IAM for access control
