# Google ADK Configuration

This directory contains configuration files for Google ADK agent integration.

## Setup

### 1. Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 2. Google Cloud Authentication

For **local development** with Google Cloud:

```bash
# Install Google Cloud SDK
# Follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

For **production deployment**:

```bash
# Create service account
gcloud iam service-accounts create adk-agent-sa \
    --display-name="ADK Agent Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create adk-agent-key.json \
    --iam-account=adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/adk-agent-key.json
```

### 3. Install Dependencies

Install the Google ADK optional dependencies:

```bash
# Install all dependencies including Google ADK
pip install -e ".[all]"

# Or install only Google ADK dependencies
pip install -e ".[google-adk]"
```

## Backend Options

### Local Backend (Default)

- **Use Case**: Testing and development without Google Cloud
- **Limitations**: No Gemini models, limited functionality
- **Configuration**: `ADK_BACKEND=local`

### ML Dev Backend

- **Use Case**: Local development with Google Gemini API
- **Requirements**: Google Cloud project, API key
- **Configuration**:
  ```
  ADK_BACKEND=ml_dev
  GOOGLE_GENAI_USE_VERTEXAI=false
  GOOGLE_CLOUD_PROJECT=your-project-id
  ```

### Vertex AI Backend

- **Use Case**: Production deployment with full Vertex AI features
- **Requirements**: Google Cloud project, Vertex AI API enabled
- **Configuration**:
  ```
  ADK_BACKEND=vertex_ai
  GOOGLE_GENAI_USE_VERTEXAI=true
  GOOGLE_CLOUD_PROJECT=your-project-id
  GOOGLE_CLOUD_LOCATION=us-central1
  ```

## Agent-Specific Configuration

### ML Engineering Agent

Required environment variables:
- `GOOGLE_CLOUD_PROJECT`: For model registry
- `GOOGLE_CLOUD_STORAGE_BUCKET`: For artifacts
- `MLE_TASK_DIR`: Task configurations
- `MLE_OUTPUT_DIR`: Model outputs

### Data Science Agent

Required for BigQuery:
- `BIGQUERY_PROJECT_ID`
- `BIGQUERY_DATASET_ID`

Required for AlloyDB (optional):
- `ALLOYDB_INSTANCE`
- `ALLOYDB_DATABASE`
- `ALLOYDB_USER`
- `ALLOYDB_PASSWORD`

### Academic Research Agent

Required:
- `ADK_ENABLE_SEARCH=true`: Web search capability
- `CITATION_START_DATE`: Filter for recent papers

### Data Engineering Agent

Required:
- `DATAFORM_REPOSITORY_NAME`
- `DATAFORM_WORKSPACE_NAME`
- `GOOGLE_CLOUD_PROJECT`

### Deep Search Agent

Required:
- `ADK_ENABLE_SEARCH=true`: Web search capability
- `ROOT_AGENT_MODEL`: Gemini model for research

## IAM Permissions

Ensure your service account has these roles:

| Agent | Required Roles |
|-------|----------------|
| ML Engineering | `roles/aiplatform.user`, `roles/storage.objectAdmin` |
| Data Science | `roles/bigquery.dataViewer`, `roles/bigquery.jobUser` |
| Academic Research | `roles/aiplatform.user` |
| Data Engineering | `roles/dataform.editor`, `roles/bigquery.admin` |
| Deep Search | `roles/aiplatform.user` |

## Troubleshooting

### Authentication Errors

```bash
# Verify authentication
gcloud auth application-default print-access-token

# Check project
gcloud config get-value project
```

### API Not Enabled

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable BigQuery API
gcloud services enable bigquery.googleapis.com

# Enable Dataform API
gcloud services enable dataform.googleapis.com
```

### Permission Errors

```bash
# List service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"
```

## Additional Resources

- [Google ADK Documentation](https://cloud.google.com/vertex-ai/docs/agent-development-kit)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Dataform Documentation](https://cloud.google.com/dataform/docs)
