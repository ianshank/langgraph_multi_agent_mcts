# Hugging Face Spaces Deployment Guide

This guide walks you through deploying the LangGraph Multi-Agent MCTS demo to Hugging Face Spaces.

## Prerequisites

- [Hugging Face Account](https://huggingface.co/join)
- Git installed locally
- Python 3.10+ (for local testing)

## Step 1: Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the form:
   - **Owner**: Your username or organization
   - **Space name**: `langgraph-mcts-demo` (or your choice)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (Free tier - sufficient for demo)
   - **Visibility**: Public (or Private)
4. Click **"Create Space"**

## Step 2: Clone and Deploy

### Option A: Git-based Deployment (Recommended)

```bash
# 1. Clone your new empty Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/langgraph-mcts-demo
cd langgraph-mcts-demo

# 2. Copy demo files from this directory
cp -r /path/to/huggingface_space/* .
cp -r /path/to/huggingface_space/.gitignore .

# 3. Verify structure
ls -la
# Should show:
# - app.py
# - requirements.txt
# - README.md
# - .gitignore
# - demo_src/
#   - __init__.py
#   - agents_demo.py
#   - llm_mock.py
#   - mcts_demo.py

# 4. Commit and push
git add -A
git commit -m "Initial deployment of LangGraph Multi-Agent MCTS demo"
git push

# 5. Space will automatically build and deploy (takes 2-5 minutes)
```

### Option B: Direct Upload via Web UI

1. Navigate to your Space on Hugging Face
2. Click **"Files"** tab
3. Click **"Add file"** → **"Upload files"**
4. Upload all files maintaining the directory structure:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `demo_src/__init__.py`
   - `demo_src/agents_demo.py`
   - `demo_src/llm_mock.py`
   - `demo_src/mcts_demo.py`
5. Commit changes

## Step 3: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/langgraph-mcts-demo`
2. Click **"Logs"** tab to monitor build progress
3. Wait for "Running on" message
4. Your demo is now live!

## Step 4: Test the Demo

1. Enter a query or select an example
2. Enable/disable different agents
3. Adjust MCTS parameters
4. Click "Process Query"
5. Review results and consensus scores

## Optional: Enable Real LLM Responses

To use Hugging Face Inference API instead of mock responses:

### 1. Update requirements.txt

```txt
gradio>=4.0.0,<5.0.0
numpy>=1.24.0,<2.0.0
huggingface_hub>=0.20.0
```

### 2. Add Secret Token

1. Go to Space Settings → **Repository secrets**
2. Add new secret:
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token (from [Settings → Access Tokens](https://huggingface.co/settings/tokens))

### 3. Update app.py Initialization

Change line ~290 in `app.py`:

```python
# From:
framework = MultiAgentFrameworkDemo(use_hf_inference=False)

# To:
import os
framework = MultiAgentFrameworkDemo(
    use_hf_inference=True,
    hf_model="mistralai/Mistral-7B-Instruct-v0.2"
)
```

### 4. Commit and Push

```bash
git add -A
git commit -m "Enable Hugging Face Inference API"
git push
```

## Customization Options

### Change Gradio Theme

In `app.py`, modify:

```python
with gr.Blocks(
    theme=gr.themes.Soft(),  # Try: Default(), Monochrome(), Glass()
    ...
) as demo:
```

### Add Custom Examples

Update `EXAMPLE_QUERIES` list in `app.py`:

```python
EXAMPLE_QUERIES = [
    "Your custom query 1",
    "Your custom query 2",
    ...
]
```

### Adjust MCTS Parameters

Modify sliders in `app.py`:

```python
mcts_iterations = gr.Slider(
    minimum=10,
    maximum=200,  # Increase for more thorough search
    value=50,     # Change default
    ...
)
```

### Add More Agent Types

1. Create new agent in `demo_src/agents_demo.py`
2. Add to `MultiAgentFrameworkDemo` in `app.py`
3. Add UI controls in Gradio interface

## Troubleshooting

### Build Fails

- Check **Logs** tab for error details
- Verify `requirements.txt` has compatible versions
- Ensure all imports in `app.py` are satisfied

### Slow Performance

- Reduce default MCTS iterations
- Use mock LLM (no API calls)
- Simplify tree visualization

### Memory Issues (Free Tier)

- Limit max MCTS iterations to 100
- Reduce tree depth in `demo_src/mcts_demo.py`
- Simplify response generation

### Missing Files

Ensure directory structure:
```
your-space/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── demo_src/
    ├── __init__.py
    ├── agents_demo.py
    ├── llm_mock.py
    └── mcts_demo.py
```

## Upgrading Hardware

For better performance:

1. Go to Space Settings
2. Under **Hardware**, select:
   - **CPU Upgrade** ($0.03/hr) - Faster processing
   - **T4 Small** ($0.60/hr) - GPU for neural models
3. Save changes

## Sharing Your Space

### Embed in Website

```html
<iframe
  src="https://YOUR_USERNAME-langgraph-mcts-demo.hf.space"
  frameborder="0"
  width="100%"
  height="600"
></iframe>
```

### Direct Link

Share: `https://huggingface.co/spaces/YOUR_USERNAME/langgraph-mcts-demo`

### API Access

Gradio automatically provides API endpoint:
```
https://YOUR_USERNAME-langgraph-mcts-demo.hf.space/api/predict
```

## Next Steps

1. **Collect Feedback**: Enable flagging for user feedback
2. **Add Analytics**: Track usage patterns
3. **Extend Agents**: Add domain-specific reasoning modules
4. **Integrate RAG**: Connect to vector databases for real context
5. **Add Visualization**: Enhanced tree and consensus displays

## Support

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://www.gradio.app/docs
- **Full Framework**: https://github.com/ianshank/langgraph_multi_agent_mcts

---

**Happy Deploying!** 🚀
