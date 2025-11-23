#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Multi-Agent MCTS Training Orchestrator ===${NC}"

# 1. Build Docker Image
echo -e "\n${YELLOW}Step 1: Building Docker image...${NC}"
docker build -f Dockerfile.train -t langgraph-mcts-train:demo --target demo .

# Create cache directories
mkdir -p cache/synthetic_data
mkdir -p cache/research_corpus

# 2. Generate Synthetic Data
echo -e "\n${YELLOW}Step 2: Generating synthetic training data...${NC}"
# We run this inside the container to ensure all dependencies are met
docker run --rm \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/training:/app/training" \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    langgraph-mcts-train:demo \
    python -m scripts.generate_synthetic_training_data \
    --output-dir /app/cache/synthetic_data \
    --num-samples 50 \
    --provider openai \
    --model gpt-3.5-turbo \
    --batch-size 5

# 3. Build arXiv Corpus (Optional but requested)
echo -e "\n${YELLOW}Step 3: Building arXiv research corpus...${NC}"
docker run --rm \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/training:/app/training" \
    langgraph-mcts-train:demo \
    python -m training.examples.build_arxiv_corpus \
    --mode keywords \
    --max-papers 20 \
    --cache-dir /app/cache/research_corpus

# 4. Run Training Demo
echo -e "\n${YELLOW}Step 4: Starting Training Demo...${NC}"
echo "Monitoring logs for 'Loaded <N> samples from <M> synthetic data files'..."

# We use docker-compose logic here but running directly for simplicity and control over volume mounts matches above
# Or we can use the docker-compose command if updated.
# Let's use docker run to be consistent and ensure volume mappings for cache are correct.
# Note: config_local_demo.yaml expects ./cache/synthetic_data. Inside /app, that is /app/cache/synthetic_data.

docker run --name mcts-training-demo -d --rm \
    --gpus all \
    -v "$(pwd)/training:/app/training" \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/logs:/app/logs" \
    -e PINECONE_API_KEY=${PINECONE_API_KEY} \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -e GITHUB_TOKEN=${GITHUB_TOKEN} \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    langgraph-mcts-train:demo \
    python -m training.cli train --demo --skip-verification

echo -e "${GREEN}Training started! Check logs with: docker logs -f mcts-training-demo${NC}"

