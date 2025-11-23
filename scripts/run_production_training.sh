#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Multi-Agent MCTS Production Training ===${NC}"

# 1. Build Docker Image (Target: Production)
echo -e "\n${YELLOW}Step 1: Building Production Docker image...${NC}"
docker build -f Dockerfile.train -t langgraph-mcts-train:prod --target production .

# Create cache directories
mkdir -p cache/synthetic_data
mkdir -p cache/research_corpus
mkdir -p cache/dabstep
mkdir -p cache/primus_seed
mkdir -p checkpoints/prod
mkdir -p logs/prod

# 2. Generate/Update Synthetic Data (Larger batch for production)
echo -e "\n${YELLOW}Step 2: Generating synthetic training data (Production Scale)...${NC}"
# Only run if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY not set, skipping synthetic data generation."
else
    docker run --rm \
        -v "$(pwd)/cache:/app/cache" \
        -v "$(pwd)/training:/app/training" \
        -e OPENAI_API_KEY=${OPENAI_API_KEY} \
        langgraph-mcts-train:prod \
        python -m scripts.generate_synthetic_training_data \
        --output-dir /app/cache/synthetic_data \
        --num-samples 1000 \
        --provider openai \
        --model gpt-4-turbo-preview \
        --batch-size 20
fi

# 3. Build Full arXiv Corpus
echo -e "\n${YELLOW}Step 3: Building full arXiv research corpus...${NC}"
docker run --rm \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/training:/app/training" \
    langgraph-mcts-train:prod \
    python -m training.examples.build_arxiv_corpus \
    --mode categories \
    --max-papers 200 \
    --cache-dir /app/cache/research_corpus

# 4. Run Production Training
echo -e "\n${YELLOW}Step 4: Starting Production Training...${NC}"

# Check for W&B key
WANDB_ENV=""
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: WANDB_API_KEY not set. Disabling W&B logging.${NC}"
    WANDB_ENV="-e WANDB_MODE=disabled"
else
    WANDB_ENV="-e WANDB_API_KEY=${WANDB_API_KEY}"
fi

docker run --name mcts-training-prod -d \
    --rm \
    --gpus all \
    -v "$(pwd)/training:/app/training" \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/logs:/app/logs" \
    -e PINECONE_API_KEY=${PINECONE_API_KEY} \
    $WANDB_ENV \
    -e GITHUB_TOKEN=${GITHUB_TOKEN} \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    langgraph-mcts-train:prod \
    python -m training.cli train --config training/config.yaml

if [ -z "$PINECONE_API_KEY" ] || [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: Missing PINECONE_API_KEY or GITHUB_TOKEN. Training might fail or have limited functionality.${NC}"
fi

echo -e "${GREEN}Production Training started! Check logs with: docker logs -f mcts-training-prod${NC}"

