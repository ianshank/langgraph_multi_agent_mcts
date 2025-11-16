"""
Mock and lightweight LLM clients for demo purposes.
"""

import asyncio
import random
from typing import Any


class MockLLMClient:
    """Mock LLM client that generates plausible demo responses."""

    def __init__(self):
        self.response_templates = {
            "architecture": [
                "Consider scalability requirements and team expertise",
                "Evaluate coupling, deployment complexity, and operational overhead",
                "Balance between development speed and long-term maintainability"
            ],
            "optimization": [
                "Profile first to identify actual bottlenecks",
                "Consider memory-mapped files and streaming processing",
                "Implement parallel processing with appropriate chunk sizes"
            ],
            "database": [
                "Consider data consistency requirements and query patterns",
                "Evaluate write-heavy vs read-heavy workload characteristics",
                "Plan for horizontal scaling and data distribution strategies"
            ],
            "distributed": [
                "Implement proper failure detection and recovery mechanisms",
                "Use circuit breakers and bulkhead patterns for resilience",
                "Consider eventual consistency vs strong consistency trade-offs"
            ],
            "default": [
                "Break down the problem into smaller components",
                "Consider trade-offs between different approaches",
                "Evaluate based on specific use case requirements"
            ]
        }

    async def generate(self, prompt: str, context: str = "") -> dict[str, Any]:
        """Generate a mock response based on the prompt."""
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.3))

        # Determine response category
        prompt_lower = prompt.lower()
        if "architecture" in prompt_lower or "microservice" in prompt_lower or "monolith" in prompt_lower:
            category = "architecture"
        elif "optim" in prompt_lower or "performance" in prompt_lower or "process" in prompt_lower:
            category = "optimization"
        elif "database" in prompt_lower or "sql" in prompt_lower or "nosql" in prompt_lower:
            category = "database"
        elif "distribut" in prompt_lower or "fault" in prompt_lower or "rate limit" in prompt_lower:
            category = "distributed"
        else:
            category = "default"

        templates = self.response_templates[category]

        # Generate response with some randomness
        response = random.choice(templates)
        confidence = random.uniform(0.6, 0.95)

        # Add more detail based on prompt length (simulating "understanding")
        if len(prompt) > 100:
            confidence = min(0.95, confidence + 0.1)
            response += f". Additionally, {random.choice(self.response_templates['default'])}"

        return {
            "response": response,
            "confidence": round(confidence, 3),
            "tokens_used": len(prompt.split()) * 2 + len(response.split())
        }

    async def generate_reasoning_steps(self, query: str, num_steps: int = 3) -> list[str]:
        """Generate mock reasoning steps."""
        await asyncio.sleep(random.uniform(0.05, 0.15))

        base_steps = [
            f"Analyzing query: '{query[:50]}...'",
            "Identifying key requirements and constraints",
            "Evaluating potential approaches",
            "Considering trade-offs and implications",
            "Synthesizing recommendations based on analysis",
            "Validating conclusions against requirements"
        ]

        return random.sample(base_steps, min(num_steps, len(base_steps)))


class HuggingFaceClient:
    """Lightweight Hugging Face Inference API client."""

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize with a Hugging Face model.

        Args:
            model_id: The model ID on Hugging Face Hub
        """
        self.model_id = model_id
        self._client = None

    def _get_client(self):
        """Lazy load the HF client."""
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(model=self.model_id)
            except ImportError:
                raise ImportError(
                    "huggingface_hub not installed. Install with: pip install huggingface_hub"
                )
        return self._client

    async def generate(self, prompt: str, context: str = "") -> dict[str, Any]:
        """Generate response using Hugging Face Inference API."""
        try:
            client = self._get_client()

            # Format prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            else:
                full_prompt = f"Question: {prompt}\n\nProvide a concise, technical answer:\n\nAnswer:"

            # Call HF Inference API (sync call wrapped in async)
            response_text = await asyncio.to_thread(
                client.text_generation,
                full_prompt,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )

            # Estimate confidence based on response characteristics
            confidence = min(0.95, 0.6 + len(response_text) / 500)

            return {
                "response": response_text.strip(),
                "confidence": round(confidence, 3),
                "tokens_used": len(full_prompt.split()) + len(response_text.split())
            }

        except Exception as e:
            # Fallback to mock on error
            print(f"HF Inference error: {e}. Falling back to mock.")
            mock = MockLLMClient()
            return await mock.generate(prompt, context)

    async def generate_reasoning_steps(self, query: str, num_steps: int = 3) -> list[str]:
        """Generate reasoning steps using HF model."""
        try:
            client = self._get_client()

            prompt = f"""Break down this question into {num_steps} reasoning steps:
Question: {query}

Reasoning steps (one per line):
1."""

            response = await asyncio.to_thread(
                client.text_generation,
                prompt,
                max_new_tokens=200,
                temperature=0.5
            )

            # Parse steps from response
            lines = response.strip().split("\n")
            steps = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove numbering
                    if line[0].isdigit() and "." in line[:3]:
                        line = line.split(".", 1)[1].strip()
                    steps.append(line)

            return steps[:num_steps] if steps else ["Analysis in progress"]

        except Exception as e:
            print(f"HF reasoning error: {e}. Falling back to mock.")
            mock = MockLLMClient()
            return await mock.generate_reasoning_steps(query, num_steps)
