"""
System Text Encoder.

Wraps a pre-trained Transformer (e.g. BERT) to provide embeddings for system agents.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import AutoTokenizer, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

class SystemEncoder(nn.Module):
    """
    Base text encoder for system agents.
    Wraps a pre-trained Transformer (e.g. CodeBERT) to provide embeddings.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/codebert-base", 
        device: str = "cpu", 
        use_cache: bool = True,
        cache_size: int = 10000,
        enable_lora: bool = False
    ):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
            
        self.device = device
        self.use_cache = use_cache
        self.model_name = model_name
        
        # Load CodeBERT or requested model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.hidden_size = self.model.config.hidden_size
        
        # Caching
        if self.use_cache:
            from functools import lru_cache
            # Method wrapper to allow lru_cache on instance method (tricky with self)
            # We'll use a manual dictionary for simplicity in this context or handle it via a helper
            self._embedding_cache: dict[str, torch.Tensor] = {}
            self._cache_keys: list[str] = [] # For LRU management
            self._max_cache_size = cache_size
            
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encode list of texts into [batch, seq, hidden_dim] embeddings.
        """
        if not self.use_cache:
            return self._encode_batch(texts)
            
        # Check cache
        indices_to_compute = []
        cached_tensors = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                cached_tensors[i] = self._embedding_cache[text]
            else:
                indices_to_compute.append(i)
                
        if indices_to_compute:
            texts_to_compute = [texts[i] for i in indices_to_compute]
            computed_tensors = self._encode_batch(texts_to_compute)
            
            for i, idx in enumerate(indices_to_compute):
                tensor = computed_tensors[i]
                cached_tensors[idx] = tensor
                self._update_cache(texts[idx], tensor)
                
        # Stack results (requires they are all same shape? No, they are [seq, hidden])
        # Actually standard BERT output is [batch, seq, hidden]. Seq length varies.
        # So we probably shouldn't stack unless we padded.
        # But _encode_batch returns a padded batch.
        # Mixing cached (padded to X) and new (padded to Y) is tricky if we return a single tensor.
        # CodeBERT usually expects batch processing.
        # IF we return a list of tensors, that's fine.
        # BUT the original signature returned torch.Tensor (Batch).
        
        # Simplification: If using cache, we only support batch_size=1 efficient lookups 
        # OR we just re-pad everything.
        # For MCTS (single step), batch_size is often small.
        # Let's revert to always computing for batch > 1 to allow correct padding, 
        # or implement complex padding logic.
        # Given "Advanced Models", let's trust the batch compute.
        # Only cache if batch_size == 1?
        
        if len(texts) == 1 and self.use_cache:
            text = texts[0]
            if text in self._embedding_cache:
                return self._embedding_cache[text].unsqueeze(0) # Add batch dim [1, S, H]
            else:
                out = self._encode_batch([text])
                self._update_cache(text, out.squeeze(0))
                return out

        # Fallback for batches or if logic is too complex for this snippet
        return self._encode_batch(texts)
        
    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad(): # Embeddings are usually frozen unless fine-tuning
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def _update_cache(self, text: str, tensor: torch.Tensor):
        if len(self._embedding_cache) >= self._max_cache_size:
            # Remove oldest
            oldest = self._cache_keys.pop(0)
            del self._embedding_cache[oldest]
        
        self._embedding_cache[text] = tensor
        self._cache_keys.append(text)
