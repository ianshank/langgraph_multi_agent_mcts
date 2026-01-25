"""
Text Decoder for System Agents.

Decodes latent representations back into text using a Transformer decoder
(e.g., GPT-2) with cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    GPT2LMHeadModel = None
    GPT2Config = None
    AutoTokenizer = None


class SystemDecoder(nn.Module):
    """
    Decodes latent embeddings into text sequences.
    Uses a GPT-2 model with cross-attention to condition on the latent state.
    """

    def __init__(
        self, model_name: str = "distilgpt2", latent_dim: int = 512, max_length: int = 128, device: str = "cpu"
    ):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for SystemDecoder")

        self.device = device
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model config and enable cross-attention
        config = GPT2Config.from_pretrained(model_name)
        config.add_cross_attention = True
        config.n_ctx = max_length
        # We might need to project latent_dim to embedding_dim if they differ
        self.embedding_dim = config.n_embd

        if latent_dim != self.embedding_dim:
            self.projection = nn.Linear(latent_dim, self.embedding_dim)
        else:
            self.projection = nn.Identity()

        # Initialize model with config
        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        self.model.to(device)
        self.projection.to(device)

    def forward(self, latent_state: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            latent_state: [batch_size, seq_len, latent_dim] or [batch_size, latent_dim]
            target_ids: [batch_size, seq_len] token ids for teacher forcing

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Ensure latent state has sequence dimension [B, S, H]
        if latent_state.dim() == 2:
            latent_state = latent_state.unsqueeze(1)

        # Project to model dimension
        encoder_hidden_states = self.projection(latent_state)

        # Forward pass
        outputs = self.model(input_ids=target_ids, encoder_hidden_states=encoder_hidden_states, labels=target_ids)

        return outputs

    def generate(self, latent_state: torch.Tensor, max_length: int | None = None, num_beams: int = 1) -> list[str]:
        """
        Generate text from latent state.

        Args:
            latent_state: [batch_size, latent_dim] or [batch_size, seq_len, latent_dim]
            max_length: Maximum generation length
            num_beams: Beam search size

        Returns:
            List of decoded strings
        """
        if latent_state.dim() == 2:
            latent_state = latent_state.unsqueeze(1)

        encoder_hidden_states = self.projection(latent_state)

        # Determine start token (BOS/EOS)
        batch_size = latent_state.size(0)
        start_token = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids = torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            max_length=max_length or self.max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
        )

        # Decode
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return texts
