"""
Unit tests for SystemEncoder and SystemDecoder.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.agents.common.system_encoder import SystemEncoder
from src.agents.common.text_decoder import SystemDecoder


@pytest.fixture
def mock_transformers():
    with (
        patch("src.agents.common.system_encoder.AutoTokenizer") as enc_tok,
        patch("src.agents.common.system_encoder.AutoModel") as enc_model,
        patch("src.agents.common.text_decoder.AutoTokenizer") as dec_tok,
        patch("src.agents.common.text_decoder.GPT2LMHeadModel") as dec_model,
        patch("src.agents.common.text_decoder.GPT2Config") as dec_config,
    ):
        # Setup Encoder Mocks
        enc_tok_instance = MagicMock()
        enc_tok.from_pretrained.return_value = enc_tok_instance
        # tokenizer call returns object with .to() method
        enc_output_mock = MagicMock()
        enc_output_mock.to.return_value = enc_output_mock
        enc_output_mock.__getitem__.side_effect = lambda k: torch.ones(
            1, 10
        )  # support ["input_ids"] access if needed or **kwargs
        enc_tok_instance.return_value = enc_output_mock

        enc_model_instance = MagicMock()
        enc_model.from_pretrained.return_value = enc_model_instance
        enc_model_instance.config.hidden_size = 768
        enc_model_instance.to.return_value = enc_model_instance  # Allow chaining .to()
        enc_model_instance.return_value = MagicMock(last_hidden_state=torch.randn(1, 10, 768))

        # Setup Decoder Mocks
        dec_config_instance = MagicMock()
        dec_config.from_pretrained.return_value = dec_config_instance
        dec_config_instance.n_embd = 768

        dec_tok_instance = MagicMock()
        dec_tok.from_pretrained.return_value = dec_tok_instance
        dec_tok_instance.pad_token_id = 0
        dec_tok_instance.eos_token_id = 1
        dec_tok_instance.bos_token_id = 2
        # Mock batch_decode to return list of strings
        dec_tok_instance.batch_decode.return_value = ["decoded text"]

        dec_model_instance = MagicMock()
        dec_model.from_pretrained.return_value = dec_model_instance
        dec_model_instance.to.return_value = dec_model_instance  # Allow chaining .to()
        # Mock forward pass output
        dec_model_instance.return_value = MagicMock(loss=torch.tensor(0.5), logits=torch.randn(1, 10, 50257))
        # Mock generate output
        dec_model_instance.generate.return_value = torch.tensor([[2, 5, 10, 1]])  # simple sequence

        yield {"enc_tok": enc_tok, "enc_model": enc_model, "dec_tok": dec_tok, "dec_model": dec_model}


def test_system_encoder_init(mock_transformers):
    encoder = SystemEncoder(device="cpu")
    assert encoder.hidden_size == 768
    mock_transformers["enc_model"].from_pretrained.assert_called_once()


def test_system_encoder_forward(mock_transformers):
    encoder = SystemEncoder(device="cpu")
    texts = ["hello world"]
    embeddings = encoder(texts)

    assert embeddings.shape == (1, 10, 768)
    mock_transformers["enc_tok"].from_pretrained.return_value.assert_called()


def test_system_decoder_init(mock_transformers):
    decoder = SystemDecoder(device="cpu", latent_dim=768)
    assert decoder.embedding_dim == 768
    mock_transformers["dec_model"].from_pretrained.assert_called_once()


def test_system_decoder_forward(mock_transformers):
    decoder = SystemDecoder(device="cpu", latent_dim=768)
    latent = torch.randn(1, 768)
    target_ids = torch.ones(1, 10, dtype=torch.long)

    output = decoder(latent, target_ids)
    assert output.loss is not None


def test_system_decoder_generate(mock_transformers):
    decoder = SystemDecoder(device="cpu", latent_dim=768)
    latent = torch.randn(1, 768)

    texts = decoder.generate(latent)
    assert len(texts) == 1
    mock_transformers["dec_model"].from_pretrained.return_value.generate.assert_called_once()
