"""
Tests for meta-controller config loader module.

Tests RNNConfig, BERTConfig, InferenceConfig, MetaControllerConfig,
and MetaControllerConfigLoader methods.
"""

import pytest

from src.agents.meta_controller.config_loader import (
    BERTConfig,
    InferenceConfig,
    MetaControllerConfig,
    MetaControllerConfigLoader,
    RNNConfig,
)


@pytest.mark.unit
class TestRNNConfig:
    def test_defaults(self):
        cfg = RNNConfig()
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 1
        assert cfg.dropout == 0.1
        assert cfg.model_path is None


@pytest.mark.unit
class TestBERTConfig:
    def test_defaults(self):
        cfg = BERTConfig()
        assert cfg.model_name == "prajjwal1/bert-mini"
        assert cfg.use_lora is True
        assert cfg.lora_r == 4
        assert cfg.lora_alpha == 16


@pytest.mark.unit
class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.device is None
        assert cfg.seed == 42


@pytest.mark.unit
class TestMetaControllerConfig:
    def test_defaults(self):
        cfg = MetaControllerConfig()
        assert cfg.enabled is False
        assert cfg.type == "rnn"
        assert cfg.fallback_to_rule_based is True
        assert isinstance(cfg.rnn, RNNConfig)
        assert isinstance(cfg.bert, BERTConfig)


@pytest.mark.unit
class TestMetaControllerConfigLoader:
    def test_get_default_config(self):
        cfg = MetaControllerConfigLoader.get_default_config()
        assert isinstance(cfg, MetaControllerConfig)
        assert cfg.enabled is False

    def test_load_from_dict(self):
        d = {"enabled": True, "type": "bert", "bert": {"model_name": "bert-base-uncased"}}
        cfg = MetaControllerConfigLoader.load_from_dict(d)
        assert cfg.enabled is True
        assert cfg.type == "bert"
        assert cfg.bert.model_name == "bert-base-uncased"

    def test_load_from_dict_defaults(self):
        cfg = MetaControllerConfigLoader.load_from_dict({})
        assert cfg.enabled is False
        assert cfg.type == "rnn"

    def test_to_dict(self):
        cfg = MetaControllerConfig(enabled=True, type="bert")
        d = MetaControllerConfigLoader.to_dict(cfg)
        assert d["enabled"] is True
        assert d["type"] == "bert"
        assert "rnn" in d
        assert "bert" in d

    def test_validate_valid(self):
        cfg = MetaControllerConfig()
        MetaControllerConfigLoader.validate(cfg)

    def test_validate_invalid_type(self):
        cfg = MetaControllerConfig(type="transformer")
        with pytest.raises(ValueError, match="Invalid controller type"):
            MetaControllerConfigLoader.validate(cfg)

    def test_validate_invalid_rnn_hidden_dim(self):
        cfg = MetaControllerConfig(rnn=RNNConfig(hidden_dim=-1))
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MetaControllerConfigLoader.validate(cfg)

    def test_validate_invalid_rnn_dropout(self):
        cfg = MetaControllerConfig(rnn=RNNConfig(dropout=1.5))
        with pytest.raises(ValueError, match="dropout must be between"):
            MetaControllerConfigLoader.validate(cfg)

    def test_validate_invalid_bert_lora_r(self):
        cfg = MetaControllerConfig(bert=BERTConfig(lora_r=0))
        with pytest.raises(ValueError, match="lora_r must be positive"):
            MetaControllerConfigLoader.validate(cfg)

    def test_validate_invalid_device(self):
        cfg = MetaControllerConfig(inference=InferenceConfig(device="tpu"))
        with pytest.raises(ValueError, match="Invalid device"):
            MetaControllerConfigLoader.validate(cfg)

    def test_validate_cuda_device(self):
        cfg = MetaControllerConfig(inference=InferenceConfig(device="cuda:0"))
        MetaControllerConfigLoader.validate(cfg)

    def test_validate_model_path_not_found(self):
        cfg = MetaControllerConfig(rnn=RNNConfig(model_path="/nonexistent/model.pt"))
        with pytest.raises(FileNotFoundError):
            MetaControllerConfigLoader.validate(cfg)

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
meta_controller:
  enabled: true
  type: bert
  bert:
    model_name: bert-large
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)
        cfg = MetaControllerConfigLoader.load_from_yaml(str(path))
        assert cfg.enabled is True
        assert cfg.type == "bert"
        assert cfg.bert.model_name == "bert-large"

    def test_load_from_yaml_missing_key(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("other_key: value\n")
        with pytest.raises(KeyError, match="meta_controller"):
            MetaControllerConfigLoader.load_from_yaml(str(path))

    def test_load_from_yaml_not_found(self):
        with pytest.raises(FileNotFoundError):
            MetaControllerConfigLoader.load_from_yaml("/nonexistent.yaml")

    def test_save_and_load_yaml(self, tmp_path):
        cfg = MetaControllerConfig(enabled=True, type="rnn")
        path = str(tmp_path / "saved.yaml")
        MetaControllerConfigLoader.save_to_yaml(cfg, path)
        loaded = MetaControllerConfigLoader.load_from_yaml(path)
        assert loaded.enabled is True
        assert loaded.type == "rnn"
