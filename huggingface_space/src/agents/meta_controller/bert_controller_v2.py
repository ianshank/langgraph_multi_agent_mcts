"""
BERT-based Meta-Controller V2 with Graceful LoRA Fallback (2025-11-25).

This is version 2 with improved error handling and graceful degradation.
If PEFT fails to load due to version mismatches, falls back to base BERT.

VERSION: 2025-11-25-FIX-REDUX
"""

import logging
import warnings
from typing import Any

import torch

from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.utils import features_to_text

# Configure logging
logger = logging.getLogger(__name__)

# Version identifier for debugging
CONTROLLER_VERSION = "2025-11-25-FIX-REDUX"

# Handle optional transformers and peft imports gracefully
_TRANSFORMERS_AVAILABLE = False
_PEFT_AVAILABLE = False
_PEFT_ERROR: Exception | None = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
    logger.info(f"✅ BERT Controller V2 ({CONTROLLER_VERSION}): transformers loaded successfully")
except ImportError as e:
    warnings.warn(
        f"transformers library not installed: {e}",
        ImportWarning,
        stacklevel=2,
    )
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
    logger.info(f"✅ BERT Controller V2 ({CONTROLLER_VERSION}): peft loaded successfully")
except ImportError as e:
    # Graceful degradation - PEFT is optional
    _PEFT_AVAILABLE = False
    _PEFT_ERROR = e
    logger.warning(
        f"⚠️ BERT Controller V2 ({CONTROLLER_VERSION}): peft not available (will use base BERT): {e}"
    )
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore
    PeftModel = None  # type: ignore
except Exception as e:
    # Catch all other errors (like the transformers.modeling_layers issue)
    _PEFT_AVAILABLE = False
    _PEFT_ERROR = e
    logger.error(
        f"❌ BERT Controller V2 ({CONTROLLER_VERSION}): peft failed to load: {type(e).__name__}: {e}"
    )
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore
    PeftModel = None  # type: ignore


class BERTMetaController(AbstractMetaController):
    """
    BERT-based meta-controller V2 with graceful LoRA fallback.

    This version (V2) improves error handling:
    - Falls back to base BERT if PEFT fails to load
    - Continues working even with version mismatches
    - Provides clear logging about what's loaded

    Attributes:
        DEFAULT_MODEL_NAME: Default BERT model to use.
        NUM_LABELS: Number of output labels (agents to choose from).
        device: PyTorch device for tensor operations.
        model_name: Name of the pre-trained model.
        lora_r: LoRA rank parameter.
        lora_alpha: LoRA alpha scaling parameter.
        lora_dropout: LoRA dropout rate.
        use_lora: Whether to use LoRA adapters (may be False if PEFT unavailable).
        tokenizer: BERT tokenizer for text processing.
        model: BERT sequence classification model (with or without LoRA).
    """

    DEFAULT_MODEL_NAME = "prajjwal1/bert-mini"
    NUM_LABELS = 3

    def __init__(
        self,
        name: str = "BERTMetaController",
        seed: int = 42,
        model_name: str | None = None,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str | None = None,
        use_lora: bool = True,
    ) -> None:
        """
        Initialize the BERT meta-controller V2 with graceful LoRA fallback.

        Args:
            name: Name identifier for this controller.
            seed: Random seed for reproducibility.
            model_name: Pre-trained model name from HuggingFace.
            lora_r: LoRA rank parameter (lower = more compression).
            lora_alpha: LoRA alpha scaling parameter.
            lora_dropout: Dropout rate for LoRA layers.
            device: Device to run model on ('cpu', 'cuda', 'mps', etc.).
            use_lora: Whether to attempt LoRA (will fall back if unavailable).

        Raises:
            ImportError: Only if transformers library is not installed.
        """
        super().__init__(name=name, seed=seed)

        logger.info(f"🚀 Initializing BERT Controller V2 ({CONTROLLER_VERSION})")

        # Check for required dependencies
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for BERTMetaController. Install it with: pip install transformers"
            )

        # Handle PEFT availability gracefully
        if use_lora and not _PEFT_AVAILABLE:
            logger.warning(
                f"⚠️ LoRA requested but PEFT unavailable (error: {_PEFT_ERROR}). "
                "Falling back to base BERT model without LoRA."
            )
            use_lora = False

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"📍 Using device: {self.device}")

        # Store configuration parameters
        self.model_name = model_name if model_name is not None else self.DEFAULT_MODEL_NAME
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_lora = use_lora  # May be False even if requested

        logger.info(f"📦 Loading model: {self.model_name}")
        logger.info(f"🔧 LoRA enabled: {self.use_lora}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize base model for sequence classification
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.NUM_LABELS
        )

        # Apply LoRA adapters if requested AND available
        if self.use_lora:
            try:
                logger.info("🎯 Applying LoRA adapters...")
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=["query", "value"],
                )
                self.model = get_peft_model(base_model, lora_config)
                logger.info("✅ LoRA adapters applied successfully")
            except Exception as e:
                logger.error(f"❌ Failed to apply LoRA adapters: {e}. Using base model.")
                self.model = base_model
                self.use_lora = False
        else:
            logger.info("📦 Using base BERT model (no LoRA)")
            self.model = base_model

        # Move model to device
        self.model = self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Initialize tokenization cache for performance optimization
        self._tokenization_cache: dict[str, Any] = {}

        logger.info(f"✅ BERT Controller V2 ({CONTROLLER_VERSION}) initialized successfully")

    def predict(self, features: MetaControllerFeatures) -> MetaControllerPrediction:
        """
        Predict which agent should handle the current query.

        Args:
            features: Features extracted from the current agent state.

        Returns:
            Prediction containing the selected agent, confidence score,
            and probability distribution over all agents.
        """
        # Convert features to structured text
        text = features_to_text(features)

        # Check cache for tokenized text
        if text in self._tokenization_cache:
            inputs = self._tokenization_cache[text]
        else:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # Cache the tokenized result
            self._tokenization_cache[text] = inputs

        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Perform inference without gradient tracking
        with torch.no_grad():
            # Get logits from model
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get predicted agent index (argmax)
            predicted_idx = torch.argmax(probabilities, dim=-1).item()

            # Extract confidence for selected agent
            confidence = probabilities[0, predicted_idx].item()

            # Create probability dictionary
            prob_dict: dict[str, float] = {}
            for i, agent_name in enumerate(self.AGENT_NAMES):
                prob_dict[agent_name] = probabilities[0, i].item()

        # Get agent name
        selected_agent = self.AGENT_NAMES[predicted_idx]

        return MetaControllerPrediction(
            agent=selected_agent,
            confidence=float(confidence),
            probabilities=prob_dict,
        )

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk with graceful error handling.

        Args:
            path: Path to the saved model file or directory.
        """
        logger.info(f"📥 Loading model from: {path}")

        if self.use_lora and _PEFT_AVAILABLE:
            try:
                # Load PEFT adapter weights
                logger.info("🔧 Loading LoRA adapters...")
                base_model = self.model.get_base_model()
                self.model = PeftModel.from_pretrained(base_model, path)
                self.model = self.model.to(self.device)
                logger.info("✅ LoRA adapters loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load LoRA adapters: {e}")
                logger.warning("⚠️ Continuing with base model")
        else:
            try:
                # Load base model state dict
                logger.info("📦 Loading base model weights...")
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info("✅ Base model weights loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load model weights: {e}")
                logger.warning("⚠️ Continuing with pre-trained weights")

        # Ensure model is in evaluation mode
        self.model.eval()

    def save_model(self, path: str) -> None:
        """
        Save the current model to disk.

        Args:
            path: Path where the model should be saved.
        """
        logger.info(f"💾 Saving model to: {path}")

        try:
            if self.use_lora:
                # Save PEFT adapter weights
                self.model.save_pretrained(path)
                logger.info("✅ LoRA adapters saved successfully")
            else:
                # Save base model state dict
                torch.save(self.model.state_dict(), path)
                logger.info("✅ Base model weights saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        self._tokenization_cache.clear()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current tokenization cache."""
        truncated_keys = [
            key[:50] + "..." if len(key) > 50 else key
            for key in self._tokenization_cache
        ]

        return {
            "cache_size": len(self._tokenization_cache),
            "cache_keys": truncated_keys,
        }

    def get_trainable_parameters(self) -> dict[str, int]:
        """Get the number of trainable and total parameters in the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0.0

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": round(trainable_percentage, 2),
        }

    def get_version_info(self) -> dict[str, Any]:
        """Get version and capability information."""
        return {
            "controller_version": CONTROLLER_VERSION,
            "transformers_available": _TRANSFORMERS_AVAILABLE,
            "peft_available": _PEFT_AVAILABLE,
            "peft_error": str(_PEFT_ERROR) if _PEFT_ERROR else None,
            "using_lora": self.use_lora,
            "model_name": self.model_name,
            "device": str(self.device),
        }
