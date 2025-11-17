"""
Pydantic Settings v2 configuration management for LangGraph Multi-Agent MCTS.

Provides:
- Secure configuration loading from environment variables and .env files
- Type-safe settings with validation
- Secrets protection using SecretStr
- MCTS parameter bounds validation
- Support for multiple LLM providers
"""

from enum import Enum

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LMSTUDIO = "lmstudio"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with security-first configuration.

    All sensitive values use SecretStr to prevent accidental exposure in logs.
    Configuration is loaded from environment variables with .env file support.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    # LLM Provider Configuration
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OPENAI, description="LLM provider to use (openai, anthropic, lmstudio)"
    )

    # API Keys (Secrets)
    OPENAI_API_KEY: SecretStr | None = Field(
        default=None, description="OpenAI API key (required if using OpenAI provider)"
    )

    ANTHROPIC_API_KEY: SecretStr | None = Field(
        default=None, description="Anthropic API key (required if using Anthropic provider)"
    )

    BRAINTRUST_API_KEY: SecretStr | None = Field(
        default=None, description="Braintrust API key for experiment tracking (optional)"
    )

    PINECONE_API_KEY: SecretStr | None = Field(
        default=None, description="Pinecone API key for vector storage (optional)"
    )

    PINECONE_HOST: str | None = Field(
        default=None, description="Pinecone host URL (e.g., https://index.svc.environment.pinecone.io)"
    )

    # Local LLM Configuration
    LMSTUDIO_BASE_URL: str | None = Field(
        default="http://localhost:1234/v1", description="LM Studio API base URL for local inference"
    )

    LMSTUDIO_MODEL: str | None = Field(default=None, description="LM Studio model identifier (e.g., liquid/lfm2-1.2b)")

    # MCTS Configuration with bounds validation
    MCTS_ITERATIONS: int = Field(default=100, ge=1, le=10000, description="Number of MCTS iterations (1-10000)")

    MCTS_C: float = Field(
        default=1.414, ge=0.0, le=10.0, description="MCTS exploration weight (UCB1 constant, 0.0-10.0)"
    )

    # Random seed for reproducibility
    SEED: int | None = Field(default=None, ge=0, description="Random seed for reproducibility (optional)")

    # Logging Configuration
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, description="Application log level")

    # OpenTelemetry Configuration
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        default=None, description="OpenTelemetry OTLP exporter endpoint URL"
    )

    # S3 Storage Configuration
    S3_BUCKET: str | None = Field(default=None, description="S3 bucket name for artifact storage")

    S3_PREFIX: str = Field(default="mcts-artifacts", description="S3 key prefix for stored artifacts")

    S3_REGION: str = Field(default="us-east-1", description="AWS region for S3 bucket")

    # Network Configuration (security)
    HTTP_TIMEOUT_SECONDS: int = Field(default=30, ge=1, le=300, description="HTTP request timeout in seconds")

    HTTP_MAX_RETRIES: int = Field(default=3, ge=0, le=10, description="Maximum HTTP request retries")

    # Security Settings
    MAX_QUERY_LENGTH: int = Field(
        default=10000, ge=1, le=100000, description="Maximum allowed query length in characters"
    )

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60, ge=1, le=1000, description="Rate limit for API requests per minute"
    )

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate OpenAI API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "sk-xxx", "REPLACE_ME"):
                raise ValueError("OpenAI API key appears to be a placeholder value")
            if not secret_value.startswith("sk-"):
                raise ValueError("OpenAI API key should start with 'sk-'")
            if len(secret_value) < 20:
                raise ValueError("OpenAI API key appears to be too short")
        return v

    @field_validator("ANTHROPIC_API_KEY")
    @classmethod
    def validate_anthropic_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Anthropic API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("Anthropic API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("Anthropic API key appears to be too short")
        return v

    @field_validator("BRAINTRUST_API_KEY")
    @classmethod
    def validate_braintrust_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Braintrust API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("Braintrust API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("Braintrust API key appears to be too short")
        return v

    @field_validator("PINECONE_API_KEY")
    @classmethod
    def validate_pinecone_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Pinecone API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("Pinecone API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("Pinecone API key appears to be too short")
        return v

    @field_validator("PINECONE_HOST")
    @classmethod
    def validate_pinecone_host(cls, v: str | None) -> str | None:
        """Validate Pinecone host URL format."""
        if v is not None and v != "":
            if not v.startswith("https://"):
                raise ValueError("Pinecone host must start with https://")
            if "pinecone.io" not in v:
                raise ValueError("Pinecone host should be a valid pinecone.io URL")
        return v

    @field_validator("LMSTUDIO_BASE_URL")
    @classmethod
    def validate_lmstudio_url(cls, v: str | None) -> str | None:
        """Validate LM Studio base URL format."""
        if v is not None:
            if not v.startswith(("http://", "https://")):
                raise ValueError("LM Studio base URL must start with http:// or https://")
            # Warn if not localhost (potential security concern)
            if not any(host in v for host in ("localhost", "127.0.0.1", "::1")):
                import warnings

                warnings.warn(
                    "LM Studio URL points to non-localhost address. Ensure this is intentional and secure.",
                    UserWarning,
                    stacklevel=2,
                )
        return v

    @field_validator("OTEL_EXPORTER_OTLP_ENDPOINT")
    @classmethod
    def validate_otel_endpoint(cls, v: str | None) -> str | None:
        """Validate OpenTelemetry endpoint URL."""
        if v is not None and v != "" and not v.startswith(("http://", "https://", "grpc://")):
            raise ValueError("OpenTelemetry endpoint must start with http://, https://, or grpc://")
        return v

    @field_validator("S3_BUCKET")
    @classmethod
    def validate_s3_bucket_name(cls, v: str | None) -> str | None:
        """Validate S3 bucket name format."""
        if v is not None:
            # S3 bucket naming rules
            if len(v) < 3 or len(v) > 63:
                raise ValueError("S3 bucket name must be 3-63 characters long")
            if not v.replace("-", "").replace(".", "").isalnum():
                raise ValueError("S3 bucket name can only contain lowercase letters, numbers, hyphens, and periods")
            if v.startswith("-") or v.endswith("-"):
                raise ValueError("S3 bucket name cannot start or end with a hyphen")
        return v

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "Settings":
        """Ensure required API keys are provided for the selected provider."""
        if self.LLM_PROVIDER == LLMProvider.OPENAI:
            if self.OPENAI_API_KEY is None:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI provider. "
                    "Set the OPENAI_API_KEY environment variable."
                )
        elif self.LLM_PROVIDER == LLMProvider.ANTHROPIC:
            if self.ANTHROPIC_API_KEY is None:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when using Anthropic provider. "
                    "Set the ANTHROPIC_API_KEY environment variable."
                )
        elif self.LLM_PROVIDER == LLMProvider.LMSTUDIO and self.LMSTUDIO_BASE_URL is None:
            raise ValueError("LMSTUDIO_BASE_URL is required when using LM Studio provider.")
        return self

    def get_api_key(self) -> str | None:
        """
        Get the API key for the current provider.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.LLM_PROVIDER == LLMProvider.OPENAI and self.OPENAI_API_KEY:
            return self.OPENAI_API_KEY.get_secret_value()
        elif self.LLM_PROVIDER == LLMProvider.ANTHROPIC and self.ANTHROPIC_API_KEY:
            return self.ANTHROPIC_API_KEY.get_secret_value()
        return None

    def safe_dict(self) -> dict:
        """
        Return settings as dictionary with secrets masked.

        Safe for logging and display purposes.
        """
        data = self.model_dump()
        # Mask sensitive fields
        if "OPENAI_API_KEY" in data and data["OPENAI_API_KEY"]:
            data["OPENAI_API_KEY"] = "***MASKED***"
        if "ANTHROPIC_API_KEY" in data and data["ANTHROPIC_API_KEY"]:
            data["ANTHROPIC_API_KEY"] = "***MASKED***"
        if "BRAINTRUST_API_KEY" in data and data["BRAINTRUST_API_KEY"]:
            data["BRAINTRUST_API_KEY"] = "***MASKED***"
        if "PINECONE_API_KEY" in data and data["PINECONE_API_KEY"]:
            data["PINECONE_API_KEY"] = "***MASKED***"
        return data

    def get_braintrust_api_key(self) -> str | None:
        """
        Get the Braintrust API key if configured.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.BRAINTRUST_API_KEY:
            return self.BRAINTRUST_API_KEY.get_secret_value()
        return None

    def get_pinecone_api_key(self) -> str | None:
        """
        Get the Pinecone API key if configured.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.PINECONE_API_KEY:
            return self.PINECONE_API_KEY.get_secret_value()
        return None

    def __repr__(self) -> str:
        """Safe string representation that doesn't expose secrets."""
        return f"Settings(LLM_PROVIDER={self.LLM_PROVIDER}, LOG_LEVEL={self.LOG_LEVEL})"


# Global settings instance (lazily loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Settings are loaded once and cached. To reload, call reset_settings() first.

    Returns:
        Settings: Application configuration instance

    Raises:
        ValidationError: If configuration is invalid
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    Reset the global settings instance.

    Forces settings to be reloaded from environment on next get_settings() call.
    Useful for testing.
    """
    global _settings
    _settings = None


# Type exports for external use
__all__ = [
    "Settings",
    "LLMProvider",
    "LogLevel",
    "get_settings",
    "reset_settings",
]
