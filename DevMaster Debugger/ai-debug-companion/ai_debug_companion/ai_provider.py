"""
AI Provider abstraction for flexible integration.

Supports multiple AI providers: Claude, OpenAI, Ollama, and more.
Users can configure their preferred provider via config file or environment variables.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import os
import json
from pathlib import Path


@dataclass
class AIResponse:
    """Standardized AI response."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.default_model()

    @abstractmethod
    def default_model(self) -> str:
        """Return the default model name for this provider."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> AIResponse:
        """Generate a response from the AI."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        pass


class ClaudeProvider(AIProvider):
    """Anthropic Claude provider."""

    def default_model(self) -> str:
        return "claude-3-5-sonnet-20241022"

    def is_available(self) -> bool:
        return self.api_key is not None or os.getenv("ANTHROPIC_API_KEY") is not None

    def generate(self, prompt: str, max_tokens: int = 1000) -> AIResponse:
        """Generate using Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic package: pip install anthropic")

        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return AIResponse(
            content=message.content[0].text,
            model=self.model,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens
        )


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider."""

    def default_model(self) -> str:
        return "gpt-4o"

    def is_available(self) -> bool:
        return self.api_key is not None or os.getenv("OPENAI_API_KEY") is not None

    def generate(self, prompt: str, max_tokens: int = 1000) -> AIResponse:
        """Generate using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("Install openai package: pip install openai")

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return AIResponse(
            content=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens
        )


class OllamaProvider(AIProvider):
    """Ollama local model provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: str = "http://localhost:11434"):
        super().__init__(api_key, model)
        self.base_url = base_url

    def default_model(self) -> str:
        return "llama3.1"

    def is_available(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str, max_tokens: int = 1000) -> AIResponse:
        """Generate using Ollama API."""
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests package: pip install requests")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")

        data = response.json()

        return AIResponse(
            content=data["response"],
            model=self.model,
            tokens_used=None  # Ollama doesn't always report tokens
        )


class GroqProvider(AIProvider):
    """Groq fast inference provider."""

    def default_model(self) -> str:
        return "llama-3.1-70b-versatile"

    def is_available(self) -> bool:
        return self.api_key is not None or os.getenv("GROQ_API_KEY") is not None

    def generate(self, prompt: str, max_tokens: int = 1000) -> AIResponse:
        """Generate using Groq API."""
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq package: pip install groq")

        api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return AIResponse(
            content=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens
        )


class OpenRouterProvider(AIProvider):
    """OpenRouter aggregator provider (access to many models)."""

    def default_model(self) -> str:
        return "anthropic/claude-3.5-sonnet"

    def is_available(self) -> bool:
        return self.api_key is not None or os.getenv("OPENROUTER_API_KEY") is not None

    def generate(self, prompt: str, max_tokens: int = 1000) -> AIResponse:
        """Generate using OpenRouter API."""
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests package: pip install requests")

        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.text}")

        data = response.json()

        return AIResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            tokens_used=data.get("usage", {}).get("total_tokens")
        )


class AIProviderFactory:
    """Factory for creating AI providers based on configuration."""

    PROVIDERS = {
        "claude": ClaudeProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "groq": GroqProvider,
        "openrouter": OpenRouterProvider,
    }

    @staticmethod
    def create(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> AIProvider:
        """Create an AI provider instance."""
        provider_name = provider_name.lower()

        if provider_name not in AIProviderFactory.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {', '.join(AIProviderFactory.PROVIDERS.keys())}"
            )

        provider_class = AIProviderFactory.PROVIDERS[provider_name]
        return provider_class(api_key=api_key, model=model)

    @staticmethod
    def from_config(config_path: Optional[Path] = None) -> AIProvider:
        """Create provider from configuration file."""
        if config_path is None:
            # Try default locations
            config_locations = [
                Path.home() / ".config" / "devtools" / "ai.json",
                Path.home() / ".devtools-ai.json",
                Path.cwd() / ".devtools-ai.json",
            ]

            for loc in config_locations:
                if loc.exists():
                    config_path = loc
                    break

        if config_path is None or not config_path.exists():
            # Try to auto-detect from environment
            return AIProviderFactory.auto_detect()

        with open(config_path) as f:
            config = json.load(f)

        provider_name = config.get("provider", "claude")
        api_key = config.get("api_key")
        model = config.get("model")

        return AIProviderFactory.create(provider_name, api_key, model)

    @staticmethod
    def auto_detect() -> AIProvider:
        """Auto-detect available provider from environment."""
        # Try in order of preference
        for provider_name, provider_class in [
            ("ollama", OllamaProvider),  # Local first (free!)
            ("claude", ClaudeProvider),
            ("openai", OpenAIProvider),
            ("groq", GroqProvider),
            ("openrouter", OpenRouterProvider),
        ]:
            try:
                provider = provider_class()
                if provider.is_available():
                    return provider
            except:
                continue

        # No provider available
        raise RuntimeError(
            "No AI provider configured. Set one of:\n"
            "  - ANTHROPIC_API_KEY for Claude\n"
            "  - OPENAI_API_KEY for OpenAI\n"
            "  - GROQ_API_KEY for Groq\n"
            "  - OPENROUTER_API_KEY for OpenRouter\n"
            "  - Or run Ollama locally (free!)\n"
            "Or create ~/.devtools-ai.json config file"
        )

    @staticmethod
    def list_available() -> list[str]:
        """List available (configured) providers."""
        available = []

        for name, provider_class in AIProviderFactory.PROVIDERS.items():
            try:
                provider = provider_class()
                if provider.is_available():
                    available.append(name)
            except:
                pass

        return available


def create_config_template(output_path: Path):
    """Create a template configuration file."""
    template = {
        "provider": "claude",
        "api_key": "your-api-key-here",
        "model": "claude-3-5-sonnet-20241022",
        "_comment": "Supported providers: claude, openai, ollama, groq, openrouter",
        "_examples": {
            "claude": {
                "provider": "claude",
                "api_key": "sk-ant-...",
                "model": "claude-3-5-sonnet-20241022"
            },
            "openai": {
                "provider": "openai",
                "api_key": "sk-...",
                "model": "gpt-4o"
            },
            "ollama": {
                "provider": "ollama",
                "model": "llama3.1",
                "api_key": null
            },
            "groq": {
                "provider": "groq",
                "api_key": "gsk_...",
                "model": "llama-3.1-70b-versatile"
            }
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"[OK] Created config template at: {output_path}")
    print(f"[INFO] Edit this file with your AI provider settings")
