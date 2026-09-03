"""Build the configured provider. The rest of the app calls this, never a concrete class."""

from app.core.config import Settings
from app.core.errors import ProviderError
from app.llm.fake import FakeProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.provider import CallLog, LLMProvider


def build_provider(settings: Settings, call_log: CallLog | None = None) -> LLMProvider:
    if settings.llm_provider == "fake":
        return FakeProvider(call_log=call_log)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ProviderError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set; "
                "set it in the environment or use LLM_PROVIDER=fake"
            )
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            timeout_s=settings.llm_timeout_s,
            max_attempts=settings.llm_max_attempts,
            call_log=call_log,
        )
    raise ProviderError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
