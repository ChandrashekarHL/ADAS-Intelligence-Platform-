"""Provider-neutral LLM contracts.

Nothing in here knows about OpenAI. Agents build an :class:`LLMRequest`, get back an
:class:`LLMResponse`, and every call leaves a :class:`CallRecord` so latency, token use
and failures are observable from day one (spec §6.3 principle 6).
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = 0  # deterministic where the provider supports it
    purpose: str = ""  # free text for observability, e.g. "aeb_diagnosis"

    @classmethod
    def simple(cls, system: str, user: str, **kwargs: object) -> "LLMRequest":
        return cls.model_validate(
            {
                "messages": (
                    ChatMessage(role=Role.SYSTEM, content=system),
                    ChatMessage(role=Role.USER, content=user),
                ),
                **kwargs,
            }
        )


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    provider: str
    usage: TokenUsage
    latency_s: float
    finish_reason: str | None = None


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    vectors: tuple[tuple[float, ...], ...]
    model: str
    provider: str
    usage: TokenUsage
    latency_s: float

    @property
    def dimension(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


CallKind = Literal["completion", "structured", "embedding"]


class CallRecord(BaseModel):
    """One provider call, successful or not. Never contains prompt text or secrets."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model: str
    kind: CallKind
    purpose: str
    usage: TokenUsage = TokenUsage()
    latency_s: float
    attempts: int = 1
    ok: bool
    error: str | None = None
