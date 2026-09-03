"""OpenAI-backed provider. The only module in the codebase allowed to import ``openai``.

The API key arrives via the constructor (from :class:`~app.core.config.Settings`), is
handed straight to the SDK client and is never stored on this object, logged or included
in any error message.
"""

import time
from collections.abc import Callable, Sequence
from typing import TypeVar, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.core.errors import ProviderError, StructuredOutputError
from app.llm.provider import CallLog, with_retries
from app.llm.schemas import (
    CallKind,
    CallRecord,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _messages(request: LLMRequest) -> list[ChatCompletionMessageParam]:
    return [
        cast(ChatCompletionMessageParam, {"role": m.role.value, "content": m.content})
        for m in request.messages
    ]


class OpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embedding_model: str,
        timeout_s: float = 60.0,
        max_attempts: int = 3,
        call_log: CallLog | None = None,
        client: OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key and client is None:
            raise ProviderError("OpenAI API key is empty")
        self._client = client or OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._model = model
        self._embedding_model = embedding_model
        self._max_attempts = max_attempts
        self._sleep = sleep
        self.call_log = call_log if call_log is not None else CallLog()

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    # --- internals ------------------------------------------------------------------------

    def _call(
        self, fn: Callable[[], R], *, kind: CallKind, model: str, purpose: str
    ) -> tuple[R, float, int]:
        """Run ``fn`` with retries and log the outcome. Never lets the key leak into errors."""
        start = time.perf_counter()
        try:
            result, attempts = with_retries(
                fn, attempts=self._max_attempts, retry_on=TRANSIENT_ERRORS, sleep=self._sleep
            )
        except ProviderError as exc:
            self._log(kind, model, purpose, TokenUsage(), start, self._max_attempts, str(exc))
            raise
        except openai.APIStatusError as exc:  # 4xx that will not improve with a retry
            msg = f"OpenAI returned HTTP {exc.status_code}: {exc.message}"
            self._log(kind, model, purpose, TokenUsage(), start, 1, msg)
            raise ProviderError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"OpenAI client error: {type(exc).__name__}"
            self._log(kind, model, purpose, TokenUsage(), start, 1, msg)
            raise ProviderError(msg) from exc
        return result, time.perf_counter() - start, attempts

    def _log(
        self,
        kind: CallKind,
        model: str,
        purpose: str,
        usage: TokenUsage,
        start: float,
        attempts: int,
        error: str | None,
    ) -> None:
        self.call_log.add(
            CallRecord(
                provider=self.name,
                model=model,
                kind=kind,
                purpose=purpose,
                usage=usage,
                latency_s=time.perf_counter() - start,
                attempts=attempts,
                ok=error is None,
                error=error,
            )
        )

    @staticmethod
    def _usage(raw: object) -> TokenUsage:
        prompt = getattr(raw, "prompt_tokens", 0) or 0
        completion = getattr(raw, "completion_tokens", 0) or 0
        return TokenUsage(prompt_tokens=int(prompt), completion_tokens=int(completion))

    # --- protocol -------------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()

        def go() -> openai.types.chat.ChatCompletion:
            return self._client.chat.completions.create(
                model=self._model,
                messages=_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
            )

        completion, latency, attempts = self._call(
            go, kind="completion", model=self._model, purpose=request.purpose
        )
        choice = completion.choices[0]
        usage = self._usage(completion.usage)
        self._log("completion", self._model, request.purpose, usage, start, attempts, None)
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            provider=self.name,
            usage=usage,
            latency_s=latency,
            finish_reason=choice.finish_reason,
        )

    def complete_structured(self, request: LLMRequest, schema: type[T]) -> tuple[T, LLMResponse]:
        start = time.perf_counter()

        def go() -> openai.types.chat.ParsedChatCompletion[T]:
            return self._client.chat.completions.parse(
                model=self._model,
                messages=_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
                response_format=schema,
            )

        completion, latency, attempts = self._call(
            go, kind="structured", model=self._model, purpose=request.purpose
        )
        choice = completion.choices[0]
        usage = self._usage(completion.usage)
        if choice.message.refusal:
            self._log("structured", self._model, request.purpose, usage, start, attempts, "refusal")
            raise StructuredOutputError(f"model refused: {choice.message.refusal[:200]}")
        parsed = choice.message.parsed
        if parsed is None:
            self._log(
                "structured", self._model, request.purpose, usage, start, attempts, "unparsed"
            )
            raise StructuredOutputError(
                f"no parsed {schema.__name__} in response (finish_reason={choice.finish_reason})"
            )
        self._log("structured", self._model, request.purpose, usage, start, attempts, None)
        response = LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            provider=self.name,
            usage=usage,
            latency_s=latency,
            finish_reason=choice.finish_reason,
        )
        return parsed, response

    def embed(self, texts: Sequence[str], *, purpose: str = "") -> EmbeddingResponse:
        start = time.perf_counter()
        inputs = list(texts)

        def go() -> openai.types.CreateEmbeddingResponse:
            return self._client.embeddings.create(model=self._embedding_model, input=inputs)

        result, latency, attempts = self._call(
            go, kind="embedding", model=self._embedding_model, purpose=purpose
        )
        ordered = sorted(result.data, key=lambda d: d.index)
        usage = self._usage(result.usage)
        self._log("embedding", self._embedding_model, purpose, usage, start, attempts, None)
        return EmbeddingResponse(
            vectors=tuple(tuple(float(x) for x in d.embedding) for d in ordered),
            model=result.model,
            provider=self.name,
            usage=usage,
            latency_s=latency,
        )
