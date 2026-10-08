"""
LLM provider layer: one `generate()` for the agent, two providers behind it.

Why two providers:
  - Gemini (google-genai) runs the multi-turn GATHER loop. A full agent
    run sends many requests within a minute, which does not fit Groq's
    free-tier tokens-per-minute limit.
  - Groq runs the small, single-shot calls (self-critique by default)
    and is the automatic fallback when Gemini errors or is rate limited.

Both providers return the same small normalized message:
    msg.content      str
    msg.tool_calls   list of OpenAI-style dicts {"id", "type", "function": {"name", "arguments"}}
    msg.gemini_parts original Gemini parts (keeps thought_signature) or []
    msg.provider     "gemini" | "groq"

Configuration (environment variables, all optional except one key):
    GEMINI_API_KEY / GROQ_API_KEY   at least one is required
    LLM_PRIMARY                     "gemini" (default) or "groq"
    LLM_CRITIQUE_PROVIDER           provider for critique calls (default "groq" when its key exists)
    GEMINI_MODEL                    default "gemini-3.7-flash"
    GROQ_MODEL                      default "openai/gpt-oss-120b"

Clients are created lazily, so importing this module never needs keys.
"""

import os
import json
import time

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_clients: dict[str, object] = {}

# After an overload/rate-limit error a provider is skipped for a while, so
# one bad minute doesn't add a failed round-trip to every later call.
COOLDOWN_SECONDS = 90
RETRY_DELAY_SECONDS = 2
_cooldown_until: dict[str, float] = {}


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in (
        "503", "429", "500", "502", "504", "unavailable", "overloaded", "rate limit",
        "resource_exhausted", "high demand", "timeout", "timed out", "connection",
    ))


class LLMError(RuntimeError):
    pass


class Message:
    def __init__(self, content: str = "", tool_calls: list | None = None,
                 gemini_parts: list | None = None, provider: str = ""):
        self.fallback_errors: list[str] = []
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.gemini_parts = gemini_parts or []
        self.provider = provider


def _has_key(provider: str) -> bool:
    return bool(os.environ.get("GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"))


def available_providers() -> list[str]:
    return [p for p in ("gemini", "groq") if _has_key(p)]


def _provider_order(task: str) -> list[str]:
    primary = os.environ.get("LLM_PRIMARY", "gemini").lower()
    if task == "critique":
        default_critique = "groq" if _has_key("groq") else primary
        primary = os.environ.get("LLM_CRITIQUE_PROVIDER", default_critique).lower()
    order = [primary] + [p for p in ("gemini", "groq") if p != primary]
    order = [p for p in order if _has_key(p)]
    now = time.time()
    healthy = [p for p in order if _cooldown_until.get(p, 0) <= now]
    # If every provider is cooling down, still try them all.
    return healthy + [p for p in order if p not in healthy]


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def generate(messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.3, task: str = "gather", notify=None) -> Message:
    """
    Run one LLM call. `tools` is the OpenAI-style tool schema list (or None).
    Tries providers in order for `task` and falls back on any error.
    """
    order = _provider_order(task)
    if not order:
        raise LLMError("No LLM key configured. Set GEMINI_API_KEY and/or GROQ_API_KEY.")

    errors = []
    for provider in order:
        call = _gemini_generate if provider == "gemini" else _groq_generate
        for attempt in range(2):
            try:
                msg = call(messages, tools, temperature)
                msg.fallback_errors = errors
                return msg
            except Exception as exc:  # noqa: BLE001 — any provider failure triggers fallback
                errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:300]}")
                if not _is_transient(exc):
                    break  # permanent failure (bad key, bad request): no "busy" signal
                if notify:
                    notify(provider, str(exc)[:160])
                if attempt == 0:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    _cooldown_until[provider] = time.time() + COOLDOWN_SECONDS
    raise LLMError(" | ".join(errors))


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------

def _gemini_client():
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _clients["gemini"]


def _gemini_tools(tools: list[dict]):
    """Convert OpenAI-style tool schemas into Gemini function declarations."""
    from google.genai import types

    declarations = []
    for tool in tools:
        fn = tool["function"]
        params = fn["parameters"]
        properties = {}
        for name, spec in params.get("properties", {}).items():
            item = {
                "type": getattr(types.Type, spec.get("type", "string").upper(), types.Type.STRING),
                "description": spec.get("description", ""),
            }
            if "enum" in spec:
                item["enum"] = spec["enum"]
            if spec.get("type") == "array":
                item["items"] = types.Schema(type=types.Type.STRING)
            properties[name] = item
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=params.get("required", []),
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _gemini_contents(messages: list[dict]):
    """Convert the agent's OpenAI-style message history into Gemini contents."""
    from google.genai import types

    contents = []
    for message in messages:
        role = message.get("role")

        if role == "system":
            # System instructions are passed separately.
            continue

        if role == "user":
            contents.append(types.Content(
                role="user", parts=[types.Part.from_text(text=message.get("content", ""))]
            ))

        elif role == "assistant":
            # IMPORTANT for Gemini 3.x: a function_call Part may carry a
            # thought_signature. Rebuilding the call from name/arguments
            # loses it and Gemini rejects the next turn with
            # INVALID_ARGUMENT, so reuse the original Parts verbatim.
            preserved = message.get("gemini_parts")
            if preserved:
                contents.append(types.Content(role="model", parts=preserved))
                continue

            parts = []
            if message.get("content"):
                parts.append(types.Part.from_text(text=message["content"]))
            for tc in message.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parts.append(types.Part.from_function_call(name=fn.get("name", "search"), args=args))
            if parts:
                contents.append(types.Content(role="model", parts=parts))

        elif role == "tool":
            try:
                result = json.loads(message.get("content", "{}"))
            except Exception:
                result = {"result": message.get("content", "")}
            # FunctionResponse.response must be a dict; search can return a list.
            if not isinstance(result, dict):
                result = {"result": result}
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=message.get("name", "search"), response=result)],
            ))

    return contents


def _gemini_generate(messages, tools, temperature) -> Message:
    from google.genai import types

    system = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system").strip()
    config = {"temperature": temperature, "system_instruction": system or None}
    if tools:
        config["tools"] = _gemini_tools(tools)
        # We run the tool loop ourselves; never let the SDK call tools.
        config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

    response = _gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=_gemini_contents(messages),
        config=types.GenerateContentConfig(**config),
    )

    try:
        candidate = response.candidates[0]
        parts = list(candidate.content.parts or [])
    except Exception:
        parts = []

    text, tool_calls = [], []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc:
            tool_calls.append({
                "id": f"gemini_call_{len(tool_calls) + 1}",
                "type": "function",
                "function": {
                    "name": fc.name,
                    "arguments": json.dumps(dict(fc.args or {}), ensure_ascii=False),
                },
            })
        elif getattr(part, "text", None) and not getattr(part, "thought", False):
            text.append(part.text)

    return Message("".join(text), tool_calls, parts, "gemini")


# ---------------------------------------------------------------------
# Groq (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------

def _groq_client():
    if "groq" not in _clients:
        from groq import Groq
        # Fail fast and let generate() handle retry/fallback instead of
        # the SDK silently waiting out rate limits.
        _clients["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60, max_retries=0)
    return _clients["groq"]


def _groq_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            item = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                item["tool_calls"] = m["tool_calls"]
            out.append(item)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m.get("content", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _groq_generate(messages, tools, temperature) -> Message:
    kwargs = {
        "model": GROQ_MODEL,
        "messages": _groq_messages(messages),
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = _groq_client().chat.completions.create(**kwargs)
    msg = response.choices[0].message
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
        }
        for tc in (msg.tool_calls or [])
    ]
    return Message(msg.content or "", tool_calls, [], "groq")
