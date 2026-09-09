import json
import logging
import random
import re
import threading
import time
from typing import Optional

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger("factlayer.llm_client")

_client = None


class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini reports a quota error that a short backoff won't
    fix (e.g. a per-day request cap). Callers should stop making further
    calls rather than retry — that's what was causing the pipeline to sit
    there silently "stuck": the old code retried a per-day quota error every
    ~60s forever, which never succeeds until the quota window resets."""


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env, get a free "
                "key at https://aistudio.google.com/apikey, and add it there."
            )
        _client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            # Without this, a network hiccup or a stuck request hangs the
            # background worker forever with no error and no log line.
            # Timeout is in milliseconds.
            http_options=types.HttpOptions(timeout=60_000),
        )
    return _client


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _extract_json(text: str):
    """Best-effort JSON parsing out of an LLM response that should be JSON
    but might have stray prose or code fences around it."""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to grabbing the first [...] or {...} block in the text.
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output: {text[:500]!r}")


# ---------------------------------------------------------------------------
# Rate-limit detection + retry
# ---------------------------------------------------------------------------
# The google-genai SDK raises google.genai.errors.ClientError (a subclass of
# APIError) for 4xx responses, usually with a `.code` (HTTP status int) and
# sometimes a `.details` dict containing the raw error body. We deliberately
# don't import the errors module directly — the detection below is duck-typed
# so it keeps working even if the SDK's exception hierarchy shifts slightly
# between versions.


def _is_rate_limit_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (429, "429"):
        return True
    status = getattr(exc, "status", None)
    if status and "RESOURCE_EXHAUSTED" in str(status).upper():
        return True
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg.upper()
        or "rate limit" in msg.lower()
        or "quota" in msg.lower()
    )


def _quota_violations(exc: Exception) -> list:
    """Pull the structured QuotaFailure.violations list out of the error
    body, if present — this is what tells us WHICH quota was hit (e.g. a
    per-minute RPM limit vs. a per-day RPD limit)."""
    try:
        details = getattr(exc, "details", None)
        if not details:
            return []
        error_body = details.get("error", details) if isinstance(details, dict) else {}
        violations = []
        for item in error_body.get("details", []):
            if str(item.get("@type", "")).endswith("QuotaFailure"):
                violations.extend(item.get("violations", []))
        return violations
    except Exception:  # noqa: BLE001
        return []


def _is_daily_quota_error(exc: Exception) -> bool:
    for v in _quota_violations(exc):
        quota_id = f"{v.get('quotaId', '')} {v.get('quotaMetric', '')}"
        if any(tok in quota_id for tok in ("PerDay", "Daily", "PerProjectPerDay")):
            return True
    msg = str(exc).lower()
    if "resource_exhausted" not in msg and "429" not in str(exc):
        return False
    return any(tok in msg for tok in ("per day", "daily", "requests per day"))


_quota_lock = threading.Lock()
_daily_quota_exhausted_at: Optional[float] = None


def _daily_quota_still_cooling_down() -> bool:
    """True if we recently hit a daily-quota error and haven't waited out
    the cooldown yet — lets every subsequent call fail fast instead of each
    one independently rediscovering the same exhausted quota."""
    global _daily_quota_exhausted_at
    with _quota_lock:
        if _daily_quota_exhausted_at is None:
            return False
        if time.monotonic() - _daily_quota_exhausted_at >= settings.LLM_DAILY_QUOTA_COOLDOWN:
            _daily_quota_exhausted_at = None
            return False
        return True


def _mark_daily_quota_exhausted() -> None:
    global _daily_quota_exhausted_at
    with _quota_lock:
        _daily_quota_exhausted_at = time.monotonic()


def _retry_delay_from_error(exc: Exception) -> Optional[float]:
    """Gemini sometimes tells you exactly how long to wait via a RetryInfo
    entry in the error details. Use it when present instead of guessing."""
    try:
        details = getattr(exc, "details", None)
        if not details:
            return None
        error_body = details.get("error", details) if isinstance(details, dict) else None
        if not error_body:
            return None
        for item in error_body.get("details", []):
            if str(item.get("@type", "")).endswith("RetryInfo"):
                delay = item.get("retryDelay", "")
                if isinstance(delay, str) and delay.endswith("s"):
                    return float(delay[:-1])
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return None


def _finish_reason_str(response) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:  # noqa: BLE001
        return "unknown"


def _generate_json_once(system_prompt: str, user_content: str, max_output_tokens: int):
    """Single attempt — no retry logic here, that lives in the wrapper below
    so it can be reused for both extraction and classification calls."""
    client = get_client()
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            max_output_tokens=max_output_tokens,
        ),
    )
    text = response.text or ""
    if _finish_reason_str(response) == "MAX_TOKENS" or not text.strip():
        raise ValueError(
            "Gemini response was truncated (hit max_output_tokens) before the "
            "JSON could close. Reduce PAGES_PER_CHUNK, or raise max_output_tokens."
        )
    return _extract_json(text)


def _generate_json(system_prompt: str, user_content: str, max_output_tokens: int):
    """Call Gemini with thinking disabled, retrying on rate-limit errors with
    capped exponential backoff + jitter. Two distinct failure modes are
    handled differently:

    - A per-minute RPM rate limit: worth waiting out. Retried up to
      LLM_RETRY_MAX_ATTEMPTS times / LLM_RETRY_MAX_TOTAL_WAIT seconds total,
      then raised as a normal error so the caller's existing "skip this
      chunk/pair" handling kicks in — it does NOT retry forever.
    - A longer-window quota (e.g. per-day request cap): retrying won't help
      any time soon, so we raise QuotaExhaustedError immediately and set a
      cooldown flag so every other in-flight/queued call also fails fast
      instead of each independently discovering the same exhausted quota.

    Any other kind of error (bad JSON, truncation, network) still raises
    immediately, same as before.
    """
    if _daily_quota_still_cooling_down():
        raise QuotaExhaustedError(
            "Gemini daily request quota was exhausted earlier in this run — "
            f"pausing further calls for up to {int(settings.LLM_DAILY_QUOTA_COOLDOWN)}s."
        )

    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return _generate_json_once(system_prompt, user_content, max_output_tokens)
        except Exception as e:  # noqa: BLE001
            if _is_daily_quota_error(e):
                _mark_daily_quota_exhausted()
                logger.error("Gemini daily quota exhausted — pausing further calls: %s", e)
                raise QuotaExhaustedError(str(e)) from e

            if not _is_rate_limit_error(e):
                raise

            elapsed = time.monotonic() - started
            if attempt >= settings.LLM_RETRY_MAX_ATTEMPTS or elapsed >= settings.LLM_RETRY_MAX_TOTAL_WAIT:
                logger.error(
                    "Gemini rate limit persisted for %.0fs across %d attempts — "
                    "giving up on this call",
                    elapsed,
                    attempt,
                )
                raise

            suggested = _retry_delay_from_error(e)
            if suggested is not None:
                delay = suggested
            else:
                delay = min(
                    settings.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    settings.LLM_RETRY_MAX_DELAY,
                )
            delay += random.uniform(0, delay * 0.25)  # jitter so concurrent
            # workers hitting the same limit don't all retry in lockstep
            remaining_budget = settings.LLM_RETRY_MAX_TOTAL_WAIT - elapsed
            delay = max(0.5, min(delay, remaining_budget))
            logger.warning(
                "Gemini rate limit hit (attempt %d, %.0fs elapsed) — retrying in %.1fs",
                attempt,
                elapsed,
                delay,
            )
            time.sleep(delay)


EXTRACTION_SYSTEM_PROMPT = """You are a careful fact-extraction engine. You read a \
chunk of a real-world document (any domain: financial reports, government \
surveys, legal text, contracts, technical docs, etc.) and pull out atomic, \
checkable facts.

A "fact" is a single, specific, checkable claim — usually numeric or a clear \
categorical/status claim. Good facts:
- "India's real GDP growth rate for FY2024-25 (first advance estimate) is 6.4%"
- "The RBI's repo rate was reduced to 6.25% at the February 2025 MPC meeting"
- "Company X's CFO resigned effective March 2025"

Bad "facts" (do NOT extract these): vague statements, opinions, section \
headers, table-of-contents entries, boilerplate, or anything with no concrete \
value/state attached.

For each fact, you MUST be able to point to a short, VERBATIM quote from the \
provided text that supports it — copy the quote exactly, character for \
character, do not paraphrase it, do not fix typos in it. If you cannot find a \
verbatim supporting span, do not emit the fact.

Only extract facts that are actually present in the given text chunk. Do not \
infer facts from outside knowledge. Do not invent numbers.

Respond with ONLY a JSON array (no prose, no markdown fences). Each element:
{
  "statement": "one sentence, your own words, restating the fact",
  "subject": "normalized name of the real-world thing this fact is about, \
suitable for matching the same fact if stated differently elsewhere \
(e.g. 'India real GDP growth rate', not 'the growth rate mentioned above')",
  "fact_type": "your own short free-text category, e.g. rate | monetary_amount \
| event | status | ratio | count | forecast | date | other",
  "value": "the number or state, as a plain string, e.g. '6.4' or 'resigned' \
(omit units here)",
  "unit": "unit if applicable, e.g. '%', 'INR crore', 'USD billion', or null",
  "time_scope": "the period/date this fact applies to, as stated or clearly \
implied by the text, or null if none",
  "geo_scope": "country/region/entity this fact is scoped to, or null",
  "evidence_quote": "verbatim short quote (roughly 5-40 words) from the text \
that supports this fact",
  "evidence_page": <integer page number this quote came from, using the \
[PAGE N] markers in the text>,
  "confidence": <float 0-1, your own confidence that this is a correctly \
extracted, real fact from this document>
}

If there are no extractable facts in the chunk, return [].
Extract at most 25 facts from this chunk — prioritize the clearest, most \
specific, most important ones."""


def extract_facts_from_chunk(chunk_text: str) -> list[dict]:
    data = _generate_json(EXTRACTION_SYSTEM_PROMPT, chunk_text, max_output_tokens=8192)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of facts")
    return data


RELATIONSHIP_SYSTEM_PROMPT = """You compare two facts that were independently \
extracted from (possibly different) documents, and decide how they relate.

Choose exactly one relation_type:
- "corroborates": both facts assert essentially the same real-world thing, \
consistent with each other, even if worded differently or with minor \
rounding differences.
- "contradicts": the facts genuinely conflict — same subject, same scope \
(time/unit/geography), but incompatible values/states, with no evident \
reason for the difference.
- "contextual_reconciliation": the facts LOOK contradictory at first glance \
but are actually both true because of a difference in context — e.g. \
different time periods, different units, one is provisional/estimate and the \
other is final/revised, different geographic or organizational scope, or \
different definitions of the same-sounding metric.
- "unrelated": on closer inspection these aren't really about the same thing \
and shouldn't be linked at all.

Be conservative about "contradicts" — only use it when you cannot identify a \
reasonable contextual explanation for the difference. Prefer \
"contextual_reconciliation" when there IS a plausible explanation, and name \
that explanation explicitly.

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "relation_type": "corroborates" | "contradicts" | "contextual_reconciliation" | "unrelated",
  "explanation": "1-3 sentences: what you compared and why you reached this \
conclusion. If contextual_reconciliation, name the specific reconciling \
factor (which time period/unit/scope differs). If contradicts, name exactly \
what is incompatible.",
  "confidence": <float 0-1>
}"""


def classify_relationship(fact_a: dict, fact_b: dict) -> dict:
    user_content = (
        "FACT A (from document: "
        + fact_a.get("document_filename", "unknown")
        + f", page {fact_a['evidence_page']}):\n"
        f"Statement: {fact_a['statement']}\n"
        f"Subject: {fact_a['subject']}\n"
        f"Value: {fact_a.get('value')} {fact_a.get('unit') or ''}\n"
        f"Time scope: {fact_a.get('time_scope')}\n"
        f"Geo scope: {fact_a.get('geo_scope')}\n"
        f"Evidence quote: \"{fact_a['evidence_quote']}\"\n\n"
        "FACT B (from document: "
        + fact_b.get("document_filename", "unknown")
        + f", page {fact_b['evidence_page']}):\n"
        f"Statement: {fact_b['statement']}\n"
        f"Subject: {fact_b['subject']}\n"
        f"Value: {fact_b.get('value')} {fact_b.get('unit') or ''}\n"
        f"Time scope: {fact_b.get('time_scope')}\n"
        f"Geo scope: {fact_b.get('geo_scope')}\n"
        f"Evidence quote: \"{fact_b['evidence_quote']}\"\n"
    )
    data = _generate_json(RELATIONSHIP_SYSTEM_PROMPT, user_content, max_output_tokens=1024)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object for relationship classification")
    return data
