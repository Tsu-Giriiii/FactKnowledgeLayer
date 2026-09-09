import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # gemini-2.5-flash is stable and has a generous free tier. Swap to
    # gemini-3-flash-preview (or whatever is current) for the newer model —
    # check https://ai.google.dev/gemini-api/docs/models for the latest list.
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./factlayer.db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    LINK_SIMILARITY_THRESHOLD: float = float(
        os.getenv("LINK_SIMILARITY_THRESHOLD", "0.55")
    )
    LINK_TOP_K: int = int(os.getenv("LINK_TOP_K", "5"))

    # Number of PDF pages grouped into one extraction window
    PAGES_PER_CHUNK: int = int(os.getenv("PAGES_PER_CHUNK", "3"))

    # How many Gemini calls are allowed in flight at once (extraction chunks,
    # or relationship-classification calls). Higher = faster documents, but
    # more likely to hit the free-tier RPM limit and trigger retries. Tune
    # this down if you're on a very low free-tier RPM, or up if you have a
    # paid tier.
    LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))

    # Rate-limit retry backoff (only used when Gemini returns a 429 /
    # RESOURCE_EXHAUSTED — other errors still fail fast as before).
    LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "2"))
    LLM_RETRY_MAX_DELAY: float = float(os.getenv("LLM_RETRY_MAX_DELAY", "60"))
    # Give up on a single call after this many retries OR this much total
    # wait time, whichever comes first — a per-minute rate limit should
    # clear well within this; if it hasn't, something else is wrong and
    # we'd rather fail that one chunk/pair than hang forever.
    LLM_RETRY_MAX_ATTEMPTS: int = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "12"))
    LLM_RETRY_MAX_TOTAL_WAIT: float = float(os.getenv("LLM_RETRY_MAX_TOTAL_WAIT", "300"))
    # When Gemini reports a longer-window quota exhausted (e.g. a per-day
    # request cap, not a per-minute rate limit), retrying every minute won't
    # help. We stop making calls for this long instead of hammering the API.
    LLM_DAILY_QUOTA_COOLDOWN: float = float(os.getenv("LLM_DAILY_QUOTA_COOLDOWN", "3600"))


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
