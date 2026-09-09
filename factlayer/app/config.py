import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

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


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
