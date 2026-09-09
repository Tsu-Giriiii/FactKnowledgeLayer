import logging
import os

import numpy as np

from app.config import settings

logger = logging.getLogger("factlayer.embeddings")

# huggingface_hub retries with backoff on network errors by default, which on
# a machine with restricted/no internet can look exactly like an indefinite
# hang. Force it to give up quickly and raise instead.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")

_model = None


def get_model():
    global _model
    if _model is None:
        # Imported lazily so the rest of the app can be imported/tested
        # without pulling in torch/sentence-transformers.
        from sentence_transformers import SentenceTransformer

        logger.info(
            "loading embedding model %s (first run downloads it, ~90MB, needs internet)",
            settings.EMBEDDING_MODEL,
        )
        try:
            _model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')
        except Exception:
            logger.exception(
                "failed to load/download embedding model %s — check internet "
                "access from this machine, or pre-download it separately",
                settings.EMBEDDING_MODEL,
            )
            raise
        logger.info("embedding model ready")
    return _model


def fact_embedding_text(fact_like: dict) -> str:
    """Canonicalize the fields that matter for matching (not the full
    sentence, so wording differences in `statement` don't dominate)."""
    parts = [
        fact_like.get("subject") or "",
        str(fact_like.get("value") or ""),
        fact_like.get("unit") or "",
        fact_like.get("time_scope") or "",
        fact_like.get("geo_scope") or "",
    ]
    return " | ".join(p for p in parts if p)


def embed_text(text: str) -> list[float]:
    model = get_model()
    vector = model.encode([text], normalize_embeddings=True)[0]
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
