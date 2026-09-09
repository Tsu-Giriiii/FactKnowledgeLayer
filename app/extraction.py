import logging
import re

from sqlalchemy.orm import Session

from app.config import settings
from app.pdf_utils import extract_pages, chunk_pages
from app.llm_client import extract_facts_from_chunk
from app.embeddings import embed_texts, fact_embedding_text
from app.linking import link_new_facts
from app.models import Document, Fact

logger = logging.getLogger("factlayer.extraction")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _quote_is_grounded(quote: str, page_text: str) -> bool:
    if not quote or not page_text:
        return False
    return _normalize(quote) in _normalize(page_text)


REQUIRED_FIELDS = ("statement", "subject", "evidence_quote", "evidence_page")


def _save_chunk_facts(db: Session, document: Document, pages: list[str], raw_facts: list[dict]) -> list[Fact]:
    """Verify grounding, embed, and persist one chunk's worth of facts.
    Committed immediately so they show up in the UI right away, instead of
    waiting for the whole document to finish."""
    if not raw_facts:
        return []

    for f in raw_facts:
        page_num = f.get("evidence_page")
        page_text = ""
        if isinstance(page_num, int) and 1 <= page_num <= len(pages):
            page_text = pages[page_num - 1]
        f["_grounded"] = _quote_is_grounded(f.get("evidence_quote", ""), page_text)

    embedding_texts = [fact_embedding_text(f) for f in raw_facts]
    vectors = embed_texts(embedding_texts)

    saved: list[Fact] = []
    for f, vector in zip(raw_facts, vectors):
        fact = Fact(
            document_id=document.id,
            statement=f.get("statement", "")[:2000],
            subject=f.get("subject", "")[:500],
            fact_type=(f.get("fact_type") or None),
            value=(str(f.get("value")) if f.get("value") is not None else None),
            unit=(f.get("unit") or None),
            time_scope=(f.get("time_scope") or None),
            geo_scope=(f.get("geo_scope") or None),
            evidence_quote=f.get("evidence_quote", "")[:1000],
            evidence_page=int(f.get("evidence_page", 1)),
            confidence=float(f.get("confidence", 0.5) or 0.5),
            grounded=bool(f.get("_grounded", False)),
        )
        fact.set_embedding(vector)
        db.add(fact)
        saved.append(fact)

    db.commit()  # <- visible in GET /facts and the UI immediately after this
    for fact in saved:
        db.refresh(fact)
    return saved


def process_document(db: Session, document: Document) -> None:
    try:
        logger.info("document %s: extracting PDF text", document.id)
        pages = extract_pages(document.filepath)
        document.num_pages = len(pages)
        db.commit()
        logger.info("document %s: %d pages of text extracted", document.id, len(pages))

        chunks = list(chunk_pages(pages, settings.PAGES_PER_CHUNK))
        total_facts = 0

        for i, (start_page, end_page, chunk_text) in enumerate(chunks, start=1):
            if not chunk_text.strip():
                continue
            logger.info(
                "document %s: LLM extraction on chunk %d/%d (pages %d-%d)",
                document.id,
                i,
                len(chunks),
                start_page,
                end_page,
            )
            try:
                chunk_facts = extract_facts_from_chunk(chunk_text)
            except Exception:  # noqa: BLE001 - keep going on other chunks
                chunk_facts = []
                logger.exception(
                    "document %s: chunk %d-%d extraction failed",
                    document.id,
                    start_page,
                    end_page,
                )
            kept = [f for f in chunk_facts if all(f.get(k) not in (None, "") for k in REQUIRED_FIELDS)]

            # Save + commit this chunk's facts right away, so they're visible
            # in the UI before the next chunk's LLM call even starts.
            saved = _save_chunk_facts(db, document, pages, kept)
            total_facts += len(saved)
            logger.info(
                "document %s: chunk %d/%d yielded %d facts (%d saved) — committed to DB",
                document.id,
                i,
                len(chunks),
                len(chunk_facts),
                len(saved),
            )

            # Cross-link this chunk's facts against everything ingested so
            # far (other documents' facts, plus earlier chunks of this one
            # are skipped by design — see linking.py). Also committed
            # immediately, so relationships show up live too.
            if saved:
                link_new_facts(db, saved)

        if total_facts == 0:
            logger.warning("document %s: no facts extracted from any chunk", document.id)

        document.status = "done"
        db.commit()
        logger.info("document %s: pipeline complete, %d facts total", document.id, total_facts)
    except Exception as e:  # noqa: BLE001
        logger.exception("document %s: processing failed", document.id)
        document.status = "failed"
        document.error = str(e)
        db.commit()
        raise
