import concurrent.futures as cf
import logging
import re

from sqlalchemy.orm import Session

from app.config import settings
from app.pdf_utils import extract_pages, chunk_pages
from app.llm_client import extract_facts_from_chunk, QuotaExhaustedError
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

        # Drop empty windows up front so every chunk we submit to the thread
        # pool is real work.
        all_windows = list(chunk_pages(pages, settings.PAGES_PER_CHUNK))
        chunks = [
            (idx, start_page, end_page, chunk_text)
            for idx, (start_page, end_page, chunk_text) in enumerate(all_windows, start=1)
            if chunk_text.strip()
        ]
        total_facts = 0
        total_chunks = len(chunks)
        quota_exhausted = False

        if total_chunks == 0:
            logger.warning("document %s: no text found in any chunk", document.id)
        else:
            # Extraction calls are pure network I/O (no DB touched inside
            # extract_facts_from_chunk), so we can fire several off at once —
            # this is what gets a multi-chunk document done in a couple of
            # minutes instead of one chunk-latency at a time. The DB session
            # itself is NOT thread-safe, so all saving/linking still happens
            # back on this thread as each future resolves.
            max_workers = max(1, min(settings.LLM_MAX_CONCURRENCY, total_chunks))
            logger.info(
                "document %s: extracting %d chunks with up to %d concurrent Gemini calls",
                document.id,
                total_chunks,
                max_workers,
            )

            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(extract_facts_from_chunk, chunk_text): (idx, start_page, end_page)
                    for idx, start_page, end_page, chunk_text in chunks
                }
                # Process in original chunk order (stable UI/log ordering)
                # even though the underlying LLM calls already ran
                # concurrently and may have finished out of order.
                ordered_futures = sorted(
                    future_to_chunk.items(), key=lambda item: item[1][0]
                )

                for future, (idx, start_page, end_page) in ordered_futures:
                    if quota_exhausted:
                        # Every remaining future is either already running
                        # or about to fail-fast on the same cooldown, so
                        # there's nothing useful left to do — stop instead
                        # of grinding through the rest of the chunks.
                        future.cancel()
                        continue

                    try:
                        chunk_facts = future.result()
                    except QuotaExhaustedError as e:
                        logger.error(
                            "document %s: stopping at chunk %d/%d — %s",
                            document.id,
                            idx,
                            total_chunks,
                            e,
                        )
                        document.error = (
                            f"Stopped at chunk {idx}/{total_chunks} after "
                            f"{total_facts} facts — Gemini quota exhausted "
                            f"({e}). Re-upload once the quota resets to "
                            f"pick up where this left off."
                        )
                        quota_exhausted = True
                        continue
                    except Exception:  # noqa: BLE001 - keep going on other chunks
                        chunk_facts = []
                        logger.exception(
                            "document %s: chunk %d-%d extraction failed",
                            document.id,
                            start_page,
                            end_page,
                        )
                    kept = [
                        f for f in chunk_facts
                        if all(f.get(k) not in (None, "") for k in REQUIRED_FIELDS)
                    ]

                    # Save + commit this chunk's facts right away, so they're
                    # visible in the UI as soon as they're ready.
                    saved = _save_chunk_facts(db, document, pages, kept)
                    total_facts += len(saved)
                    logger.info(
                        "document %s: chunk %d/%d yielded %d facts (%d saved) — committed to DB",
                        document.id,
                        idx,
                        total_chunks,
                        len(chunk_facts),
                        len(saved),
                    )

                    # Cross-link this chunk's facts against everything
                    # ingested so far (other documents' facts; earlier
                    # chunks of this same document are skipped by design —
                    # see linking.py). Also committed immediately, so
                    # relationships show up live too.
                    if saved:
                        try:
                            link_new_facts(db, saved)
                        except QuotaExhaustedError as e:
                            logger.error(
                                "document %s: stopping after chunk %d/%d (linking) — %s",
                                document.id,
                                idx,
                                total_chunks,
                                e,
                            )
                            document.error = (
                                f"Stopped after chunk {idx}/{total_chunks} — Gemini "
                                f"quota exhausted while linking ({e}). "
                                f"{total_facts} facts were saved before this."
                            )
                            quota_exhausted = True

        if total_facts == 0 and not quota_exhausted:
            logger.warning("document %s: no facts extracted from any chunk", document.id)

        if quota_exhausted:
            # Partial success: keep the facts/relationships already saved,
            # but flag the document so it's obvious in the UI it didn't
            # finish, with document.error explaining why (set above).
            document.status = "failed"
        else:
            document.status = "done"
        db.commit()
        logger.info(
            "document %s: pipeline %s, %d facts total",
            document.id,
            "stopped early" if document.status == "failed" else "complete",
            total_facts,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("document %s: processing failed", document.id)
        document.status = "failed"
        document.error = str(e)
        db.commit()
        raise
