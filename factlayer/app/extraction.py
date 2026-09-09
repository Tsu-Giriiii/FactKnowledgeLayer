import re

from sqlalchemy.orm import Session

from app.config import settings
from app.pdf_utils import extract_pages, chunk_pages
from app.llm_client import extract_facts_from_chunk
from app.embeddings import embed_texts, fact_embedding_text
from app.models import Document, Fact


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _quote_is_grounded(quote: str, page_text: str) -> bool:
    if not quote or not page_text:
        return False
    return _normalize(quote) in _normalize(page_text)


REQUIRED_FIELDS = ("statement", "subject", "evidence_quote", "evidence_page")


def process_document(db: Session, document: Document) -> None:
    try:
        pages = extract_pages(document.filepath)
        document.num_pages = len(pages)
        db.commit()

        raw_facts: list[dict] = []
        for start_page, end_page, chunk_text in chunk_pages(
            pages, settings.PAGES_PER_CHUNK
        ):
            if not chunk_text.strip():
                continue
            try:
                chunk_facts = extract_facts_from_chunk(chunk_text)
            except Exception as e:  # noqa: BLE001 - keep going on other chunks
                chunk_facts = []
                print(f"[extraction] chunk {start_page}-{end_page} failed: {e}")
            for f in chunk_facts:
                if all(f.get(k) not in (None, "") for k in REQUIRED_FIELDS):
                    raw_facts.append(f)

        if not raw_facts:
            document.status = "done"
            db.commit()
            return

        # Verify each fact's evidence quote actually appears on the page it
        # claims to come from. Ungrounded facts are kept (visible in the UI)
        # but flagged, rather than silently dropped or silently trusted.
        for f in raw_facts:
            page_num = f.get("evidence_page")
            page_text = ""
            if isinstance(page_num, int) and 1 <= page_num <= len(pages):
                page_text = pages[page_num - 1]
            f["_grounded"] = _quote_is_grounded(f.get("evidence_quote", ""), page_text)

        embedding_texts = [fact_embedding_text(f) for f in raw_facts]
        vectors = embed_texts(embedding_texts)

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

        document.status = "done"
        db.commit()
    except Exception as e:  # noqa: BLE001
        document.status = "failed"
        document.error = str(e)
        db.commit()
        raise
