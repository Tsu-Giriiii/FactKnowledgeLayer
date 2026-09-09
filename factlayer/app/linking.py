from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import cosine_similarity
from app.llm_client import classify_relationship
from app.models import Fact, FactRelationship, Document


def _fact_to_dict(fact: Fact, filename: str) -> dict:
    return {
        "statement": fact.statement,
        "subject": fact.subject,
        "fact_type": fact.fact_type,
        "value": fact.value,
        "unit": fact.unit,
        "time_scope": fact.time_scope,
        "geo_scope": fact.geo_scope,
        "evidence_quote": fact.evidence_quote,
        "evidence_page": fact.evidence_page,
        "document_filename": filename,
    }


def link_new_facts(db: Session, new_facts: list[Fact]) -> None:
    """For each newly extracted fact, find similar existing facts from OTHER
    documents and classify the relationship between them. This only costs
    O(new_facts) LLM calls per candidate, not O(all_facts^2) — adding one more
    document never re-processes facts that were already linked before."""
    if not new_facts:
        return

    new_ids = {f.id for f in new_facts}
    all_other_facts = (
        db.query(Fact)
        .filter(Fact.id.notin_(new_ids))
        .filter(Fact.embedding_json.isnot(None))
        .all()
    )
    if not all_other_facts:
        return

    filenames: dict[int, str] = {}

    def get_filename(document_id: int) -> str:
        if document_id not in filenames:
            doc = db.query(Document).get(document_id)
            filenames[document_id] = doc.filename if doc else "unknown"
        return filenames[document_id]

    other_by_doc: dict[int, list[Fact]] = {}
    for f in all_other_facts:
        other_by_doc.setdefault(f.document_id, []).append(f)

    for new_fact in new_facts:
        new_vec = new_fact.get_embedding()
        if new_vec is None:
            continue

        scored = []
        for other in all_other_facts:
            # Skip comparing against facts from the same document — the
            # interesting relationships here are cross-document.
            if other.document_id == new_fact.document_id:
                continue
            other_vec = other.get_embedding()
            if other_vec is None:
                continue
            sim = cosine_similarity(new_vec, other_vec)
            if sim >= settings.LINK_SIMILARITY_THRESHOLD:
                scored.append((sim, other))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[: settings.LINK_TOP_K]

        for sim, other in candidates:
            # Avoid duplicate relationship rows if this pair was already
            # linked (possible if processing is re-run).
            existing = (
                db.query(FactRelationship)
                .filter(
                    (
                        (FactRelationship.fact_a_id == new_fact.id)
                        & (FactRelationship.fact_b_id == other.id)
                    )
                    | (
                        (FactRelationship.fact_a_id == other.id)
                        & (FactRelationship.fact_b_id == new_fact.id)
                    )
                )
                .first()
            )
            if existing:
                continue

            fact_a_dict = _fact_to_dict(new_fact, get_filename(new_fact.document_id))
            fact_b_dict = _fact_to_dict(other, get_filename(other.document_id))

            try:
                result = classify_relationship(fact_a_dict, fact_b_dict)
            except Exception as e:  # noqa: BLE001
                print(f"[linking] classification failed for facts "
                      f"{new_fact.id}/{other.id}: {e}")
                continue

            relation_type = result.get("relation_type", "unrelated")
            if relation_type == "unrelated":
                continue

            rel = FactRelationship(
                fact_a_id=new_fact.id,
                fact_b_id=other.id,
                relation_type=relation_type,
                explanation=result.get("explanation", ""),
                confidence=float(result.get("confidence", 0.5) or 0.5),
                similarity=sim,
            )
            db.add(rel)
        db.commit()
