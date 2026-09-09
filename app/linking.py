import concurrent.futures as cf
import logging

from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import cosine_similarity
from app.llm_client import classify_relationship, QuotaExhaustedError
from app.models import Fact, FactRelationship, Document

logger = logging.getLogger("factlayer.linking")


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
    document never re-processes facts that were already linked before.

    The candidate-finding + dedup checks (all DB reads) happen on this
    thread; the classify_relationship LLM calls for all candidates across all
    new_facts are then fired off concurrently, and the resulting rows are
    written back on this thread once each call resolves.
    """
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
        logger.info("linking: no existing facts from other documents to compare against yet")
        return

    filenames: dict[int, str] = {}

    def get_filename(document_id: int) -> str:
        if document_id not in filenames:
            doc = db.query(Document).get(document_id)
            filenames[document_id] = doc.filename if doc else "unknown"
        return filenames[document_id]

    # Build the full list of (new_fact, other_fact, similarity) pairs worth
    # classifying, across all new_facts, skipping pairs already linked.
    tasks: list[tuple[Fact, Fact, float]] = []
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
        logger.info(
            "linking: fact %s has %d candidate matches above threshold",
            new_fact.id,
            len(candidates),
        )

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
            tasks.append((new_fact, other, sim))

    if not tasks:
        return

    max_workers = max(1, min(settings.LLM_MAX_CONCURRENCY, len(tasks)))
    logger.info(
        "linking: classifying %d candidate pairs with up to %d concurrent Gemini calls",
        len(tasks),
        max_workers,
    )

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for new_fact, other, sim in tasks:
            fact_a_dict = _fact_to_dict(new_fact, get_filename(new_fact.document_id))
            fact_b_dict = _fact_to_dict(other, get_filename(other.document_id))
            future = executor.submit(classify_relationship, fact_a_dict, fact_b_dict)
            future_map[future] = (new_fact, other, sim)

        quota_hit = False
        for future in cf.as_completed(future_map):
            new_fact, other, sim = future_map[future]
            try:
                result = future.result()
            except QuotaExhaustedError:
                # Don't bother logging one of these per pair — there could
                # be dozens queued up, all failing fast for the same reason.
                quota_hit = True
                continue
            except Exception:  # noqa: BLE001
                logger.exception(
                    "linking: classification failed for facts %s/%s",
                    new_fact.id,
                    other.id,
                )
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

    db.commit()  # keep whatever classifications did succeed before the quota hit

    if quota_hit:
        logger.error("linking: stopped early — Gemini quota exhausted")
        raise QuotaExhaustedError("Gemini quota exhausted during relationship classification")
