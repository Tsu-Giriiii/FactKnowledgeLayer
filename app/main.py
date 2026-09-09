import logging
import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, engine, get_db, SessionLocal
from app.models import Document, Fact, FactRelationship
from app.schemas import DocumentOut, FactOut, FactDetailOut, RelationshipOut
from app.extraction import process_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("factlayer")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Fact Knowledge Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _run_pipeline(document_id: int) -> None:
    """Runs in a background thread with its own DB session. Wrapped so that
    literally anything going wrong here — including bugs we didn't
    anticipate — still ends with the document marked "failed" with a
    message, instead of silently sitting at "processing" forever."""
    logger.info("document %s: starting pipeline", document_id)
    db = SessionLocal()
    try:
        document = db.query(Document).get(document_id)
        if not document:
            logger.warning("document %s: not found, aborting", document_id)
            return
        try:
            # process_document saves + links facts chunk-by-chunk internally
            # (see app/extraction.py), committing after each chunk so the UI
            # can show facts and relationships while later chunks are still
            # being processed.
            process_document(db, document)
            logger.info(
                "document %s: pipeline complete, status=%s",
                document_id,
                document.status,
            )
        except Exception:
            logger.exception("document %s: pipeline failed", document_id)
            db.rollback()
            document = db.query(Document).get(document_id)
            if document and document.status != "failed":
                document.status = "failed"
                document.error = "Unexpected error — check server logs."
                db.commit()
    finally:
        db.close()


@app.post("/documents", response_model=DocumentOut)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    document = Document(filename=file.filename, filepath=dest_path, status="processing")
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(_run_pipeline, document.id)

    return document


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.id.desc()).all()


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


def _fact_out(fact: Fact) -> FactOut:
    data = FactOut.model_validate(fact)
    data.document_filename = fact.document.filename if fact.document else None
    return data


@app.get("/facts", response_model=list[FactOut])
def list_facts(
    document_id: Optional[int] = None,
    grounded: Optional[bool] = None,
    q: Optional[str] = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    query = db.query(Fact)
    if document_id is not None:
        query = query.filter(Fact.document_id == document_id)
    if grounded is not None:
        query = query.filter(Fact.grounded == grounded)
    if q:
        like = f"%{q}%"
        query = query.filter(Fact.subject.ilike(like) | Fact.statement.ilike(like))
    facts = query.order_by(Fact.id.desc()).limit(limit).all()
    return [_fact_out(f) for f in facts]


@app.get("/facts/{fact_id}", response_model=FactDetailOut)
def get_fact(fact_id: int, db: Session = Depends(get_db)):
    fact = db.query(Fact).get(fact_id)
    if not fact:
        raise HTTPException(404, "Fact not found")
    rels = (
        db.query(FactRelationship)
        .filter(
            (FactRelationship.fact_a_id == fact_id)
            | (FactRelationship.fact_b_id == fact_id)
        )
        .all()
    )
    out = FactDetailOut.model_validate(_fact_out(fact).model_dump())
    out.relationships = [
        RelationshipOut(
            id=r.id,
            relation_type=r.relation_type,
            explanation=r.explanation,
            confidence=r.confidence,
            similarity=r.similarity,
            fact_a=_fact_out(r.fact_a),
            fact_b=_fact_out(r.fact_b),
        )
        for r in rels
    ]
    return out


@app.get("/relationships", response_model=list[RelationshipOut])
def list_relationships(
    type: Optional[str] = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    query = db.query(FactRelationship)
    if type:
        query = query.filter(FactRelationship.relation_type == type)
    rels = query.order_by(FactRelationship.id.desc()).limit(limit).all()
    return [
        RelationshipOut(
            id=r.id,
            relation_type=r.relation_type,
            explanation=r.explanation,
            confidence=r.confidence,
            similarity=r.similarity,
            fact_a=_fact_out(r.fact_a),
            fact_b=_fact_out(r.fact_b),
        )
        for r in rels
    ]


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    return {
        "documents": db.query(Document).count(),
        "facts": db.query(Fact).count(),
        "ungrounded_facts": db.query(Fact).filter(Fact.grounded.is_(False)).count(),
        "relationships": db.query(FactRelationship).count(),
        "corroborates": db.query(FactRelationship)
        .filter(FactRelationship.relation_type == "corroborates")
        .count(),
        "contradicts": db.query(FactRelationship)
        .filter(FactRelationship.relation_type == "contradicts")
        .count(),
        "contextual_reconciliation": db.query(FactRelationship)
        .filter(FactRelationship.relation_type == "contextual_reconciliation")
        .count(),
    }
