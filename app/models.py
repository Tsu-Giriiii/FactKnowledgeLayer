import datetime
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    Boolean,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    num_pages = Column(Integer, default=0)
    status = Column(String, default="processing")  # processing | done | failed
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    facts = relationship("Fact", back_populates="document", cascade="all, delete-orphan")


class Fact(Base):
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)

    statement = Column(Text, nullable=False)
    subject = Column(Text, nullable=False)
    fact_type = Column(String, nullable=True)
    value = Column(String, nullable=True)
    unit = Column(String, nullable=True)
    time_scope = Column(String, nullable=True)
    geo_scope = Column(String, nullable=True)

    evidence_quote = Column(Text, nullable=False)
    evidence_page = Column(Integer, nullable=False)

    confidence = Column(Float, default=0.5)
    grounded = Column(Boolean, default=True)  # did evidence_quote verify against page text?

    embedding_json = Column(Text, nullable=True)  # json-encoded list[float]

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="facts")

    def set_embedding(self, vector):
        self.embedding_json = json.dumps(list(map(float, vector)))

    def get_embedding(self):
        if not self.embedding_json:
            return None
        return json.loads(self.embedding_json)


class FactRelationship(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True)
    fact_a_id = Column(Integer, ForeignKey("facts.id"), nullable=False)
    fact_b_id = Column(Integer, ForeignKey("facts.id"), nullable=False)

    relation_type = Column(String, nullable=False)  # corroborates | contradicts | contextual_reconciliation | unrelated
    explanation = Column(Text, nullable=False)
    confidence = Column(Float, default=0.5)
    similarity = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    fact_a = relationship("Fact", foreign_keys=[fact_a_id])
    fact_b = relationship("Fact", foreign_keys=[fact_b_id])
