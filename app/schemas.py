from typing import Optional, List
from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    filename: str
    num_pages: int
    status: str
    error: Optional[str] = None

    class Config:
        from_attributes = True


class FactOut(BaseModel):
    id: int
    document_id: int
    document_filename: Optional[str] = None
    statement: str
    subject: str
    fact_type: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    time_scope: Optional[str] = None
    geo_scope: Optional[str] = None
    evidence_quote: str
    evidence_page: int
    confidence: float
    grounded: bool

    class Config:
        from_attributes = True


class RelationshipOut(BaseModel):
    id: int
    relation_type: str
    explanation: str
    confidence: float
    similarity: Optional[float] = None
    fact_a: FactOut
    fact_b: FactOut

    class Config:
        from_attributes = True


class FactDetailOut(FactOut):
    relationships: List[RelationshipOut] = []
