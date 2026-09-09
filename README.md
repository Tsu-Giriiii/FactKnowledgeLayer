# Fact Knowledge Layer

A document-agnostic system that ingests PDFs, extracts atomic, evidence-grounded facts, and determines how facts across documents **corroborate**, **contradict**, or can be **reconciled through context** such as different time periods, scopes, units, or revisions.

Built for the **Superjoin VIT 2026 Engineering Intern Assignment**.

Nothing in the pipeline is hard-coded to the three starter PDFs (RBI Annual Report, Economic Survey, and IMF Article IV). The same pipeline is designed to work with arbitrary PDF documents.

---

## Setup and Run Instructions

### 1. Requirements

* Python 3.11+
* A Gemini API key
* Internet access for Gemini API calls and the initial download of the local embedding model

Get a Gemini API key from:

https://aistudio.google.com/apikey

### 2. Install

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows, you can also simply copy `.env.example` to `.env` manually.

Then set:

```env
GEMINI_API_KEY=your_api_key_here
```

The application uses:

```env
GEMINI_MODEL=gemini-3.5-flash-lite
```

The Gemini model is used for both fact extraction and relationship classification.

The extraction pipeline uses Gemini 3.x's `thinking_level="minimal"` configuration to keep latency and output overhead low while retaining compatibility with the Gemini 3.x API.

For current Gemini model information, see:

https://ai.google.dev/gemini-api/docs/models

### 4. Local Embedding Model

The system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for local semantic embeddings.

The model is downloaded the first time the application is run and is approximately 90 MB. It is then loaded locally and does not require an API call for generating embeddings.

The embedding model is used to efficiently identify potentially related facts before sending candidate pairs to Gemini for relationship classification.

### 5. Run

Start the FastAPI application:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000
```

for the web UI.

The interactive API documentation is available at:

```text
http://localhost:8000/docs
```

### 6. Deploying on Render

A `render.yaml` is included for deployment as a Python web service.

The service runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

A persistent disk is mounted at `/data` for the SQLite database and uploaded PDFs.

Set the following as a secret environment variable in the Render dashboard:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key should never be committed to the repository.

---

## Basic Usage

Upload a PDF:

```bash
curl -F "file=@somefile.pdf" http://localhost:8000/documents
```

This extracts facts from the document and cross-links them against facts already present in the knowledge layer.

List extracted facts:

```bash
curl http://localhost:8000/facts
```

List relationships:

```bash
curl http://localhost:8000/relationships
```

List only contradictions:

```bash
curl http://localhost:8000/relationships?type=contradicts
```

Retrieve a single fact with its evidence and related facts:

```bash
curl http://localhost:8000/facts/12
```

---

## Video Demo

*(Add the final video/demo link here.)*

The demo should ideally show:

1. Uploading a PDF.
2. Facts being extracted with evidence.
3. Adding another document incrementally.
4. Facts being linked across documents.
5. A corroborating relationship.
6. A contradiction.
7. A contextual reconciliation.
8. An extraction failure / ungrounded fact.
9. The evidence quote and source page for verification.

---

# Approach

## What Counts as a Fact?

Instead of forcing every document into a fixed schema such as subject/predicate/object, or limiting the system to predefined metrics such as GDP growth or inflation, each extracted fact is represented as a semi-structured record.

| Field            | Meaning                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `statement`      | One-sentence restatement of the fact                                                                          |
| `subject`        | The normalized real-world entity or concept the fact concerns                                                 |
| `fact_type`      | Model-assigned category such as `rate`, `monetary_amount`, `event`, `status`, `ratio`, `count`, or `forecast` |
| `value`          | The numerical value or state associated with the fact                                                         |
| `unit`           | Unit associated with the value                                                                                |
| `time_scope`     | Period for which the fact applies                                                                             |
| `geo_scope`      | Geographic/entity scope when relevant                                                                         |
| `evidence_quote` | Short verbatim quote from the source supporting the fact                                                      |
| `evidence_page`  | Page number containing the evidence                                                                           |
| `confidence`     | Model-assigned extraction confidence from 0–1                                                                 |
| `grounded`       | Whether the evidence quote could be verified against the source text                                          |

The `subject` and `fact_type` fields are intentionally flexible rather than being restricted to a predefined enum.

This allows the same knowledge layer to represent facts from fundamentally different documents. For example, a macroeconomic report and a legal document can use different types of facts while still participating in the same extraction and linking pipeline.

---

# Pipeline

### 1. PDF Text Extraction

`app/pdf_utils.py` extracts text from each PDF page using `pdfplumber`.

Keeping page boundaries allows every extracted fact to retain a reference to its original evidence.

---

### 2. Chunking and Fact Extraction

`app/extraction.py` and `app/llm_client.py` group pages into approximately three-page windows.

Each window is sent to Gemini with a generic extraction prompt requesting:

* Atomic facts
* Evidence-grounded statements
* Structured JSON output
* Evidence quotes
* Source page numbers
* Confidence scores

The extraction prompt is document-agnostic and does not contain assumptions about RBI, IMF, GDP, inflation, or any other domain-specific metric.

Every returned evidence quote is subsequently checked against the actual page text.

If the quote cannot be sufficiently matched, the fact is retained but marked:

```text
grounded = False
```

This prevents an unsupported model-generated fact from being silently treated as verified information.

---

### 3. Local Semantic Embeddings

`app/embeddings.py` uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to generate local embeddings.

Each fact is represented using its:

```text
subject + value + unit
```

This allows semantically similar facts to be retrieved even when their wording differs.

For example:

```text
"headline retail inflation"
```

and:

```text
"CPI inflation"
```

may still be identified as potential matches.

No external API call is required for this embedding step.

---

### 4. Candidate Fact Linking

`app/linking.py` compares each newly extracted fact against existing facts from **other documents** using cosine similarity.

Only candidates above a similarity threshold are passed to the next stage.

This is important for scalability.

Instead of asking the LLM to evaluate every possible pair of facts:

```text
O(all facts²)
```

the embedding layer first narrows the search to a small number of plausible candidates.

The LLM is therefore primarily used for semantic reasoning rather than brute-force similarity search.

---

### 5. Relationship Classification

Each candidate pair is sent to Gemini along with:

* Fact A
* Fact A's evidence
* Fact B
* Fact B's evidence
* Relevant metadata such as time and geographic scope

The model classifies the relationship as one of:

```text
corroborates
contradicts
contextual_reconciliation
unrelated
```

It also produces an explanation.

`contextual_reconciliation` is specifically intended for cases where two statements initially appear contradictory but can be reconciled because of contextual differences.

Examples include:

* Different fiscal years
* Different geographic scopes
* Different units
* Forecast vs actual values
* Provisional vs revised/final figures
* Different reporting periods

---

### 6. Persistence and API

The extracted knowledge is persisted in SQLite using three primary entities:

```text
documents
facts
relationships
```

The data is exposed through a FastAPI backend and a minimal vanilla-JavaScript web UI.

The resulting structure can be viewed conceptually as:

```text
Document
   │
   └── Fact
        │
        ├── corroborates ────── Fact
        ├── contradicts ─────── Fact
        └── contextual_reconciliation ─── Fact
```

---

# Why This Design?

### Incremental Processing

Documents are processed incrementally.

When a new document is uploaded:

1. Only the new document is parsed.
2. Only its facts are embedded.
3. Its facts are compared against the existing knowledge layer.
4. Existing facts do not need to be re-extracted or re-embedded.

This makes the system naturally suitable for progressively adding documents.

---

### Two-Stage LLM Architecture

The system deliberately separates:

```text
Fact Extraction
       ↓
Candidate Retrieval
       ↓
Relationship Classification
```

rather than using a single large LLM prompt.

Extraction operates on one document chunk at a time, while relationship classification operates on two candidate facts at a time.

The embedding layer reduces the number of relationship-classification calls by filtering out obviously unrelated facts before invoking the LLM.

---

### Evidence-First Design

Every extracted fact is associated with:

* A source document
* A source page
* A verbatim evidence quote
* A grounding status

This makes the system auditable.

A user can inspect the evidence behind a fact instead of having to blindly trust an LLM-generated statement.

Ungrounded evidence is explicitly surfaced rather than hidden.

---

### SQLite Instead of a Graph Database

The assignment does not require a graph database.

The relationship table already represents the graph structure:

```text
fact → relationship → fact
```

SQLite keeps the prototype simple while still supporting the required knowledge representation and API operations.

A dedicated graph database could be introduced later if relationship traversal became substantially more complex.

---

# AI Tools Used

The application uses **Gemini 3.5 Flash-Lite** through the Gemini API for runtime reasoning.

Gemini is used in two stages:

1. **Fact extraction** — extracting atomic, evidence-grounded facts from document chunks.
2. **Relationship classification** — determining whether candidate facts corroborate, contradict, reconcile through context, or are unrelated.

The application configures Gemini 3.x using:

```python
thinking_config=types.ThinkingConfig(
    thinking_level="minimal"
)
```

This is the Gemini 3.x configuration used by the current implementation.

Local semantic similarity is handled separately using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

so embedding generation does not require additional LLM API calls.

---

# The Four Required Cases

The system is designed to surface the four cases required by the assignment.

Exact fact IDs and values may vary between runs because extraction depends on the model's output. Therefore, the examples below describe the expected structure rather than hard-coding specific IDs.

## 1. Corroborated Across Documents

Two documents contain the same underlying fact, potentially using different wording.

For example:

```text
Document A:
India's real GDP growth was X%.

Document B:
Real economic activity expanded by X% during the same period.
```

Embedding similarity identifies the statements as candidate matches, and Gemini can classify the resulting pair as:

```text
corroborates
```

---

## 2. Genuine or Likely Contradiction

Two documents describe what appears to be the same fact with different values and no obvious contextual explanation.

For example:

```text
Document A:
Inflation was 5.2%.

Document B:
Inflation was 7.1%.
```

If the subject, period, unit, and scope are sufficiently aligned, the relationship can be classified as:

```text
contradicts
```

The explanation should identify what differs between the two statements.

---

## 3. Apparent Contradiction Reconciled by Context

Two facts appear contradictory but actually refer to different contexts.

Examples:

```text
Different fiscal years
Different geographic scopes
Different units
Forecast vs actual
Provisional vs revised value
```

The relationship is classified as:

```text
contextual_reconciliation
```

The explanation identifies the contextual factor that resolves the apparent contradiction.

---

## 4. Extraction or Reasoning Failure

The system also exposes extraction failures rather than hiding them.

A concrete example can be obtained from:

```bash
GET /facts?grounded=false
```

Potential failure signals include:

* `grounded = False`
* Low extraction confidence
* Incorrect or unverifiable evidence quotes
* Incorrect relationship classification

A useful demo example should show the extracted fact, its failed evidence verification, and the corresponding source page.

Possible improvements include:

* Tighter chunking
* Requiring contiguous evidence spans
* A second-pass evidence verifier
* OCR fallback for scanned PDFs
* More structured validation of extracted fields

---

# Limitations and Next Steps

### 1. Gemini 3.x Minimal Thinking Is Not a Hard Zero

The current implementation uses:

```python
thinking_level="minimal"
```

for Gemini 3.x.

This is intended to minimize thinking overhead and latency. It should not be interpreted as a guarantee that zero internal reasoning tokens are used.

If the model is changed, the supported thinking configuration should be reviewed against that model's API specification.

---

### 2. Evidence Verification Is Approximate

Evidence verification currently relies on string/fuzzy matching against the extracted PDF text.

This can produce false negatives when the model changes:

* Whitespace
* Punctuation
* Number formatting
* Unicode characters

A stronger implementation could normalize the source and quote text more aggressively or use a dedicated evidence-verification pass.

---

### 3. Embedding-Based Candidate Retrieval Can Produce False Positives

The candidate-generation stage uses semantic similarity.

Two facts may have similar wording while referring to different entities.

For example:

```text
GDP growth in India
GDP growth in the manufacturing sector
```

could potentially become candidate matches.

The relationship classifier provides a second safety layer, but candidate generation could be improved by additionally considering:

* `fact_type`
* Entity overlap
* Geographic scope
* Time scope
* Numerical compatibility

---

### 4. No OCR Fallback

The current PDF extraction pipeline relies on text being available through `pdfplumber`.

Scanned/image-only PDFs may therefore produce little or no usable text.

A production version should detect near-empty pages and automatically invoke an OCR pipeline.

---

### 5. Single-Tenant SQLite Architecture

SQLite is appropriate for the assignment prototype, but a production deployment would likely use:

```text
PostgreSQL
+
background task queue
+
object storage
```

This would allow document processing to happen asynchronously without blocking API requests.

---

### 6. Limited Intra-Document Deduplication

The current extraction process does not perform a dedicated semantic deduplication stage within the same document.

Highly repetitive documents could therefore produce near-duplicate facts.

A future version could perform semantic deduplication after extraction.

---

### 7. Pairwise Relationship Classification

Relationships are currently evaluated pairwise.

For example:

```text
2023 Report → Fact A
2024 Report → Fact B
2025 Report → Fact C
```

is treated as separate pairs rather than one evolving fact history.

A future version could build temporal fact chains and represent how a value changes across successive documents.

---

# Additional Notes

The extraction and relationship-classification prompts are intentionally generic.

The system does not contain special rules for:

* RBI
* IMF
* Economic Survey
* GDP
* Inflation
* Monetary policy

The same pipeline can therefore be applied to additional PDFs without modifying the extraction logic.

The architecture is designed around a general principle:

```text
PDFs
 ↓
Text + Page Boundaries
 ↓
Atomic Facts + Evidence
 ↓
Local Semantic Retrieval
 ↓
Candidate Fact Pairs
 ↓
LLM Relationship Reasoning
 ↓
Knowledge Layer
```

This separation keeps document ingestion, semantic retrieval, reasoning, persistence, and presentation independent of one another.
