# Fact Knowledge Layer

A document-agnostic system that ingests PDFs, extracts atomic, evidence-grounded
facts, and figures out how facts across documents **corroborate**, **contradict**,
or can be **reconciled through context** (different time period, scope, unit, etc).

Built for the Superjoin VIT 2026 engineering intern assignment. Nothing here is
hard-coded to the three starter PDFs (RBI Annual Report, Economic Survey, IMF
Article IV) — the same pipeline should work on any PDF you throw at it.

> This README covers the parts I (Claude) built: the extraction/linking engine,
> the API, and a minimal UI. Fill in the **Video Demo** section and anything about
> your own setup/deployment before you submit.

---

## Setup and Run Instructions

### 1. Requirements
- Python 3.11+
- A Gemini API key (free tier is enough) — get one at https://aistudio.google.com/apikey

### 2. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
```

`GEMINI_MODEL` defaults to `gemini-2.5-flash` (stable, generous free tier).
Check https://ai.google.dev/gemini-api/docs/models for the current lineup —
Google ships new model generations (e.g. `gemini-3-flash-preview`) fairly
often, and 2.5 is currently slated for shutdown in October 2026.

The first run will download a small local embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) — this happens once, requires
internet, and after that the app works fully offline except for the calls to
the Gemini API itself.

### 4. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** for the UI, or use the API directly (docs at
`http://localhost:8000/docs`).

### 5. Deploy (Render.com)

A `render.yaml` is included (Python web service, `uvicorn app.main:app --host
0.0.0.0 --port $PORT`, persistent disk mounted at `/data` for the SQLite file
and uploaded PDFs). Set `GEMINI_API_KEY` as a secret env var in the Render
dashboard — don't commit it.

### Basic usage

```bash
# upload a PDF — this extracts facts and cross-links them against everything
# already in the knowledge layer
curl -F "file=@somefile.pdf" http://localhost:8000/documents

# list everything extracted so far
curl http://localhost:8000/facts
curl http://localhost:8000/relationships
curl http://localhost:8000/relationships?type=contradicts

# a single fact with its full evidence + related facts
curl http://localhost:8000/facts/12
```

---

## Video Demo

_(link to be added by the candidate)_

---

## Approach

### What counts as a "fact" here

Rather than forcing every document into one fixed schema (e.g. "always
subject/predicate/object", or a hard-coded list of metrics like "GDP growth,
inflation, repo rate"), each extracted fact is a semi-structured record:

| field | meaning |
|---|---|
| `statement` | one-sentence restatement of the fact, in the model's own words |
| `subject` | the normalized real-world thing the fact is about (e.g. "India real GDP growth rate", "RBI repo rate", "Net FDI inflows into India") — this is what facts get matched on across documents |
| `fact_type` | free-text category the model assigns (e.g. `rate`, `monetary_amount`, `event`, `status`, `ratio`, `count`, `forecast`) — not a fixed enum, so new kinds of facts don't need schema changes |
| `value` / `unit` | the number/state and its unit, kept separate so a value can be compared even when phrasing differs |
| `time_scope` | the period the fact applies to (fiscal year, "as of March 2025", a forecast horizon, etc.) |
| `geo_scope` | country/region/entity the fact is scoped to, when relevant |
| `evidence_quote` | a short verbatim quote from the source page that supports the fact |
| `evidence_page` | page number in the source PDF |
| `confidence` | the extraction model's own confidence (0–1) |

Letting the LLM populate `subject` and `fact_type` freely (instead of picking
from a fixed list) is what makes the schema "evolve" as new kinds of documents
come in — a legal contract and a macro report don't need the same fact
taxonomy, but they can live in the same table.

### Pipeline

1. **Extract text per page** (`app/pdf_utils.py`, via `pdfplumber`).
2. **Chunk & extract facts** (`app/extraction.py` + `app/llm_client.py`): pages
   are grouped into ~3-page windows (to keep enough context for things like "as
   of the previous paragraph") and sent to the LLM with a prompt that asks for
   atomic, evidence-grounded facts as JSON. Every returned fact's
   `evidence_quote` is checked with a fuzzy substring match against the actual
   page text — if the model invented a quote that isn't really there, the fact
   is kept but flagged (`grounded = False`) rather than silently trusted. This
   is the main defence against hallucinated "facts."
3. **Embed facts** (`app/embeddings.py`, local `sentence-transformers` model,
   no extra API calls/cost): each fact's `subject` + `value` + `unit` is
   embedded so facts about the "same thing" can be found across documents even
   when the wording differs a lot (e.g. "headline retail inflation" vs "CPI
   inflation").
4. **Link facts** (`app/linking.py`): for every newly extracted fact, find the
   top-K most similar existing facts *from other documents* by cosine
   similarity. Only reasonably similar candidates (above a threshold) go to
   the next step — this keeps LLM calls down to O(new facts) instead of
   O(all pairs).
5. **Classify the relationship** with a second LLM call that sees both facts
   *and* both evidence quotes side by side, and has to return one of
   `corroborates`, `contradicts`, `contextual_reconciliation`, or `unrelated`,
   plus a one-paragraph explanation. `contextual_reconciliation` is
   specifically for "these look contradictory but aren't, because X" — e.g.
   different fiscal years, different units (crore vs billion), or a
   provisional vs. final figure.
6. Everything is persisted in SQLite (`documents`, `facts`, `relationships`)
   and served through a small FastAPI app + a single-page vanilla-JS UI.

### Why this design

- **Incremental, not batch.** Adding a new document only re-runs linking for
  *its* new facts against the existing pool — it never re-embeds or
  re-classifies old facts. This satisfies the "add documents incrementally"
  extension somewhat naturally, and also just makes it fast.
- **Two separate LLM calls (extract, then compare) instead of one big call.**
  Extraction only ever looks at one document at a time, so the prompt doesn't
  grow with the size of the knowledge layer. Comparison only looks at two
  facts at a time. This is what lets the system scale to "many PDFs" without
  the prompt blowing up — the embedding step is what keeps the *number* of
  comparison calls small.
- **Evidence is a verbatim quote + page number, not a paraphrase**, so a human
  can always jump back to the PDF and check the model's work. Ungrounded
  quotes are flagged instead of hidden, which doubles as the built-in
  "extraction failure" surface for case #4 (see below).
- **SQLite over a graph DB.** The assignment explicitly says a graph DB isn't
  the point. Facts + a `relationships` join table already *is* a graph; adding
  Neo4j would just be another moving part without changing what's actually
  being computed.

### AI tools used

Claude (this conversation) designed and wrote the extraction pipeline, linking
logic, API, and UI. The Gemini API (`gemini-2.5-flash`) is used at runtime,
twice per document: once to extract facts from each page-window, and once per
candidate pair to classify the relationship between two facts. Gemini's
free tier is what makes running/demoing this without a paid key practical.

---

## The four required cases

These fall out of running the pipeline on the three starter PDFs; exact
fact IDs will depend on what gets extracted on a given run, so treat the shape
of each case as the important part, not specific numbers:

1. **Corroborated across documents.** The same underlying figure (e.g. a
   growth/inflation number for the same period) shows up in both the Economic
   Survey and the IMF Article IV report, phrased differently — `subject`
   embedding similarity brings them together and the classifier should mark
   them `corroborates`.
2. **Genuine/likely contradiction.** Two documents state different values for
   what looks like the same fact and scope, with no time/unit/scope difference
   the classifier can point to — marked `contradicts`, with the explanation
   naming exactly what differs.
3. **Apparent contradiction reconciled by context.** Two figures that look
   like they disagree turn out to cover different fiscal years, or one is
   provisional and the other revised/final, or the units differ (INR crore vs
   USD billion) — marked `contextual_reconciliation`, with the explanation
   stating the reconciling factor.
4. **An extraction/reasoning failure.** Surfaced automatically via
   `grounded = False` facts (quote didn't verify against the source page) and
   via low-`confidence` facts — both are visible in the UI and via
   `GET /facts?grounded=false`. Pull a concrete example for the demo and
   describe how you'd fix it (e.g. tighter chunking, requiring the model to
   quote a contiguous span, a second-pass verifier call) in **Limitations**.

Use the `/facts` and `/relationships` endpoints (or the UI) to pull the actual
IDs/evidence for whichever examples you find when you run it, and screenshot
those for the video + this section.

---

## Limitations and Next Steps

- **Gemini 2.5 models "think" by default**, and thinking tokens are drawn from
  the same `max_output_tokens` budget as the actual answer — left on, this
  silently truncates the JSON output on chunks with a lot of facts. The code
  disables thinking for both LLM calls (`thinking_budget=0` in
  `app/llm_client.py`) and raises a clear error instead of a cryptic JSON
  parse failure if a response still gets cut off, but if you swap in a
  different model, check whether it has the same behavior.
- **Quote verification is exact-ish string matching**, not semantic — it will
  false-flag a true fact if the model reformats whitespace/numbers slightly
  when quoting. A fuzzy/normalized match (strip punctuation, collapse
  whitespace, allow small edit distance) would reduce false flags.
- **Candidate matching is embedding similarity on a single vector per fact.**
  Two facts about very different things but similar surface wording (e.g. two
  different "growth rate" facts about different sectors) can produce
  borderline-similar embeddings; the relationship classifier is the real
  safety net here, but a bad classification on a bad candidate is still
  possible. Next step: also compare `fact_type` and require some entity/token
  overlap before spending an LLM call.
- **No OCR fallback.** If a PDF is scanned images rather than text,
  `pdfplumber` will return little/no text and extraction will silently produce
  few or no facts. Next step: detect near-empty pages and fall back to an
  OCR pass.
- **Single-tenant, single SQLite file.** Fine for a prototype; would move to
  Postgres + a proper task queue (so PDF processing doesn't block the request)
  before this touched real traffic.
- **No de-duplication of facts within the same document** beyond what the
  extraction prompt naturally avoids — a very repetitive document could
  produce near-duplicate facts.
- **Relationship classification is pairwise**, so it doesn't currently notice
  three-or-more-way relationships (e.g. a fact revised across three
  consecutive annual reports) as a single narrative — each pair is judged
  independently.

## Additional Notes

The system prompt for extraction and linking is intentionally generic — it
never mentions RBI, IMF, or the Economic Survey — the goal was for the exact
same code to work on the "additional PDFs" mentioned in the assignment without
changes.
