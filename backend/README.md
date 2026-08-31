# Mo3allimAI API (Windows CMD)

```cmd
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs. Run tests with `pytest`.

Create the first admin: `python -m app.scripts.create_admin`.

Logout is currently client-side: remove the bearer token. Refresh tokens, rotation, revocation and HttpOnly cookies can be added later.
# Retrieval pipeline activation (H7)

The default deployment remains rollback-safe: dense retrieval, pedagogical
ranking, and final context composition are disabled by default.

To explicitly activate the H6-validated pipeline, set this one profile in the
deployment environment (not in source control). It deterministically resolves
the H6 values itself, even if legacy/manual flags are present:

```env
PEDAGOGICAL_RETRIEVAL_PIPELINE_MODE=validated_h6
```

Rollback uses the safe defaults:

```env
PEDAGOGICAL_RETRIEVAL_PIPELINE_MODE=legacy
RETRIEVAL_MODE=dense
PEDAGOGICAL_RANKING_ENABLED=false
PEDAGOGICAL_CONTEXT_COMPOSITION_ENABLED=false
RAG_RERANKER_ENABLED=false
```

The effective pipeline is emitted only in startup and internal per-request
logs; these traces never expose prompts, chunk bodies, tokens, keys, or vector
identifiers.

## Arabic linguistic reviewer

Arabic answers are reviewed after generation without changing retrieval or the
pedagogical prompt. The primary reviewer makes two total attempts by default
(`ARABIC_REVIEW_MAX_RETRIES=1`) and honours `Retry-After` with a small jitter.
An optional independent OpenAI-compatible fallback can be enabled only when
its base URL, API key and model are configured via the
`ARABIC_REVIEW_FALLBACK_*` variables. If neither reviewer succeeds, the
original generated answer is returned safely.
