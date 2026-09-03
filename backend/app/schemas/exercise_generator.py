"""HTTP contracts for the teacher exercise generator.

The generator is KB-first: it extracts, selects and validates existing
exercise material from the knowledge base rather than inventing it. The LLM is
only used to structure/clean the selected material and, when the teacher
explicitly opts in, to adapt it. Provenance (document, page, chunk) is attached
to each exercise only when it is genuinely sourced from a KB block; the LLM
never fabricates a source.
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

ExerciseStatus = Literal["kb_original", "adapted_from_kb", "ai_generated"]
LevelSource = Literal["explicit", "inferred", "generated"]


class ExerciseGenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "A1"
    theme: str = Field(min_length=1, max_length=255)
    objective: str | None = Field(default=None, min_length=1, max_length=1000)
    skills: list[str] = Field(default_factory=lambda: ["Expression orale"])
    # "auto" (default) lets the pedagogical planner decide the per-exercise type
    # mix. A specific type (e.g. "QCM") forces the whole set to that type.
    exercise_type: str = Field(default="auto", max_length=100)
    count: int = Field(default=8, ge=1, le=20)
    language: Literal["ar", "fr", "en", "es"] = "ar"
    # Explicit, bounded opt-in: the LLM may rewrite/adapt the selected KB
    # exercises. When false (default) the LLM only structures/cleans sourced
    # blocks and never invents content unless the KB is insufficient.
    adapt_with_ai: bool = False
    special_instructions: str | None = Field(default=None, max_length=1500)


class ExerciseItem(BaseModel):
    """One sourced-or-adapted exercise with full source traceability."""

    title: str = Field(min_length=1)
    skill: str = ""
    exercise_type: str = ""
    prompt: str = ""
    # `prompt` is non-empty by construction for valid items. It intentionally
    # carries no min_length here so a single structurally-defective item (e.g. an
    # empty consigne from the model) can reach the ExerciseValidator, where it is
    # flagged ("consigne vide") and repaired by the targeted-regeneration loop —
    # rather than aborting the whole generation at the schema boundary. The
    # regeneration step keeps only the valid subset, so no empty prompt is ever
    # emitted to the client.
    context: str = ""
    answer_expectation: str | None = None
    level: str = Field(default="A1")
    # Optional structured representation for typed display (QCM options, V/F,
    # matching…). These are auxiliary to `prompt`/`answer_expectation` and never
    # replace them. The pedagogical validator can enrich them after generation.
    options: list[str] = Field(default_factory=list)
    is_true: bool | None = None
    pairs: list[dict[str, str]] = Field(default_factory=list)
    # Pedagogical traceability. A level/skill/theme/type is "explicit" only when
    # the source document actually states it; otherwise it is "inferred" and is
    # never presented as a proven value.
    level_source: LevelSource = "explicit"
    theme: str = ""
    theme_source: Literal["explicit", "inferred"] = "inferred"
    skill_source: Literal["explicit", "inferred"] = "inferred"
    type_source: Literal["explicit", "inferred"] = "inferred"
    difficulty: Literal["easy", "medium", "hard"] | None = None
    status: ExerciseStatus = "kb_original"
    # Source provenance. Only ever populated when the exercise is genuinely
    # sourced from a KB block. Never invented.
    document_title: str | None = None
    document_id: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_ids: list[int] = Field(default_factory=list)
    heading_context: list[str] = Field(default_factory=list)
    # Adaptation lineage: the source KB exercise an adapted item comes from.
    original_level: str | None = None
    original_document_title: str | None = None
    original_document_id: int | None = None
    original_page_start: int | None = None
    original_page_end: int | None = None
    original_chunk_ids: list[int] = Field(default_factory=list)
    # Search ranking score (only populated by the search endpoint).
    summary_score: float | None = None


class ExerciseAdaptIn(BaseModel):
    """Explicit adaptation of one KB-sourced exercise towards a target level.

    The source keeps its own level (original_level) and full provenance; the
    result always carries status "adapted_from_kb" and never claims to be a
    verbatim KB extract.
    """

    model_config = ConfigDict(extra="forbid")
    source: ExerciseItem
    target_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
    language: Literal["ar", "fr", "en", "es"] = "ar"
    instructions: str | None = Field(default=None, max_length=1000)


class ExerciseSearchIn(BaseModel):
    """Natural-language exercise search — constraints only when stated."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=12, ge=1, le=50)
    # Optional hard refinements applied on top of the parsed query. The level is
    # a strict filter: a chunk explicitly at another level is never relabelled.
    level: str | None = Field(default=None, pattern="^(A1|A2|B1|B2|C1|C2)$")
    skills: list[str] = Field(default_factory=list, max_length=10)
    exercise_type: str | None = Field(default=None, max_length=100)
    source_document_ids: list[int] = Field(default_factory=list)


class ExerciseSearchItem(ExerciseItem):
    """Search hit with per-item facet labels derived from the document."""
    theme_label: str | None = None
    skill_label: str | None = None


class ExerciseFilterFacet(BaseModel):
    value: str
    count: int


class ExerciseSearchMeta(BaseModel):
    llm_calls: int = 0
    # Stage-level LLM accounting: retrieval must stay 0, extraction can be N.
    retrieval_llm_calls: int = 0
    extraction_llm_calls: int = 0
    retrieval_mode: str | None = None
    dense_candidate_count: int = 0
    sparse_candidate_count: int = 0
    union_candidate_count: int = 0
    candidate_blocks: int = 0
    expanded_blocks: int = 0
    extracted_blocks: int = 0
    deduplicated_count: int = 0
    detected_blocks: int = 0
    parsed: dict[str, object] = Field(default_factory=dict)


class ExerciseSearchOut(BaseModel):
    query: str = Field(min_length=1)
    items: list[ExerciseSearchItem] = Field(default_factory=list)
    total: int = 0
    facets: dict[str, list[ExerciseFilterFacet]] = Field(default_factory=dict)
    meta: ExerciseSearchMeta = Field(default_factory=ExerciseSearchMeta)


class ExercisePlan(BaseModel):
    """Structured pedagogical rationale produced by the PedagogicalPlanner.

    `objective` is a human sentence; `learning_objectives` are per-objective
    goals; `target_vocabulary` / `target_grammar` are the notions the exercises
    should reuse; `distribution` is the ordered list of exercise types used;
    `rationale` explains the progression.
    """
    level: str
    theme: str
    skills: list[str] = Field(default_factory=list)
    objective: str = ""
    learning_objectives: list[str] = Field(default_factory=list)
    target_vocabulary: list[str] = Field(default_factory=list)
    target_grammar: list[str] = Field(default_factory=list)
    exercise_distribution: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExerciseOut(BaseModel):
    title: str = Field(min_length=1)
    level: str
    theme: str
    exercise_type: str
    language: str
    skills: list[str]
    exercises: list[ExerciseItem] = Field(min_length=1)
    # Number of exercises that are directly attributable to a KB source.
    kb_sourced_count: int = 0
    rag_sources_used: int = 0
    provider_model: str | None = None
    adapt_with_ai: bool = False
    # The pedagogical plan produced by the planner before generation. Explains
    # why these exercises take this shape (objectives, target vocabulary &
    # grammar, distribution, progression rationale).
    plan: ExercisePlan | None = None
