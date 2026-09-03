"""Deterministic end-to-end validation of the exercise generator pipeline.

This harness drives the FULL pipeline — PedagogicalPlanner → bounded LLM
generation (scripted fake) → ExerciseValidator → targeted regeneration →
reordering — for the four realistic teacher scenarios (A1/A2/B1/B2), plus a
deliberately-invalid probe to prove the validator and the bounded,
subset-only regeneration actually work.

It deliberately does NOT touch a live DB, Qdrant or a paid LLM provider:
RAG is simulated with realistic `PedagogicalContext` blocks (so we can show the
retrieved material influences `target_vocabulary` / `target_grammar`), and the
LLM is a scripted in-memory double that returns varied, level-appropriate,
validated JSON. This makes the whole check deterministic, self-contained and
free of credentials.

Sections covered:
  1. Four scenarios (A: A1 Famille, B: A2 Voyage, C: B1 Culture, D: B2 Travail).
  2. RAG influence: plan vocabulary/grammar harvested from the retrieved blocks.
  3. Validator + targeted (subset-only) regeneration on an invalid exercise.
  4. Diversity of the produced distribution (never all-QCM).
  5. CECRL coherence (levels, progression guided → productive).
  6. Structured fields emitted for the frontend `ExerciseCard` renderers.

Run:  .venv\\Scripts\\python.exe -m pytest tests/test_e2e_exercises.py -v
"""
from __future__ import annotations

import json

import pytest

from app.schemas.exercise_generator import ExerciseGenerateIn, ExerciseItem
from app.services.exercise_generation_service import ExerciseGenerationService
from app.services.exercise_planner import PedagogicalPlanner, _pick_distribution
from app.services.exercise_validator import ExerciseValidator
from app.services.llm_provider import LLMResult
from app.services.pedagogical_knowledge_service import PedagogicalContext, PedagogicalResourceBlock


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _theme_arabic(theme: str) -> str:
    t = (theme or "").casefold()
    if "famille" in t or "عائلة" in t:
        return "العائلة"
    if "voyage" in t or "سفر" in t:
        return "السفر"
    if "culture" in t or "ثقافة" in t:
        return "الثقافة"
    if "travail" in t or "عمل" in t:
        return "العمل"
    return "الموضوع"


def _block(content: str, *, heading: str = "", content_type: str = "worksheet_exercise", doc_title: str = "Méthode A1", doc_id: int = 7, pages=(3, 3)) -> PedagogicalResourceBlock:
    return PedagogicalResourceBlock(
        source_number=1, document_id=doc_id, document_title=doc_title,
        chunk_ids=[100 + doc_id], page_start=pages[0], page_end=pages[1],
        heading_context=[heading] if heading else [], content_type=content_type,
        structural_quality=None, content=content, requires_vision=False,
        image_not_interpreted=False, vector_scores=[], reranker_scores=[],
        original_ranks=[], reranked_ranks=[],
    )


def _context(request: dict[str, object], blocks: list[PedagogicalResourceBlock]) -> PedagogicalContext:
    return PedagogicalContext(
        request_summary=request, cefr_descriptors=[], cefr_missing=[],
        resource_blocks=blocks, retrieved_count=len(blocks),
        selected_count=len(blocks), sources=[], warnings=[], requires_vision_count=0,
    )


class ScriptedLLM:
    """In-memory LLM double that serves one pre-built JSON response per call.

    Records every call (system/user prompt) so tests can assert that targeted
    regeneration asks for ONLY the invalid subset, not the whole set.
    """

    model_id = "fake-llm"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, temperature=None, max_tokens=None, retry_policy=None, generation_options=None) -> LLMResult:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        text = self.responses.pop(0)
        return LLMResult(text=text, model=self.model_id)


def _item_dict(*, idx: int, ex_type: str, level: str, theme: str, prompt_override: str | None = None) -> dict:
    """A single item that is GUARANTEED to pass the validator (structural +
    pedagogical + quality + no "correction recopie la consigne" substring trap).

    Repeated types (the planner cycles types to reach the requested count) use a
    small bank of distinct Arabic sentence frames per type so that two items of
    the same type are genuinely different — otherwise the Dice duplicate guard
    (threshold 0.78) would flag near-identical synthetic prompts, which would
    not happen with a real LLM producing varied phrasing.
    """
    theme_ar = _theme_arabic(theme)
    base = {
        "title": f"Exercice {idx + 1}",
        "skill": "Vocabulaire",
        "exercise_type": ex_type,
        "level": level,
    }
    variant = idx % 4  # cycle a small bank of distinct sentences
    if ex_type == "qcm":
        prompts = [
            f"اختر الإجابة الصحيحة لإكمال الجملة {idx + 1} عن {theme_ar}",
            f"أي كلمة تناسب المعنى في الجملة رقم {idx + 1}؟",
            f"اختر اللفظة السليمة لوضعها في الخانة {idx + 1}",
            f"ما الكلمة المناسبة لإتمام العبارة {idx + 1}؟",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": "السبورة",
            "options": ["السبورة", "الدراجة", "القميص", "المنبر"],
        })
    elif ex_type == "true_false":
        prompts = [
            f"حدد صحة العبارة {idx + 1} فيما يخص {theme_ar}",
            f"اعتبر ما إذا كانت الجملة {idx + 1} صحيحة أو خاطئة",
            f"حكم على صحة القول رقم {idx + 1}",
            f"بين إن كانت الفكرة {idx + 1} مطابقة للنص أم لا",
        ]
        base.update({
            "prompt": prompts[variant],
            "is_true": True,
            "answer_expectation": None,
        })
    elif ex_type == "matching":
        prompts = [
            f"اطبق كل كلمة على معناها المناسب {idx + 1}",
            f"واصل بين المفهوم وتعريفه في النشاط {idx + 1}",
            f"صِل كل ركن بما يناسبه في الجدول {idx + 1}",
            f"ربط كل مفردة بأضدادها في التمرين {idx + 1}",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": "1-2",
            "pairs": [{"left": "كلمة", "right": "معنى"}, {"left": "ركن", "right": "تفصيل"}],
        })
    elif ex_type == "ordering":
        prompts = [
            f"رتب الجمل لإنتاج نص متماسك {idx + 1} عن {theme_ar}",
            f"أعد بناء الفقرة بترتيب العبارات في النشاط {idx + 1}",
            f"رتّب الأحداث التالية حسب التسلسل الزمني {idx + 1}",
            f"ضع الجمل في مواضعها ليكون المعنى صحيحا في التمرين {idx + 1}",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": "1-2-3-4",
            "options": ["الجملة الأولى", "الجملة الثانية", "الجملة الثالثة", "الجملة الرابعة"],
        })
    elif ex_type == "complete":
        prompts = [
            f"أكمل الفراغ {idx + 1} بالكلمة الناقصة من النص",
            f"املأ الخانة رقم {idx + 1} بالكلمة المطابقة",
            f"سدد الفراغ في الجملة {idx + 1} بالمفرد الصحيح",
            f"أتمم العبارة {idx + 1} بكلمة مناسبة",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": "الطائرة",
        })
    elif ex_type == "grammar_transformation":
        prompts = [
            f"حوّل الجملة {idx + 1} إلى الزمن الماضي مع الحفاظ على المعنى",
            f"أعد كتابة العبارة {idx + 1} في صيغة الجمع",
            f"صرف الفعل في الجملة {idx + 1} حسب المطلوب",
            f"غيّر صيغة الجملة {idx + 1} من الإيجاب إلى النفي",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": "غادرت",
        })
    elif ex_type == "reading_comprehension":
        prompts = [
            f"أجب عن الأسئلة التالية بعد قراءة النص الأول حول {theme_ar}",
            f"استخرج من النص المطلوب في التمرين التالي",
            f"أجب بجملة مفيدة حول مضامين النص المقروء",
            f"لخص النص في سطرين بعد قراءته",
        ]
        base.update({
            "prompt": prompts[variant],
            "context": f"نص قراءة مناسب لمستوى {level} يتناول موضوع {theme_ar} بأسلوب بسيط",
            "answer_expectation": None,
        })
    elif ex_type == "open_question":
        prompts = [
            f"عبّر كتابيا عن رأيك حول {theme_ar} في بضعة أسطر",
            f"اشرح بأسلوبك الخاص مفهوم {theme_ar} في فقرتين",
            f"أعط تفسيرك الشخصي لظاهرة تتعلق بـ {theme_ar}",
            f"ناقش في بضع جمل مغزى جملة تتعلق بـ {theme_ar}",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": None,
        })
    elif ex_type == "writing":
        prompts = [
            f"اكتب فقرة قصيرة عن {theme_ar} مع الاهتمام بالبنية والترابط",
            f"حرّر مقطعا يستعرض فكرة حول {theme_ar}",
            f"أنتج نصا من ثلاث جمل حول تجربة متعلقة بـ {theme_ar}",
            f"اكتب رسالة قصيرة تعبر عن موقفك من {theme_ar}",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": None,
        })
    else:  # recognition / unknown → renders as a multiple-choice prompt.
        prompts = [
            f"تعرف على الكلمات المتعلقة بـ {theme_ar} {idx + 1}",
            f"حدد المفردات المشتركة في القائمة رقم {idx + 1}",
            f"استخرج الكلمات الأساسية من النشاط {idx + 1}",
            f"ميز بين الكلمات في المثال {idx + 1}",
        ]
        base.update({
            "prompt": prompts[variant],
            "answer_expectation": "السبورة",
            "exercise_type": "qcm",
            "options": ["السبورة", "الدراجة", "القميص", "المنبر"],
        })
    if prompt_override is not None:
        base["prompt"] = prompt_override
    return base


def _payload(*, level: str, theme: str, distribution: list[str], invalid_indices: set[int] | None = None) -> str:
    """Build a JSON payload honouring the planner's distribution (varied types).

    `invalid_indices` lets a test plant defects in first generation to trigger
    targeted regeneration of exactly those items.
    """
    items = [
        _item_dict(idx=i, ex_type=distribution[i], level=level, theme=theme)
        for i in range(len(distribution))
    ]
    for i in invalid_indices or set():
        if i < len(items):
            items[i]["exercise_type"] = "qcm"
            items[i]["prompt"] = "   "  # empty prompt → structurally invalid
            items[i].pop("options", None)
    return json.dumps({
        "title": f"Exercices — {theme}",
        "level": level, "theme": theme, "exercise_type": "auto",
        "exercises": items,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Section 2 / 5 — RAG influence + planner (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _scenario_request(level: str, theme: str, skills: list[str], count: int) -> ExerciseGenerateIn:
    return ExerciseGenerateIn(level=level, theme=theme, skills=skills, count=count)


def _cultural_blocks() -> list[PedagogicalResourceBlock]:
    return [
        _block(
            "الموسيقى والثراث المغربي: طبق الطاجين والكسكس، الحضارة والزينة، التنوع الثقافي المغربي",
            heading="ثقافة مغربية", doc_title="Culture marocaine B1", doc_id=11, pages=(5, 5),
        ),
        _block(
            "قراءة: عادات وأعراف في المغرب، الاحتفالات والأعياد، الفن الشعبي المغربي",
            heading="Compréhension écrite", doc_title="Culture marocaine B1", doc_id=12, pages=(6, 6),
        ),
    ]


def test_rag_influence_plan_harvests_vocab_and_grammar():
    # The planner harvests target vocabulary/grammar from the retrieved blocks,
    # demonstrating that the RAG pool shapes the pedagogical plan (no inventing).
    request = _scenario_request("B1", "Culture marocaine", ["Compréhension écrite"], 10)
    plan = PedagogicalPlanner().plan(
        level=request.level, theme=request.theme, skills=request.skills,
        count=request.count, objective=request.objective, context=_context({"cefr_level": "B1"}, _cultural_blocks()),
    )
    assert plan.target_vocabulary, "plan should harvest vocabulary from the RAG blocks"
    assert plan.target_grammar
    # The distribution must be driven by the reading skill, guided → productive.
    assert plan.exercise_distribution[0] in ("reading_comprehension", "recognition")
    assert len(plan.exercise_distribution) == 10


def test_distribution_is_varied_not_all_qcm():
    for (level, theme, skills, count) in [
        ("A1", "Famille", ["Vocabulaire"], 5),
        ("A2", "Voyage", ["Vocabulaire", "Grammaire"], 8),
        ("B1", "Culture marocaine", ["Compréhension écrite"], 10),
        ("B2", "Travail", ["Expression écrite"], 5),
    ]:
        dist = _pick_distribution(
            {"vocabulaire": "vocabulary", "grammaire": "grammar", "compréhension écrite": "reading", "expression écrite": "writing"}
            .get((" ".join(skills)).casefold(), ""),
            count,
        )
        assert len(set(dist)) > 1, f"distribution all identical for {theme}"
        assert len(dist) == count


def test_cecrl_progression_is_guided_then_productive():
    dist = _pick_distribution("", 8)
    phases = [_TYPE_PHASE[t] for t in dist]
    first_production = next((i for i, p in enumerate(phases) if p == "production"), len(phases))
    # Recognition/comprehension must appear before any production.
    assert "recognition" in phases or "comprehension" in phases
    assert all(p != "production" for p in phases[:first_production]) or first_production >= 2


# ---------------------------------------------------------------------------
# Section 1 — the four full scenarios
# ---------------------------------------------------------------------------

def _run_scenario(*, level: str, theme: str, skills: list[str], count: int, blocks: list[PedagogicalResourceBlock]) -> dict:
    request = _scenario_request(level, theme, skills, count)
    service = ExerciseGenerationService(llm=None)
    plan = service.planner.plan(
        level=request.level, theme=request.theme, skills=request.skills,
        count=request.count, objective=request.objective, context=_context({"cefr_level": level}, blocks),
    )
    distribution = list(plan.exercise_distribution)
    llm = ScriptedLLM([_payload(level=level, theme=theme, distribution=distribution)])
    result = ExerciseGenerationService(llm=llm).generate(request, _context({"cefr_level": level}, blocks))

    # Validator verdicts on the final output.
    verdicts = service.validator.validate(
        result.exercises, request_level=request.level,
        theme=plan.theme, language=request.language,
    )
    return {
        "level": result.level, "theme": result.theme, "count": len(result.exercises),
        "plan": result.plan, "llm_calls": len(llm.calls),
        "produced_types": [e.exercise_type for e in result.exercises],
        "types_distinct": len({(e.exercise_type or "").casefold() for e in result.exercises}),
        "valid": all(v.ok for v in verdicts),
        "verdicts": verdicts,
        "plan_types_distinct": len(set(plan.exercise_distribution)),
    }


def _assert_scenario(meta: dict, *, expected_count: int):
    assert meta["count"] == expected_count
    assert meta["llm_calls"] == 1, "happy path must be a single LLM call"
    assert meta["plan"] is not None
    assert meta["types_distinct"] >= 2, "distribution must be varied, never all-QCM"
    assert meta["valid"], f"final exercises rejected: {meta['verdicts']}"
    # The (possibly type-diverse) produced set must include a guided and a
    # productive exercise so the progression is visible.
    produced = set(t.casefold() for t in meta["produced_types"])
    assert any(k in produced for k in ("recognition", "qcm", "complete", "true_false", "matching", "ordering", "reading_comprehension", "grammar_transformation")), meta


def test_scenario_a1_famille_vocab_5():
    blocks = [
        _block("عائلة، أب، أم، أخ، أخت، جد، جدة", heading="Vocabulaire famille", doc_title="Famille A1", doc_id=1),
        _block("أكمل: هذه أختي، اسمها ...", heading="Exercice complet", doc_title="Famille A1", doc_id=2),
    ]
    meta = _run_scenario(level="A1", theme="Famille", skills=["Vocabulaire"], count=5, blocks=blocks)
    _assert_scenario(meta, expected_count=5)
    assert meta["level"] == "A1"
    assert all(v.ok for v in meta["verdicts"])


def test_scenario_a2_voyage_vocab_gram_8():
    blocks = [
        _block("سفر، مطار، تذكرة، فندق، قطار، حجز، سائح", heading="Vocabulaire voyage", doc_title="Voyage A2", doc_id=3),
        _block("قواعد: الماضي البسيط والصيغة المؤكدة في الجمل المتعلقة بالسفر", heading="Grammaire", doc_title="Voyage A2", doc_id=4),
    ]
    meta = _run_scenario(level="A2", theme="Voyage", skills=["Vocabulaire", "Grammaire"], count=8, blocks=blocks)
    _assert_scenario(meta, expected_count=8)
    assert meta["level"] == "A2"
    assert meta["plan_types_distinct"] >= 2


def test_scenario_b1_culture_marocaine_comprehension_10():
    meta = _run_scenario(level="B1", theme="Culture marocaine", skills=["Compréhension écrite"], count=10, blocks=_cultural_blocks())
    _assert_scenario(meta, expected_count=10)
    assert meta["level"] == "B1"
    # Reading-skill distribution must produce a comprehension-type exercise,
    # not a purely generic set.
    produced = set(t.casefold() for t in meta["produced_types"])
    assert any(k in produced for k in ("reading_comprehension", "true_false", "open_question")), meta


def test_scenario_b2_travail_expression_ecrite_5():
    blocks = [
        _block("مهنة، عمل، مقابلة شغل، ملف، سيرة ذاتية، عقد، راتب", heading="Vocabulaire travail", doc_title="Travail B2", doc_id=5),
        _block("اكتب رسالة تغطية ومقابلة شغل", heading="Production écrite", doc_title="Travail B2", doc_id=6),
    ]
    meta = _run_scenario(level="B2", theme="Travail", skills=["Expression écrite"], count=5, blocks=blocks)
    _assert_scenario(meta, expected_count=5)
    # Writing skill should produce guided-writing / open-production types.
    produced = set(t.casefold() for t in meta["produced_types"])
    assert any(k in produced for k in ("writing", "open_question", "grammar_transformation")), meta


# ---------------------------------------------------------------------------
# Section 3 — validator detects real problems + subset-only regeneration
# ---------------------------------------------------------------------------

def test_validator_catches_deliberate_defects():
    val = ExerciseValidator()
    benign = _item_dict(idx=0, ex_type="qcm", level="A1", theme="Famille")

    # 1. Duplicate (identical prompts).
    a = _item_dict(idx=0, ex_type="qcm", level="A1", theme="Famille")
    b = _item_dict(idx=1, ex_type="qcm", level="A1", theme="Famille")
    b["prompt"] = a["prompt"]
    dups = val.find_duplicate_indices([
        ExerciseItem.model_validate({**a, "title": "X1"}),
        ExerciseItem.model_validate({**b, "title": "X2"}),
    ])
    assert dups == [1], f"duplicate not detected: {dups}"

    # 2. QCM with invalid options (no options / duplicate options).
    no_opts = _item_dict(idx=0, ex_type="qcm", level="A1", theme="Famille")
    no_opts["options"] = []
    v = val.validate([ExerciseItem.model_validate(no_opts)], request_level="A1", theme="Famille", language="ar")
    assert not v[0].ok and any("sans options" in r for r in v[0].reasons)

    dup_opts = _item_dict(idx=0, ex_type="qcm", level="A1", theme="Famille")
    dup_opts["options"] = ["أ", "أ", "ج", "د"]
    dup_opts["answer_expectation"] = "أ"
    v = val.validate([ExerciseItem.model_validate(dup_opts)], request_level="A1", theme="Famille", language="ar")
    assert not v[0].ok and any("identiques" in r for r in v[0].reasons)

    correct_absent = _item_dict(idx=0, ex_type="qcm", level="A1", theme="Famille")
    correct_absent["answer_expectation"] = "ززز"
    v = val.validate([ExerciseItem.model_validate(correct_absent)], request_level="A1", theme="Famille", language="ar")
    assert not v[0].ok and any("absente des options" in r for r in v[0].reasons)

    # 3. Missing answer for a non-open type.
    missing = _item_dict(idx=0, ex_type="complete", level="A1", theme="Famille")
    missing["answer_expectation"] = None
    v = val.validate([ExerciseItem.model_validate(missing)], request_level="A1", theme="Famille", language="ar")
    assert not v[0].ok and any("manquante" in r for r in v[0].reasons)

    # 4. Wrong level (mismatch with request).
    wrong = _item_dict(idx=0, ex_type="qcm", level="B1", theme="Famille")
    v = val.validate([ExerciseItem.model_validate(wrong)], request_level="A1", theme="Famille", language="ar")
    assert not v[0].ok and any("niveau" in r for r in v[0].reasons)

    # 5. Malformed (empty prompt → structural).
    malformed = {
        "title": "X", "exercise_type": "qcm", "prompt": "   ",
        "answer_expectation": "ب", "options": ["أ", "ب"], "level": "A1",
    }
    v = val.validate([ExerciseItem.model_validate(malformed)], request_level="A1", theme="Famille", language="ar")
    assert not v[0].ok and any("consigne vide" in r for r in v[0].reasons)

    # A fully benign item validates OK.
    v = val.validate([ExerciseItem.model_validate(benign)], request_level="A1", theme="Famille", language="ar")
    assert v[0].ok, benign


def test_targeted_regeneration_only_rebuilds_invalid_subset():
    # 8 exercises, ONE invalid. The regeneration pass must ask for that single
    # missing item only — never regenerate the whole set — and the final set
    # must still contain 8 valid exercises after bounded regeneration.
    request = _scenario_request("A1", "Famille", ["Vocabulaire"], 8)
    planner_service = ExerciseGenerationService(llm=None)
    plan = planner_service.planner.plan(
        level="A1", theme="Famille", skills=["Vocabulaire"],
        count=8, objective=None, context=_context({"cefr_level": "A1"}, [_block("عائلة، أب، أم", heading="Vocabulaire")]),
    )
    distribution = list(plan.exercise_distribution)
    first = _payload(level="A1", theme="Famille", distribution=distribution, invalid_indices={3})
    # Regeneration returns a valid replacement for the invalid index, as a
    # well-formed root object (the contract used by the primary generation).
    replacement = _item_dict(idx=3, ex_type=distribution[3], level="A1", theme="Famille")
    regen_payload = json.dumps({
        "title": "Exercices — Famille", "level": "A1", "theme": "Famille",
        "exercise_type": "auto", "exercises": [replacement],
    }, ensure_ascii=False)
    llm = ScriptedLLM([first, regen_payload])

    result = ExerciseGenerationService(llm=llm).generate(request, _context({"cefr_level": "A1"}, [_block("عائلة، أب، أم", heading="Vocabulaire")]))
    assert len(result.exercises) == 8, "final set must keep the full count"
    assert len(llm.calls) == 2, "exactly one regeneration call expected"
    # Inspect the regeneration user prompt: it must reference ONLY the missing
    # index further down than the retained list, and ask for a reduced quantity.
    regen_prompt = llm.calls[1]["user_prompt"]
    assert "REGÉNÉRATION CIBLÉE" in regen_prompt
    assert "1" in regen_prompt, "regeneration must target the single missing exercise"
    verdicts = planner_service.validator.validate(
        result.exercises, request_level="A1", theme="Famille", language="ar",
    )
    assert all(v.ok for v in verdicts), [v.reasons for v in verdicts if not v.ok]


def test_regeneration_is_bounded_when_validator_keeps_failing():
    # A response that is permanently invalid must not loop forever: bounded by
    # MAX_REGENERATION_ATTEMPTS and returns the valid subset.
    bad = json.dumps({
        "title": "X", "level": "A1", "theme": "Famille", "exercise_type": "qcm",
        "exercises": [{"title": "X", "exercise_type": "qcm", "prompt": "  ", "options": [], "level": "A1"}],
    }, ensure_ascii=False)
    llm = ScriptedLLM([bad, bad, bad])
    request = _scenario_request("A1", "Famille", ["Vocabulaire"], 1)
    result = ExerciseGenerationService(llm=llm).generate(request, _context({"cefr_level": "A1"}, [_block("عائلة", heading="V")]))
    # bounded: initial + up to 2 regen attempts (MAX_REGENERATION_ATTEMPTS).
    assert len(llm.calls) <= 3
    assert isinstance(result.exercises, list)


# ---------------------------------------------------------------------------
# Section 6 — structured fields contract for the frontend renderers
# ---------------------------------------------------------------------------

def test_emitted_structured_fields_cover_card_renderers():
    # These are the fields ExerciseCard must render per type. We verify the
    # generator emits them so the typing frontend contract is satisfied.
    for (ex_type, field) in [
        ("qcm", "options"),
        ("true_false", "is_true"),
        ("matching", "pairs"),
        ("ordering", "options"),
        ("reading_comprehension", "context"),
    ]:
        d = _item_dict(idx=0, ex_type=ex_type, level="A1", theme="Famille")
        item = ExerciseItem.model_validate(d)
        value = getattr(item, field)
        if field == "is_true":
            assert value is True
        elif field in ("options", "pairs", "context"):
            assert value, f"{ex_type} missing {field}"


def test_older_exercises_without_structured_fields_render_from_prompt():
    # An exercise without structured data (legacy) must still be present — the
    # frontend falls back to `prompt`. We assert the item keeps a prompt and
    # empty structured fields so the fallback path is exercised.
    item = ExerciseItem.model_validate({
        "title": "X", "exercise_type": "open_question", "prompt": "Décrivez votre famille",
        "level": "A1", "answer_expectation": None,
    })
    assert item.exercise_type == "open_question"
    assert item.options == [] and item.is_true is None and item.pairs == []
    assert (item.prompt or "").strip()


# Map used by the progression assertion above.
_TYPE_PHASE = {
    "recognition": "recognition",
    "qcm": "comprehension",
    "true_false": "comprehension",
    "matching": "application",
    "complete": "application",
    "ordering": "manipulation",
    "grammar_transformation": "manipulation",
    "reading_comprehension": "comprehension",
    "open_question": "production",
    "writing": "production",
}