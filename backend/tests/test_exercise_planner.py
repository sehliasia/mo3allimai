"""Tests for the PedagogicalPlanner."""
from app.services.exercise_planner import PedagogicalPlanner, _is_arabic_word, _pick_distribution
from app.services.exercise_prompts import build_task_section
from app.services.pedagogical_knowledge_service import PedagogicalContext, PedagogicalResourceBlock


def _block(content="ا", heading="Vocabulaire"):
    return PedagogicalResourceBlock(
        source_number=1, document_id=1, document_title="Doc",
        chunk_ids=[1], page_start=1, page_end=1,
        heading_context=[heading], content_type="text", structural_quality=None,
        content=content, requires_vision=False, image_not_interpreted=False,
        vector_scores=[], reranker_scores=[], original_ranks=[], reranked_ranks=[],
    )


def _context(*blocks_data):
    return PedagogicalContext(
        {"cefr_level": "A1", "language": "ar"}, [], [], blocks_data,
        len(blocks_data), len(blocks_data), [], [], 0,
    )


def _plan(planner, *, level="A1", theme="Famille", skills=("Vocabulaire",), count=5, objective=None, context=None, language=""):
    return planner.plan(
        level=level, theme=theme, skills=skills, count=count,
        objective=objective, context=context, language=language,
    )


def test_plan_produces_structured_plan():
    planner = PedagogicalPlanner()
    plan = _plan(planner)
    assert plan.level == "A1"
    assert plan.theme == "Famille"
    assert plan.skills == ["Vocabulaire"]
    assert len(plan.exercise_distribution) == 5
    assert plan.learning_objectives
    assert plan.rationale


def test_distribution_length_matches_requested_count():
    planner = PedagogicalPlanner()
    for count in (5, 8, 10, 15):
        plan = _plan(planner, count=count)
        assert len(plan.exercise_distribution) == count


def test_planner_caps_runaway_count():
    planner = PedagogicalPlanner()
    plan = _plan(planner, count=200)
    assert len(plan.exercise_distribution) == 15


def test_planner_honours_explicit_objective():
    planner = PedagogicalPlanner()
    plan = _plan(planner, objective="Travailler la famille élargie.")
    assert "Travailler la famille élargie." in plan.objective


def test_distribution_guided_to_productive():
    # The default sequence starts recognitive/guided and ends productive.
    seq = _pick_distribution("vocabulary", 10)
    assert seq
    assert seq[0] == "recognition"
    assert seq[-1] == "writing"


def test_writing_skill_prefers_writing_types():
    seq = _pick_distribution("writing", 6)
    assert "writing" in seq and "open_question" in seq


def test_planner_harvests_targets_from_kb_blocks():
    planner = PedagogicalPlanner()
    context = _context(_block(content="الأسرة والأب والأم", heading="Vocabulaire"))
    plan = _plan(planner, context=context)
    assert plan.target_vocabulary or plan.target_grammar


def test_plan_rationale_mentions_cefr_level():
    planner = PedagogicalPlanner()
    plan = _plan(planner, level="A2")
    assert "A2" in plan.rationale


def test_generic_request_theme_is_not_leaked_as_topic():
    # "Je veux des exercices de niveau A2" must NOT become the theme.
    planner = PedagogicalPlanner()
    plan = planner.plan(
        level="A2", theme="Je veux des exercices de niveau A2",
        skills=("Vocabulaire",), count=5, objective=None, context=None, language="ar",
    )
    assert "Je veux" not in plan.theme
    assert "niveau" not in plan.theme
    # Without RAG blocks, it falls back to a familiar, concrete Arabic theme.
    assert plan.theme


def test_explicit_theme_is_preserved():
    planner = PedagogicalPlanner()
    plan = planner.plan(
        level="A2", theme="Voyage", skills=("Vocabulaire",), count=5,
        objective=None, context=None, language="ar",
    )
    assert plan.theme == "Voyage"


def test_rag_theme_keyword_is_used():
    planner = PedagogicalPlanner()
    context = _context(_block(content="أذهب إلى المدرسة كل صباح وأتعلم دروسي مع الأستاذ", heading="Les sons de l'école"))
    plan = planner.plan(
        level="A2", theme="Je veux des exercices", skills=(), count=5,
        objective=None, context=context, language="ar",
    )
    assert plan.theme == "المدرسة"


def test_plan_rationale_is_neutral_for_arabic_language():
    planner = PedagogicalPlanner()
    # Arabic request → the French skill/CEFR annotations are dropped, the typo
    # "compréhension écrit" must never appear and the rationale stays neutral.
    plan = _plan(planner, level="A2", skills=("Compréhension écrite",), language="ar")
    assert "compréhension écrit" not in plan.rationale
    assert "A2" not in plan.rationale


def test_vocabulary_is_arabic_only_for_arabic_language():
    planner = PedagogicalPlanner()
    # Latin/French tokens must never leak into an Arabic plan's targets.
    context = _context(_block(content="partir constat paru nécessaire concevoir المدرسة والأستاذ الدرس", heading="Vocabulaire"))
    plan = _plan(planner, language="ar", context=context)
    assert plan.target_vocabulary or plan.target_grammar
    for token in plan.target_vocabulary:
        assert _is_arabic_word(token), f"non-Arabic token leaked: {token!r}"
    for token in plan.target_grammar:
        assert _is_arabic_word(token), f"non-Arabic grammar leaked: {token!r}"


def test_arabic_vocabulary_tokens_are_preserved():
    planner = PedagogicalPlanner()
    context = _context(_block(content="الطلاب يذهبون إلى المدرسة كل صباح", heading="Vocabulaire"))
    plan = _plan(planner, language="ar", context=context)
    joined = " ".join(plan.target_vocabulary)
    assert "المدرسة" in joined or "الطلاب" in joined or "يذهبون" in joined


def test_french_theme_does_not_force_arabic_grammar_leak():
    planner = PedagogicalPlanner()
    # Mixed French + Arabic block with the French conjunction "mais" must not
    # make "mais" the Arabic plan's target grammar.
    context = _context(_block(content="mais je و الطلاب إلى المدرسة", heading="Grammaire"))
    plan = _plan(planner, language="ar", context=context)
    for token in plan.target_grammar:
        assert _is_arabic_word(token), f"French grammar leaked: {token!r}"
    assert "mais" not in plan.target_grammar


def test_task_section_respects_plan_targets_for_arabic():
    planner = PedagogicalPlanner()
    plan = _plan(planner, language="ar", level="A1", theme="المدرسة", skills=("Vocabulaire",), count=5)
    request = {
        "count": 5, "theme": plan.theme, "objective": plan.objective,
        "skills": list(plan.skills), "language": "ar",
        "special_instructions": "", "forced_type": "",
    }
    text = build_task_section(request, plan, plan.exercise_distribution)
    # The plan's target vocabulary/grammar is surfaced as priority, not optional.
    assert "vocabulaire cible" in text
    if plan.target_grammar:
        assert "grammaire du plan" in text
    # Arabic-only mandate + no invented grammar.
    assert "QUALITÉ ARABE" in text
    assert "N'invente PAS de nouvelle notion grammaticale" in text
