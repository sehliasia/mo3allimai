from __future__ import annotations

from app.services.pedagogical_retrieval_ranker import PedagogicalRankingRequest, PedagogicalRetrievalRanker
from app.services.retrieval_service import RetrievalResult


def _result(rank: int, *, chunk_id: int | None = None, content_type: str = "paragraph", content: str = "reference", heading: list[str] | None = None, level: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=.02, vector_score=.8, original_rank=rank, chunk_id=chunk_id or rank,
        document_id=1, document_title="Source", source_page_start=1, source_page_end=1,
        content_type=content_type, language="fr", cefr_level=level, structural_quality="structured",
        has_image=False, requires_vision=False, heading_context=heading or [], content=content,
        appeared_in_dense=True, dense_rank=rank, fused_rank=rank, rrf_score=.02,
    )


def test_exercise_and_dialogue_intents_promote_matching_roles_deterministically():
    ranker = PedagogicalRetrievalRanker()
    reference = _result(2, content="generic reference")
    exercise = _result(3, content_type="worksheet_exercise", content="Exercice guidé")
    ranked = ranker.rank([reference, exercise], PedagogicalRankingRequest(intent="exercise"))
    assert ranked[0].chunk_id == exercise.chunk_id
    assert ranked[0].role_adjustment > 0 and ranked[0].concreteness_adjustment > 0
    dialogue = _result(3, chunk_id=30, content="Dialogue en binômes")
    assert ranker.rank([reference, dialogue], PedagogicalRankingRequest(intent="dialogue"))[0].chunk_id == 30


def test_role_play_activity_methodology_and_general_have_bounded_expected_behavior():
    ranker = PedagogicalRetrievalRanker()
    activity = _result(2, content="Activité de classe")
    methodology = _result(2, content="Guide méthodologique")
    activity_ranked = ranker.rank([_result(1), activity], PedagogicalRankingRequest(intent="activity"))
    methodology_ranked = ranker.rank([_result(1), methodology], PedagogicalRankingRequest(intent="methodology"))
    assert next(item for item in activity_ranked if item.chunk_id == activity.chunk_id).role_adjustment > 0
    assert next(item for item in methodology_ranked if item.chunk_id == methodology.chunk_id).role_adjustment > 0
    general = ranker.rank([activity], PedagogicalRankingRequest(intent="general"))[0]
    assert general.role_adjustment == general.concreteness_adjustment == 0.0


def test_levels_and_skills_use_only_explicit_reliable_evidence():
    ranker = PedagogicalRetrievalRanker()
    exact = ranker.rank([_result(1, level="A1")], PedagogicalRankingRequest(cefr_level="A1"))[0]
    distant = ranker.rank([_result(1, level="B2")], PedagogicalRankingRequest(cefr_level="A1"))[0]
    unknown = ranker.rank([_result(1, content="This paragraph says A1 but has no metadata")], PedagogicalRankingRequest(cefr_level="A1"))[0]
    listening = ranker.rank([_result(1, heading=["Compréhension orale"])], PedagogicalRankingRequest(skills=("listening",)))[0]
    assert exact.level_adjustment > 0 and distant.level_adjustment < 0 and unknown.level_adjustment == 0
    assert listening.skill_adjustment > 0


def test_rank_twenty_concrete_item_cannot_jump_over_rank_one_only_from_adjustments():
    ranker = PedagogicalRetrievalRanker()
    top = _result(1, content="relevant reference")
    weak = _result(20, content_type="worksheet_exercise", content="Exercice")
    ranked = ranker.rank([top, weak], PedagogicalRankingRequest(intent="exercise", cefr_level="A1", skills=("listening",)))
    assert [item.chunk_id for item in ranked] == [1, 20]
    assert top.rrf_score == .02 and weak.fused_rank == 20


def test_ranker_preserves_provenance_and_is_deterministic():
    ranker = PedagogicalRetrievalRanker()
    items = [_result(2, chunk_id=20, content="Activity"), _result(2, chunk_id=10, content="Activity")]
    first = ranker.rank(items, PedagogicalRankingRequest(intent="activity"))
    second = ranker.rank(items, PedagogicalRankingRequest(intent="activity"))
    assert [item.chunk_id for item in first] == [10, 20] == [item.chunk_id for item in second]
    assert all(item.appeared_in_dense and item.dense_rank == 2 and item.fused_rank == 2 for item in first)


def test_concreteness_requires_positive_classroom_evidence_not_only_role_or_source_name():
    ranker = PedagogicalRetrievalRanker()
    worksheet = _result(1, content_type="worksheet_exercise")
    dialogue = _result(1, content="Enseignant: Bonjour.\nApprenant: Bonjour.")
    procedure = _result(1, content="Consigne : travaillez en binômes et répondez aux questions.")
    generic_task = _result(1, content_type="table", content="Peut accomplir une tâche dans un contexte familier.")
    generic_activity = _result(1, content="Activité : Peut participer à des échanges simples.")
    concrete_table = _result(1, content_type="table", content="1. Écoutez puis répondez aux questions.")
    assert ranker.concreteness(worksheet)[0]
    assert ranker.concreteness(dialogue)[0]
    assert ranker.concreteness(procedure)[0]
    assert not ranker.concreteness(generic_task)[0]
    assert not ranker.concreteness(generic_activity)[0]
    assert ranker.concreteness(concrete_table)[0]


def test_skill_signals_do_not_turn_listening_into_speaking_and_support_arabic():
    ranker = PedagogicalRetrievalRanker()
    listening = _result(1, heading=["Compréhension de l'oral"])
    speaking = _result(1, heading=["Production orale"])
    arabic_listening = _result(1, heading=["فهم المسموع"])
    assert ranker.rank([listening], PedagogicalRankingRequest(skills=("listening",)))[0].skill_adjustment > 0
    assert ranker.rank([listening], PedagogicalRankingRequest(skills=("speaking",)))[0].skill_adjustment < 0
    assert ranker.rank([speaking], PedagogicalRankingRequest(skills=("speaking",)))[0].skill_adjustment > 0
    assert ranker.rank([arabic_listening], PedagogicalRankingRequest(skills=("listening",)))[0].skill_adjustment > 0


def test_generic_task_and_late_concrete_item_cannot_rescue_weak_h3_relevance():
    ranker = PedagogicalRetrievalRanker()
    dialogue = _result(2, chunk_id=2, content="Enseignant: Bonjour.\nApprenant: Bonjour.")
    generic_task = _result(1, chunk_id=1, content="Peut accomplir une tâche de communication.")
    role_play = ranker.rank([generic_task, dialogue], PedagogicalRankingRequest(intent="role_play"))
    assert role_play[0].chunk_id == 2
    late = _result(20, chunk_id=20, content_type="worksheet_exercise", content="Exercice : répondez.")
    top = _result(1, chunk_id=1, content="Référence fortement pertinente")
    assert [item.chunk_id for item in ranker.rank([top, late], PedagogicalRankingRequest(intent="exercise"))] == [1, 20]


def test_serialized_enumeration_and_generic_profiles_are_rejected_with_diagnostics():
    ranker = PedagogicalRetrievalRanker()
    samples = [
        _result(1, content_type="table", content="Figure 9 - Profil de compétence"),
        _result(1, content_type="table", content="Personnel, 1 = contexte privé"),
        _result(1, content_type="table", content="A1 = utilisateur élémentaire"),
        _result(1, content="Peut participer à une activité simple."),
    ]
    for sample in samples:
        concrete, reasons = ranker.concreteness(sample)
        assert not concrete
        assert reasons
    real_steps = _result(1, content="1. Écoutez le dialogue.\n2. Répondez aux questions.")
    assert ranker.concreteness(real_steps) == (True, ("numbered_procedural_steps",))


def test_multiskill_full_chunk_is_neutral_without_heading_but_heading_is_local_priority():
    ranker = PedagogicalRetrievalRanker()
    multi = _result(1, content="Compréhension de l'oral, production orale, lecture et production écrite.")
    neutral = ranker.rank([multi], PedagogicalRankingRequest(skills=("speaking",)))[0]
    assert neutral.skill_adjustment == 0 and neutral.skill_evidence_reason == "ambiguous_multiskill_chunk"
    local = _result(1, heading=["Production orale"], content=multi.content)
    matched = ranker.rank([local], PedagogicalRankingRequest(skills=("speaking",)))[0]
    assert matched.skill_adjustment > 0 and matched.skill_evidence_reason == "heading_skill_signal"
