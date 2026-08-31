from types import SimpleNamespace

import pytest

from app.services.context_builder import ContextBuilder
from app.services.pedagogical_knowledge_service import (
    PedagogicalKnowledgeRequest,
    PedagogicalKnowledgeService,
    PedagogicalKnowledgeValidationError,
)
from app.services.retrieval_service import RetrievalResult


def _result(*, chunk_id=42, document_id=15, vision=False, cefr_level="A1"):
    return RetrievalResult(
        rank=1, score=0.9, vector_score=0.9, original_rank=1, chunk_id=chunk_id,
        document_id=document_id, document_title="Pedagogical resource",
        source_page_start=4, source_page_end=4, content_type="paragraph", language="fr",
        cefr_level=cefr_level, structural_quality="structured", has_image=vision,
        requires_vision=vision, heading_context=["Lesson goals"],
        content="A useful, canonical pedagogical source passage.",
    )


def _role_result(*, rank, chunk_id, content, heading_context=None, content_type="paragraph"):
    result = _result(chunk_id=chunk_id)
    return RetrievalResult(
        **{
            **result.__dict__,
            "rank": rank,
            "original_rank": rank,
            "content": content,
            "heading_context": heading_context or [],
            "content_type": content_type,
        }
    )


class FakeCEFR:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.calls = []

    def get_descriptors(self, _db, *, level_code, scale_name):
        self.calls.append((level_code, scale_name))
        return self.rows.get((level_code, scale_name), [])

    @staticmethod
    def get_descriptor_sources(_db, descriptor_id):
        return [SimpleNamespace(document_id=19, page_start=76, page_end=76, chunk_id=205, source_order=7)]


class FakeRetrieval:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, _db, query, **kwargs):
        self.calls.append((query, kwargs))
        return SimpleNamespace(query=query, results=self.results)


def _descriptor(*, status="AVAILABLE", reference_level=None):
    return SimpleNamespace(
        id=1, level=SimpleNamespace(code="A1"), scale=SimpleNamespace(name="Interaction orale générale"),
        status=status,
        descriptor_text="Peut gérer des interactions simples." if status == "AVAILABLE" else None,
        reference_level=SimpleNamespace(code=reference_level) if reference_level else None,
    )


def _service(*, cefr_rows=None, results=None):
    cefr = FakeCEFR(cefr_rows)
    retrieval = FakeRetrieval(results or [])
    service = PedagogicalKnowledgeService(
        cefr=cefr, retrieval=retrieval,
        context_builder=ContextBuilder(max_chunks=6, max_tokens=300, neighbor_expansion=False),
        settings=SimpleNamespace(
            rag_retrieval_top_k=10,
            rag_reranker_enabled=False,
            pedagogical_context_composition_enabled=False,
            pedagogical_context_composition_pool_size=20,
        ),
    )
    return service, cefr, retrieval


def _request(**changes):
    defaults = dict(cefr_level="A1", topic="la famille", skills=("speaking",), language="fr")
    defaults.update(changes)
    return PedagogicalKnowledgeRequest(**defaults)


def test_available_cefr_descriptor_is_authoritative_and_keeps_provenance():
    service, _cefr, _retrieval = _service(
        cefr_rows={("A1", "Interaction orale générale"): [_descriptor()]}, results=[_result()]
    )
    context = service.build_context(None, _request())
    descriptor = context.cefr_descriptors[0]
    assert descriptor.level == "A1" and descriptor.status == "AVAILABLE"
    assert descriptor.descriptor_text == "Peut gérer des interactions simples."
    assert descriptor.sources[0].document_id == 19 and descriptor.sources[0].chunk_id == 205
    assert context.sources[0].source_type == "cefr_structured"
    assert any(source.source_type == "pedagogical_resource" for source in context.sources)


def test_no_descriptor_available_remains_distinct_from_missing():
    row = _descriptor(status="NO_DESCRIPTOR_AVAILABLE", reference_level="C1")
    service, _cefr, _retrieval = _service(
        cefr_rows={("A1", "Interaction orale générale"): [row]}
    )
    context = service.build_context(None, _request(include_resources=False))
    assert context.cefr_descriptors[0].status == "NO_DESCRIPTOR_AVAILABLE"
    assert context.cefr_descriptors[0].reference_level == "C1"
    assert "Interaction orale générale" not in {item.scale for item in context.cefr_missing}


def test_missing_exact_cefr_scale_does_not_block_resource_retrieval():
    service, _cefr, retrieval = _service(results=[_result()])
    context = service.build_context(None, _request())
    assert context.cefr_descriptors == []
    assert "Interaction orale générale" in {item.scale for item in context.cefr_missing}
    assert context.selected_count == 1 and retrieval.calls


def test_invalid_cefr_level_is_rejected_without_calling_dependencies():
    service, cefr, retrieval = _service()
    with pytest.raises(PedagogicalKnowledgeValidationError, match="Unsupported CEFR level"):
        service.build_context(None, _request(cefr_level="A0"))
    assert cefr.calls == [] and retrieval.calls == []


def test_request_without_cefr_level_skips_structured_lookup_without_inventing_a_default():
    service, cefr, retrieval = _service(results=[_result()])
    context = service.build_context(None, PedagogicalKnowledgeRequest(cefr_level=None, topic="la famille"))
    assert cefr.calls == []
    assert "CEFR level" not in retrieval.calls[0][0]
    assert context.request_summary["cefr_level"] is None


def test_pedagogical_level_and_language_are_not_source_filters_but_document_scope_is():
    service, cefr, retrieval = _service(results=[_result(document_id=8)])
    service.build_context(None, _request(source_document_ids=(8,), retrieval_top_k=3))
    assert {level for level, _scale in cefr.calls} == {"A1"}
    filters = retrieval.calls[0][1]["filters"]
    assert filters.document_ids == [8]
    assert filters.language is None and filters.cefr_level is None
    assert retrieval.calls[0][1]["top_k"] == 3


def test_explicit_source_metadata_filters_are_forwarded_without_changing_pedagogical_request():
    service, _cefr, retrieval = _service(results=[])
    service.build_context(None, _request(source_language="fr", source_cefr_level="A1"))
    filters = retrieval.calls[0][1]["filters"]
    assert filters.language == "fr" and filters.cefr_level == "A1"
    query = retrieval.calls[0][0]
    assert "niveau A1" in query and "enseignement de l'arabe" in query
    assert "CEFR level A1" not in query and "language: fr" not in query


def test_include_flags_are_independent_and_no_skill_does_not_broaden_cefr_lookup():
    service, cefr, retrieval = _service(results=[_result()])
    context = service.build_context(None, _request(skills=(), include_resources=False))
    assert context.resource_blocks == [] and context.retrieved_count == 0
    assert cefr.calls == [] and retrieval.calls == []
    assert "No mapped skill was supplied; structured CEFR lookup was not broadened." in context.warnings


def test_resource_context_marks_vision_without_interpreting_it_and_is_deterministic():
    service, _cefr, _retrieval = _service(results=[_result(chunk_id=9, vision=True), _result(chunk_id=10)])
    first = service.build_context(None, _request(include_cefr=False))
    second = service.build_context(None, _request(include_cefr=False))
    assert first.requires_vision_count == 1
    assert first.resource_blocks[0].image_not_interpreted is True
    assert [block.chunk_ids for block in first.resource_blocks] == [block.chunk_ids for block in second.resource_blocks]


def test_missing_source_metadata_does_not_create_a_false_cefr_conflict_warning():
    service, _cefr, _retrieval = _service(
        cefr_rows={("A1", "Interaction orale générale"): [_descriptor()]},
        results=[_result(cefr_level=None)],
    )
    context = service.build_context(None, _request())
    assert not any("remain authoritative" in warning for warning in context.warnings)


def test_semantic_query_keeps_request_signals_and_unknown_skill_is_a_warning():
    service, _cefr, retrieval = _service(results=[])
    service.build_context(None, _request(
        objective="parler de sa famille", activity_type="discussion", skills=("speaking", "unknown-skill")
    ))
    query = retrieval.calls[0][0]
    assert "enseignement de l'arabe" in query and "activité pédagogique" in query
    assert "niveau A1" in query and "thème: la famille" in query
    assert "objective: parler de sa famille" in query and "activity: discussion" in query
    assert "expression orale" in query and "interaction orale" in query


def test_structured_cefr_remains_authoritative_when_semantic_metadata_conflicts():
    service, _cefr, _retrieval = _service(
        cefr_rows={("A1", "Interaction orale générale"): [_descriptor()]},
        results=[_result(cefr_level="A2")],
    )
    context = service.build_context(None, _request())
    assert context.cefr_descriptors[0].level == "A1"
    assert any("remain authoritative" in warning for warning in context.warnings)


def test_resource_query_uses_current_message_as_topic_signal_without_inventing_a_level():
    service, _cefr, retrieval = _service(results=[])

    service.build_context(None, PedagogicalKnowledgeRequest(
        cefr_level=None,
        topic="Propose une activité sur la famille.",
        skills=("speaking",),
        language="fr",
    ))

    query = retrieval.calls[0][0]
    assert "thème: Propose une activité sur la famille." in query
    assert "niveau A1" not in query and "CEFR level" not in query
    assert "language: fr" not in query


def test_role_play_softly_prefers_dialogue_and_task_evidence_within_semantic_rank_band():
    service, _cefr, _retrieval = _service()
    results = [
        _role_result(rank=1, chunk_id=1, content="Guide méthodologique pour organiser une séance."),
        _role_result(rank=2, chunk_id=2, content="حوار: المسافر يشرح المشكلة ويطلب معلومات."),
        _role_result(rank=3, chunk_id=3, content="Consigne : répartissez les rôles et négociez une solution."),
        _role_result(rank=4, chunk_id=4, content="Activité sans rapport sur les couleurs."),
    ]

    prioritized = service._soft_prioritize_pedagogical_roles(results, "role_play")

    assert [result.chunk_id for result in prioritized] == [2, 3, 1, 4]
    assert service._pedagogical_role(prioritized[0]) == "DIALOGUE"
    assert service._pedagogical_role(prioritized[1]) == "TASK"
    assert [result.rank for result in prioritized] == [2, 3, 1, 4]


def test_activity_context_uses_role_priority_without_changing_cefr_or_source_provenance():
    results = [
        _role_result(rank=1, chunk_id=1, content="Guide méthodologique général."),
        _role_result(
            rank=2,
            chunk_id=2,
            content="تمرين 1 : حوار قصير بين متعلمين حول الأسرة.",
            content_type="worksheet_exercise",
        ),
    ]
    service, _cefr, _retrieval = _service(
        cefr_rows={("A1", "Interaction orale générale"): [_descriptor()]},
        results=results,
    )

    context = service.build_context(
        None,
        _request(topic="Propose une activité orale A1 sur la famille."),
    )

    assert context.request_summary["pedagogical_request_intent"] == "concrete_activity"
    assert context.cefr_descriptors[0].level == "A1"
    assert context.resource_blocks[0].chunk_ids == [2, 1]
    assert context.sources[-1].document_id == 15
    assert context.sources[-1].chunk_ids == [2, 1]


def test_exercise_and_listening_intents_prefer_relevant_concrete_roles_without_hard_filtering():
    service, _cefr, _retrieval = _service()
    results = [
        _role_result(rank=1, chunk_id=1, content="Référence pédagogique générale."),
        _role_result(rank=2, chunk_id=2, content="تمرين 1 : أكمل الحوار.", content_type="worksheet_exercise"),
        _role_result(rank=3, chunk_id=3, content="Dialogue à écouter puis questions de compréhension."),
    ]

    exercise = service._soft_prioritize_pedagogical_roles(results, "concrete_exercise")
    listening = service._soft_prioritize_pedagogical_roles(results, "listening_activity")

    assert exercise[0].chunk_id == 2
    assert listening[0].chunk_id == 3
    assert {result.chunk_id for result in exercise} == {1, 2, 3}


def test_methodology_and_general_requests_do_not_force_activity_evidence():
    service, _cefr, _retrieval = _service()
    results = [
        _role_result(rank=1, chunk_id=1, content="Guide méthodologique pour enseigner le vocabulaire."),
        _role_result(rank=2, chunk_id=2, content="تمرين : reliez les mots."),
    ]

    assert service._request_intent(_request(topic="Comment enseigner le vocabulaire ?")) == "methodology"
    assert service._soft_prioritize_pedagogical_roles(results, "methodology")[0].chunk_id == 1
    assert service._soft_prioritize_pedagogical_roles(results, "general") == results


def test_request_intent_is_deterministic_and_does_not_hardcode_documents():
    service, _cefr, _retrieval = _service()

    assert service._request_intent(_request(topic="Propose un jeu de rôle B1.")) == "role_play"
    assert service._request_intent(_request(topic="Crée un exercice A1.")) == "concrete_exercise"
    assert service._request_intent(_request(topic="Prépare une activité de compréhension orale A2.")) == "listening_activity"
    assert service._request_intent(_request(topic="Explique-moi cette démarche.")) == "general"
