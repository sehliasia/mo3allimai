import json
from types import SimpleNamespace

from app.services.llm_provider import LLMProviderError, LLMResult
from app.services.pedagogical_knowledge_service import (
    CEFRSourceProvenance,
    PedagogicalCEFRDescriptor,
    PedagogicalCEFRMissing,
    PedagogicalContext,
    PedagogicalResourceBlock,
)
from app.services.structured_generation_service import (
    StructuredGenerationService, StructuredPromptBuilder,
    StructuredOutputParser,
    ValidationGate,
)


class FakeProvider:
    model_id = "fake-structured"

    def __init__(self, response, *, finish_reason=None, error=None):
        self.response, self.finish_reason, self.error = response, finish_reason, error
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMResult(self.response, self.model_id, self.finish_reason, 42)


def _context(*, descriptor_status="AVAILABLE", include_missing=False):
    descriptor = PedagogicalCEFRDescriptor(
        level="A1", scale="Interaction orale générale", status=descriptor_status,
        descriptor_text="Peut gérer des interactions simples." if descriptor_status == "AVAILABLE" else None,
        reference_level="C1" if descriptor_status == "NO_DESCRIPTOR_AVAILABLE" else None,
        sources=[CEFRSourceProvenance(19, 76, 76, 205, 7)],
    )
    return PedagogicalContext(
        request_summary={"cefr_level": "A1", "language": "fr", "topic": "la famille", "skills": ["speaking"]},
        cefr_descriptors=[descriptor],
        cefr_missing=[PedagogicalCEFRMissing("A1", "Production orale générale")] if include_missing else [],
        resource_blocks=[PedagogicalResourceBlock(
            source_number=1, document_id=15, document_title="Resource", chunk_ids=[42], page_start=4,
            page_end=4, heading_context=["Goals"], content_type="paragraph", structural_quality="structured",
            content="A canonical pedagogical passage.", requires_vision=False, image_not_interpreted=False,
            vector_scores=[0.9], reranker_scores=[None], original_ranks=[1], reranked_ranks=[None],
        )],
        retrieved_count=1, selected_count=1, sources=[], warnings=[], requires_vision_count=0,
    )


def _payload(*, level="A1", language="fr", cefr_ref="cefr:1", claim=None, resource_ref="resource:1"):
    claim = claim if claim is not None else {
        "ref": cefr_ref, "level": "A1", "scale": "Interaction orale générale", "status": "AVAILABLE",
        "descriptor_text": "Peut gérer des interactions simples.", "reference_level": None,
    }
    return {
        "title": "Draft", "pedagogical_summary": {"cefr_level": level, "output_language": language, "objective": None, "skills": ["speaking"]},
        "content_sections": [{"section_type": "overview", "heading": "Objectif", "content": "Contenu fondé.", "source_refs": [resource_ref], "source_basis": "resource"}],
        "cefr_refs": [cefr_ref], "cefr_claims": [claim], "source_refs": [resource_ref], "warnings": [],
    }


def _generate(payload, **provider_kwargs):
    provider = FakeProvider(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False), **provider_kwargs)
    return StructuredGenerationService(llm=provider).generate(_context()), provider


def test_valid_structured_output_uses_closed_world_refs_and_fake_provider():
    result, provider = _generate(_payload())
    assert result.validation.is_valid and result.validated_output is not None and result.parse_succeeded
    assert result.validation.allowed_cefr_refs_count == 1 and result.validation.allowed_resource_refs_count == 1
    assert provider.calls and provider.calls[0]["max_tokens"] > 0


def test_structured_generation_uses_its_own_configured_output_budget():
    provider = FakeProvider(json.dumps(_payload()))
    settings = SimpleNamespace(
        rag_llm_temperature=0.2,
        rag_llm_max_tokens=111,
        structured_generation_max_output_tokens=222,
    )
    result = StructuredGenerationService(llm=provider, settings=settings).generate(_context())
    assert result.validation.is_valid
    assert provider.calls[0]["max_tokens"] == 222
    assert settings.rag_llm_max_tokens == 111


def test_prompt_uses_compact_resource_ids_not_repeated_provenance():
    context = _context()
    registry = ValidationGate.source_registry(context)
    system, user = StructuredPromptBuilder().build(context, registry)
    assert "at most three content sections" in system
    assert '"ref": "resource:1"' in user
    assert "document_id" not in user and "chunk_ids" not in user and "page_start" not in user


def test_fenced_json_is_accepted_but_malformed_json_is_not_repaired():
    valid, _ = _generate("```json\n" + json.dumps(_payload()) + "\n```")
    malformed, _ = _generate('{"pedagogical_summary":')
    assert valid.validation.is_valid
    assert malformed.validation.errors[0].code == "STRUCTURED_OUTPUT_PARSE_ERROR"


def test_truncated_output_never_attempts_to_validate_partial_json():
    result, _ = _generate('{"partial": true}', finish_reason="length")
    assert result.parse_succeeded is False
    assert result.validation.errors[0].code == "LLM_OUTPUT_TRUNCATED"


def test_truncated_valid_looking_json_prefix_is_still_rejected():
    result, _ = _generate(json.dumps(_payload())[:-2], finish_reason="max_tokens")
    assert result.parse_succeeded is False
    assert result.validation.errors[0].code == "LLM_OUTPUT_TRUNCATED"


def test_missing_required_field_and_invalid_level_are_explicit_errors():
    missing = _payload(); del missing["content_sections"]
    parsed = StructuredOutputParser.parse(json.dumps(_payload(level="A2")))
    report = ValidationGate().validate(_context(), parsed)
    result, _ = _generate(missing)
    assert result.validation.errors[0].code == "STRUCTURED_OUTPUT_PARSE_ERROR"
    assert any(issue.code == "INVALID_CEFR_LEVEL" for issue in report.errors)


def test_fabricated_cefr_descriptor_or_scale_is_rejected():
    invented = _payload()
    invented["cefr_claims"][0]["descriptor_text"] = "Peut faire n'importe quoi."
    descriptor_result, _ = _generate(invented)
    invented_scale = _payload(); invented_scale["cefr_claims"][0]["scale"] = "Échelle inventée"
    scale_result, _ = _generate(invented_scale)
    assert any(issue.code == "CEFR_FACT_NOT_IN_CONTEXT" for issue in descriptor_result.validation.errors)
    assert any(issue.code == "CEFR_FACT_NOT_IN_CONTEXT" for issue in scale_result.validation.errors)


def test_fabricated_document_page_chunk_reference_is_rejected():
    payload = _payload(resource_ref="resource:999")
    result, _ = _generate(payload)
    assert any(issue.code == "UNKNOWN_SOURCE_REFERENCE" for issue in result.validation.errors)


def test_cefr_claim_requires_a_cefr_reference_and_resource_section_requires_resource_reference():
    payload = _payload(); payload["cefr_refs"] = []
    first, _ = _generate(payload)
    payload = _payload(); payload["content_sections"][0]["source_refs"] = ["cefr:1"]
    second, _ = _generate(payload)
    assert any(issue.code == "MISSING_REQUIRED_SOURCE" for issue in first.validation.errors)
    assert any(issue.code == "MISSING_REQUIRED_SOURCE" for issue in second.validation.errors)


def test_no_descriptor_available_and_missing_cefr_fact_remain_closed_world_safe():
    context = _context(descriptor_status="NO_DESCRIPTOR_AVAILABLE", include_missing=True)
    payload = _payload(claim={
        "ref": "cefr:1", "level": "A1", "scale": "Interaction orale générale",
        "status": "NO_DESCRIPTOR_AVAILABLE", "descriptor_text": None, "reference_level": "C1",
    })
    parsed = StructuredOutputParser.parse(json.dumps(payload))
    report = ValidationGate().validate(context, parsed)
    registry = ValidationGate.source_registry(context)
    assert report.is_valid and "cefr-missing:1" in registry.cefr_facts
    payload["cefr_claims"][0]["status"] = "AVAILABLE"
    invalid = ValidationGate().validate(context, StructuredOutputParser.parse(json.dumps(payload)))
    assert any(issue.code == "CEFR_FACT_NOT_IN_CONTEXT" for issue in invalid.errors)


def test_output_language_metadata_mismatch_is_error_but_body_language_is_warning_only():
    result, _ = _generate(_payload(language="ar"))
    assert any(issue.code == "OUTPUT_LANGUAGE_MISMATCH" for issue in result.validation.errors)
    valid, _ = _generate(_payload())
    assert any(issue.code == "CONTENT_LANGUAGE_NOT_VERIFIED" for issue in valid.validation.warnings)


def test_registry_and_validation_order_are_deterministic_and_provider_errors_are_safe():
    context = _context(include_missing=True)
    first = ValidationGate.source_registry(context)
    second = ValidationGate.source_registry(context)
    assert list(first.cefr_facts) == list(second.cefr_facts) == ["cefr:1", "cefr-missing:1"]
    provider = FakeProvider("", error=LLMProviderError("safe failure", provider_message="safe provider failure"))
    result = StructuredGenerationService(llm=provider).generate(context)
    assert result.validation.errors[0].code == "LLM_PROVIDER_ERROR"
    assert "safe provider failure" in result.validation.errors[0].message
