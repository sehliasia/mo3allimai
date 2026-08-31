"""Generic JSON generation and closed-world validation for future pedagogy tools.

The module has no generator-specific business logic.  It accepts a completed
``PedagogicalContext`` and returns either a validated neutral draft or an
explicit, safe validation report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings, get_settings
from app.services.llm_provider import LLMProvider, LLMProviderError, LLMResult
from app.services.pedagogical_knowledge_service import PedagogicalContext


IssueCode = Literal[
    "SCHEMA_ERROR", "INVALID_CEFR_LEVEL", "CEFR_FACT_NOT_IN_CONTEXT",
    "UNKNOWN_SOURCE_REFERENCE", "MISSING_REQUIRED_SOURCE", "OUTPUT_LANGUAGE_MISMATCH",
    "LLM_OUTPUT_TRUNCATED", "STRUCTURED_OUTPUT_PARSE_ERROR", "LLM_PROVIDER_ERROR",
    "CONTENT_LANGUAGE_NOT_VERIFIED",
]
Severity = Literal["ERROR", "WARNING"]


@dataclass(frozen=True)
class ValidationIssue:
    code: IssueCode
    severity: Severity
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    allowed_cefr_refs_count: int
    allowed_resource_refs_count: int


@dataclass(frozen=True)
class PedagogicalSummary:
    cefr_level: str
    output_language: str
    objective: str | None
    skills: list[str]


@dataclass(frozen=True)
class ContentSection:
    section_type: str
    heading: str | None
    content: str
    source_refs: list[str]
    source_basis: Literal["cefr", "resource"] | None = None


@dataclass(frozen=True)
class CEFRFactClaim:
    ref: str
    level: str
    scale: str
    status: str
    descriptor_text: str | None
    reference_level: str | None


@dataclass(frozen=True)
class StructuredPedagogicalOutput:
    """Neutral draft schema shared by future pedagogical generators."""

    title: str | None
    pedagogical_summary: PedagogicalSummary
    content_sections: list[ContentSection]
    cefr_refs: list[str]
    cefr_claims: list[CEFRFactClaim]
    source_refs: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ValidatedPedagogicalOutput:
    output: StructuredPedagogicalOutput
    validation: ValidationReport
    provider_model: str
    finish_reason: str | None
    output_token_count: int | None


@dataclass(frozen=True)
class StructuredGenerationResult:
    validated_output: ValidatedPedagogicalOutput | None
    validation: ValidationReport
    provider_model: str | None
    finish_reason: str | None
    output_token_count: int | None
    parse_succeeded: bool


@dataclass(frozen=True)
class AllowedCEFRFact:
    ref: str
    level: str
    scale: str
    status: str
    descriptor_text: str | None
    reference_level: str | None


@dataclass(frozen=True)
class AllowedResource:
    ref: str
    document_id: int
    chunk_ids: list[int]
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class AllowedSourceRegistry:
    cefr_facts: dict[str, AllowedCEFRFact]
    resources: dict[str, AllowedResource]

    @property
    def source_refs(self) -> set[str]:
        return set(self.cefr_facts) | set(self.resources)


class StructuredOutputParser:
    """Strict JSON parsing, with only a single surrounding Markdown fence removed."""

    _TOP_LEVEL_KEYS = {
        "title", "pedagogical_summary", "content_sections", "cefr_refs", "cefr_claims", "source_refs", "warnings"
    }

    @staticmethod
    def _issue(message: str, path: str | None = None) -> ValueError:
        return ValueError(f"{path + ': ' if path else ''}{message}")

    @staticmethod
    def _list_of_strings(value: Any, path: str) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise StructuredOutputParser._issue("must be a list of strings", path)
        return value

    @staticmethod
    def _optional_string(value: Any, path: str) -> str | None:
        if value is not None and not isinstance(value, str):
            raise StructuredOutputParser._issue("must be a string or null", path)
        return value

    @classmethod
    def _unfence(cls, raw: str) -> str:
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) < 3 or lines[0].strip().casefold() not in {"```", "```json"}:
                raise cls._issue("unsupported Markdown fence")
            return "\n".join(lines[1:-1]).strip()
        return text

    @classmethod
    def parse(cls, raw: str) -> StructuredPedagogicalOutput:
        try:
            value = json.loads(cls._unfence(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise cls._issue("invalid JSON") from exc
        if not isinstance(value, dict):
            raise cls._issue("top-level value must be an object")
        unknown = set(value) - cls._TOP_LEVEL_KEYS
        if unknown:
            raise cls._issue("unexpected top-level fields: " + ", ".join(sorted(unknown)))
        required = {"pedagogical_summary", "content_sections", "cefr_refs", "source_refs", "warnings"}
        missing = required - set(value)
        if missing:
            raise cls._issue("missing required fields: " + ", ".join(sorted(missing)))
        summary = value["pedagogical_summary"]
        if not isinstance(summary, dict):
            raise cls._issue("must be an object", "pedagogical_summary")
        if set(summary) - {"cefr_level", "output_language", "objective", "skills"}:
            raise cls._issue("unexpected fields", "pedagogical_summary")
        if not isinstance(summary.get("cefr_level"), str) or not isinstance(summary.get("output_language"), str):
            raise cls._issue("cefr_level and output_language are required strings", "pedagogical_summary")
        pedagogical_summary = PedagogicalSummary(
            cefr_level=summary["cefr_level"], output_language=summary["output_language"],
            objective=cls._optional_string(summary.get("objective"), "pedagogical_summary.objective"),
            skills=cls._list_of_strings(summary.get("skills"), "pedagogical_summary.skills"),
        )
        if not isinstance(value["content_sections"], list):
            raise cls._issue("must be a list", "content_sections")
        sections: list[ContentSection] = []
        for index, section in enumerate(value["content_sections"]):
            path = f"content_sections[{index}]"
            if not isinstance(section, dict) or set(section) - {"section_type", "heading", "content", "source_refs", "source_basis"}:
                raise cls._issue("invalid section object", path)
            if not isinstance(section.get("section_type"), str) or not isinstance(section.get("content"), str):
                raise cls._issue("section_type and content are required strings", path)
            basis = section.get("source_basis")
            if basis not in {None, "cefr", "resource"}:
                raise cls._issue("source_basis must be cefr, resource, or null", path)
            sections.append(ContentSection(
                section_type=section["section_type"], heading=cls._optional_string(section.get("heading"), f"{path}.heading"),
                content=section["content"], source_refs=cls._list_of_strings(section.get("source_refs"), f"{path}.source_refs"),
                source_basis=basis,
            ))
        claims: list[CEFRFactClaim] = []
        raw_claims = value.get("cefr_claims", [])
        if not isinstance(raw_claims, list):
            raise cls._issue("must be a list", "cefr_claims")
        for index, claim in enumerate(raw_claims):
            path = f"cefr_claims[{index}]"
            required_claim = {"ref", "level", "scale", "status", "descriptor_text", "reference_level"}
            if not isinstance(claim, dict) or set(claim) != required_claim:
                raise cls._issue("must contain exactly the CEFR fact fields", path)
            if any(not isinstance(claim[key], str) for key in ("ref", "level", "scale", "status")):
                raise cls._issue("ref, level, scale, and status must be strings", path)
            claims.append(CEFRFactClaim(
                ref=claim["ref"], level=claim["level"], scale=claim["scale"], status=claim["status"],
                descriptor_text=cls._optional_string(claim["descriptor_text"], f"{path}.descriptor_text"),
                reference_level=cls._optional_string(claim["reference_level"], f"{path}.reference_level"),
            ))
        return StructuredPedagogicalOutput(
            title=cls._optional_string(value.get("title"), "title"), pedagogical_summary=pedagogical_summary,
            content_sections=sections, cefr_refs=cls._list_of_strings(value["cefr_refs"], "cefr_refs"),
            cefr_claims=claims, source_refs=cls._list_of_strings(value["source_refs"], "source_refs"),
            warnings=cls._list_of_strings(value["warnings"], "warnings"),
        )


class ValidationGate:
    """Closed-world, deterministic validation of generic pedagogical JSON."""

    @staticmethod
    def source_registry(context: PedagogicalContext) -> AllowedSourceRegistry:
        cefr: dict[str, AllowedCEFRFact] = {}
        for index, descriptor in enumerate(context.cefr_descriptors, start=1):
            cefr[f"cefr:{index}"] = AllowedCEFRFact(
                ref=f"cefr:{index}", level=descriptor.level, scale=descriptor.scale,
                status=descriptor.status, descriptor_text=descriptor.descriptor_text,
                reference_level=descriptor.reference_level,
            )
        for index, missing in enumerate(context.cefr_missing, start=1):
            cefr[f"cefr-missing:{index}"] = AllowedCEFRFact(
                ref=f"cefr-missing:{index}", level=missing.level, scale=missing.scale,
                status=missing.status, descriptor_text=None, reference_level=None,
            )
        resources = {
            f"resource:{block.source_number}": AllowedResource(
                ref=f"resource:{block.source_number}", document_id=block.document_id,
                chunk_ids=list(block.chunk_ids), page_start=block.page_start, page_end=block.page_end,
            )
            for block in context.resource_blocks
        }
        return AllowedSourceRegistry(cefr_facts=cefr, resources=resources)

    @staticmethod
    def _report(issues: list[ValidationIssue], registry: AllowedSourceRegistry) -> ValidationReport:
        errors = [issue for issue in issues if issue.severity == "ERROR"]
        warnings = [issue for issue in issues if issue.severity == "WARNING"]
        return ValidationReport(not errors, errors, warnings, len(registry.cefr_facts), len(registry.resources))

    def validate(self, context: PedagogicalContext, output: StructuredPedagogicalOutput) -> ValidationReport:
        registry = self.source_registry(context)
        issues: list[ValidationIssue] = []
        summary = output.pedagogical_summary
        requested_level = str(context.request_summary["cefr_level"])
        requested_language = context.request_summary.get("language")
        if summary.cefr_level != requested_level:
            issues.append(ValidationIssue("INVALID_CEFR_LEVEL", "ERROR", "Output CEFR level differs from the request.", "pedagogical_summary.cefr_level"))
        if requested_language and summary.output_language != requested_language:
            issues.append(ValidationIssue("OUTPUT_LANGUAGE_MISMATCH", "ERROR", "Output language metadata differs from the request.", "pedagogical_summary.output_language"))

        all_refs = list(output.source_refs) + list(output.cefr_refs)
        for index, section in enumerate(output.content_sections):
            all_refs.extend(section.source_refs)
            if section.source_basis == "cefr" and not any(ref.startswith("cefr") for ref in section.source_refs):
                issues.append(ValidationIssue("MISSING_REQUIRED_SOURCE", "ERROR", "A CEFR-based section requires a CEFR reference.", f"content_sections[{index}].source_refs"))
            if section.source_basis == "resource" and not any(ref.startswith("resource:") for ref in section.source_refs):
                issues.append(ValidationIssue("MISSING_REQUIRED_SOURCE", "ERROR", "A resource-based section requires a resource reference.", f"content_sections[{index}].source_refs"))
        for ref in sorted(set(all_refs)):
            if ref not in registry.source_refs:
                issues.append(ValidationIssue("UNKNOWN_SOURCE_REFERENCE", "ERROR", f"Reference is not present in the context: {ref}", "source_refs"))

        for index, claim in enumerate(output.cefr_claims):
            allowed = registry.cefr_facts.get(claim.ref)
            if claim.ref not in output.cefr_refs:
                issues.append(ValidationIssue("MISSING_REQUIRED_SOURCE", "ERROR", "A CEFR claim must be listed in cefr_refs.", f"cefr_claims[{index}].ref"))
            if allowed is None or (
                claim.level != allowed.level or claim.scale != allowed.scale or claim.status != allowed.status
                or claim.descriptor_text != allowed.descriptor_text or claim.reference_level != allowed.reference_level
            ):
                issues.append(ValidationIssue("CEFR_FACT_NOT_IN_CONTEXT", "ERROR", "CEFR fact is absent from or differs from the authoritative context.", f"cefr_claims[{index}]"))

        if output.content_sections:
            issues.append(ValidationIssue("CONTENT_LANGUAGE_NOT_VERIFIED", "WARNING", "Body language was not automatically verified; output_language metadata is authoritative.", "content_sections"))
        return self._report(issues, registry)


class StructuredPromptBuilder:
    """Builds one bounded JSON-only prompt from the closed-world registry."""

    def build(self, context: PedagogicalContext, registry: AllowedSourceRegistry) -> tuple[str, str]:
        schema = {
            "title": "optional string", "pedagogical_summary": {"cefr_level": "string", "output_language": "string", "objective": "string|null", "skills": ["string"]},
            "content_sections": [{"section_type": "string", "heading": "string|null", "content": "string", "source_refs": ["allowed ref"], "source_basis": "cefr|resource|null"}],
            "cefr_refs": ["cefr:* or cefr-missing:*"],
            "cefr_claims": [{"ref": "cefr:*", "level": "string", "scale": "string", "status": "string", "descriptor_text": "string|null", "reference_level": "string|null"}],
            "source_refs": ["allowed ref"], "warnings": ["string"],
        }
        facts = [fact.__dict__ for fact in registry.cefr_facts.values()]
        # Resource IDs are the complete citation contract for generation.  The
        # document/page/chunk provenance stays in the server-side registry and
        # is validated after generation; asking the model to echo it is wasteful.
        resources = [{"ref": resource.ref} for resource in registry.resources.values()]
        system = (
            "Return only one JSON object matching the requested schema. Do not use Markdown. "
            "Use only allowed references and CEFR facts. Structured CEFR facts are authoritative; "
            "never turn missing or unavailable descriptors into AVAILABLE descriptors. "
            "Keep the draft compact: return at most three content sections; do not reproduce document, "
            "page, or chunk metadata; reference only the allowed IDs. Include a CEFR claim only when "
            "explicitly restating that exact CEFR fact, never as a copy of the whole CEFR registry."
        )
        user = json.dumps({
            "request": context.request_summary, "allowed_cefr_facts": facts,
            "allowed_resources": resources, "schema": schema,
        }, ensure_ascii=False)
        return system, user


class StructuredGenerationService:
    """LLM/provider composition with explicit parse and validation failures."""

    def __init__(self, *, llm: LLMProvider, settings: Settings | None = None, gate: ValidationGate | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.gate = gate or ValidationGate()
        self.prompt_builder = StructuredPromptBuilder()

    def _failure(
        self, registry: AllowedSourceRegistry, issue: ValidationIssue, *, provider: str | None = None,
        finish_reason: str | None = None, output_token_count: int | None = None, parse_succeeded: bool = False,
    ) -> StructuredGenerationResult:
        report = ValidationGate._report([issue], registry)
        return StructuredGenerationResult(None, report, provider, finish_reason, output_token_count, parse_succeeded)

    def generate(self, context: PedagogicalContext) -> StructuredGenerationResult:
        registry = self.gate.source_registry(context)
        system_prompt, user_prompt = self.prompt_builder.build(context, registry)
        try:
            result: LLMResult = self.llm.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=self.settings.rag_llm_temperature,
                max_tokens=self.settings.structured_generation_max_output_tokens,
            )
        except LLMProviderError as exc:
            return self._failure(registry, ValidationIssue("LLM_PROVIDER_ERROR", "ERROR", exc.provider_message), provider=self.llm.model_id)
        if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
            return self._failure(
                registry, ValidationIssue("LLM_OUTPUT_TRUNCATED", "ERROR", "Provider stopped at its output token limit."),
                provider=result.model, finish_reason=result.finish_reason, output_token_count=result.output_token_count,
            )
        try:
            output = StructuredOutputParser.parse(result.text)
        except ValueError as exc:
            return self._failure(
                registry, ValidationIssue("STRUCTURED_OUTPUT_PARSE_ERROR", "ERROR", str(exc)),
                provider=result.model, finish_reason=result.finish_reason, output_token_count=result.output_token_count,
            )
        report = self.gate.validate(context, output)
        validated = ValidatedPedagogicalOutput(output, report, result.model, result.finish_reason, result.output_token_count) if report.is_valid else None
        return StructuredGenerationResult(validated, report, result.model, result.finish_reason, result.output_token_count, True)
