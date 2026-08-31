"""Bounded, deterministic pedagogical ordering for already retrieved candidates.

This service never queries Qdrant, embeds text, or filters canonical chunks.
It can only make small, explainable moves among H3 candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from app.services.retrieval_service import RetrievalResult


class PedagogicalRole:
    ACTIVITY = "ACTIVITY"
    EXERCISE = "EXERCISE"
    DIALOGUE = "DIALOGUE"
    TASK = "TASK"
    ASSESSMENT = "ASSESSMENT"
    METHODOLOGY = "METHODOLOGY"
    REFERENCE = "REFERENCE"
    OTHER = "OTHER"


_ROLE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (PedagogicalRole.DIALOGUE, ("dialogue", "حوار")),
    (PedagogicalRole.EXERCISE, ("exercice", "exercise", "worksheet_exercise", "تمرين")),
    (PedagogicalRole.TASK, ("jeu de rôle", "role play", "role-play", "tâche", "mission", "consigne", "مهمة", "تعليمات")),
    (PedagogicalRole.ASSESSMENT, ("évaluation", "evaluation", "correction", "assessment", "تقويم", "تصحيح")),
    (PedagogicalRole.ACTIVITY, ("activité", "activity", "نشاط")),
    (PedagogicalRole.METHODOLOGY, ("méthodolog", "methodolog", "démarche", "déroulement", "pedagogical guide", "guide pédagogique")),
)
_ROLE_PREFERENCES = {
    "activity": (PedagogicalRole.ACTIVITY, PedagogicalRole.TASK, PedagogicalRole.DIALOGUE, PedagogicalRole.EXERCISE),
    "exercise": (PedagogicalRole.EXERCISE, PedagogicalRole.TASK, PedagogicalRole.ACTIVITY),
    "dialogue": (PedagogicalRole.DIALOGUE, PedagogicalRole.TASK, PedagogicalRole.ACTIVITY),
    "role_play": (PedagogicalRole.DIALOGUE, PedagogicalRole.TASK, PedagogicalRole.ACTIVITY, PedagogicalRole.EXERCISE),
    "listening": (PedagogicalRole.ACTIVITY, PedagogicalRole.EXERCISE, PedagogicalRole.DIALOGUE, PedagogicalRole.TASK),
    "task": (PedagogicalRole.TASK, PedagogicalRole.DIALOGUE, PedagogicalRole.ACTIVITY),
    "methodology": (PedagogicalRole.METHODOLOGY, PedagogicalRole.REFERENCE, PedagogicalRole.ACTIVITY),
}
_LEVELS = ("PRE-A1", "A1", "A2", "B1", "B2", "C1", "C2")
_SKILL_MARKERS = {
    "listening": ("compréhension de l'oral", "compréhension orale", "écouter", "écoute", "listening", "الاستماع", "فهم المسموع"),
    "speaking": ("production orale", "expression orale", "prise de parole", "parler", "التعبير الشفهي", "نشاط شفهي"),
    "interaction": ("interaction orale", "dialogue", "conversation", "تفاعل شفهي", "حوار"),
    "reading": ("compréhension écrite", "lecture", "reading", "القراءة", "فهم المقروء"),
    "writing": ("production écrite", "expression écrite", "writing", "الكتابة", "التعبير الكتابي"),
}


@dataclass(frozen=True)
class PedagogicalRankingRequest:
    intent: str = "general"
    cefr_level: str | None = None
    skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class PedagogicalRankingWeights:
    """Small caps keep the original H3 rank dominant across the candidate pool."""

    primary_role: float = 0.06
    secondary_role: float = 0.04
    tertiary_role: float = 0.02
    concrete: float = 0.03
    exact_level: float = 0.025
    adjacent_level: float = -0.005
    distant_level: float = -0.02
    matched_skill: float = 0.02
    incompatible_skill: float = -0.01
    rank_step: float = 0.025


class PedagogicalRetrievalRanker:
    def __init__(self, weights: PedagogicalRankingWeights | None = None) -> None:
        self.weights = weights or PedagogicalRankingWeights()

    @staticmethod
    def role(result: RetrievalResult) -> str:
        if result.content_type == "worksheet_exercise":
            return PedagogicalRole.EXERCISE
        if re.search(r"(?:^|\n)\s*(?:enseignant|professeur|apprenant|élève|teacher|learner)\s*:", result.content, re.I):
            return PedagogicalRole.DIALOGUE
        evidence = " ".join((result.content_type, *result.heading_context, result.content)).casefold()
        for role, markers in _ROLE_MARKERS:
            if any(marker in evidence for marker in markers):
                return role
        return PedagogicalRole.REFERENCE if result.content_type in {"table", "list"} else PedagogicalRole.OTHER

    def rank(self, candidates: Iterable[RetrievalResult], request: PedagogicalRankingRequest) -> list[RetrievalResult]:
        scored: list[tuple[RetrievalResult, float, float, float, float, float, tuple[str, ...]]] = []
        for result in candidates:
            role = self.role(result)
            role_adjustment, role_reason = self._role_adjustment(role, request.intent)
            concrete, concreteness_reasons = self.concreteness(result)
            concrete_adjustment, concrete_reason = self._concreteness_adjustment(concrete, request.intent)
            level_adjustment, level_reason = self._level_adjustment(result.cefr_level, request.cefr_level)
            skill_adjustment, skill_reason = self._skill_adjustment(result, request.skills)
            reasons = tuple(reason for reason in (role_reason, concrete_reason, level_reason, skill_reason) if reason)
            raw_adjustment = role_adjustment + concrete_adjustment + level_adjustment + skill_adjustment
            # The rank foundation decays by 0.025 per original H3 position;
            # the maximum bounded adjustment is smaller than a long-rank jump.
            fused_rank = result.fused_rank or result.rank
            base = 1.0 - self.weights.rank_step * max(fused_rank - 1, 0)
            # The gate keeps a weak late H3 result from being rescued solely by
            # pedagogical cues while allowing nearby candidates to reorder.
            relevance_gate = max(0.35, 1.0 - 0.04 * max(fused_rank - 1, 0))
            adjustment = raw_adjustment * relevance_gate
            scored.append((result, base + adjustment, role_adjustment * relevance_gate, concrete_adjustment * relevance_gate, level_adjustment * relevance_gate, skill_adjustment * relevance_gate, reasons, role, concrete, concreteness_reasons, skill_reason))
        scored.sort(key=lambda item: (-item[1], item[0].fused_rank or item[0].rank, item[0].chunk_id))
        return [replace(
            result,
            pedagogical_adjustment_total=role_adjustment + concrete_adjustment + level_adjustment + skill_adjustment,
            role_adjustment=role_adjustment,
            concreteness_adjustment=concrete_adjustment,
            level_adjustment=level_adjustment,
            skill_adjustment=skill_adjustment,
            final_score=score,
            final_rank=index,
            adjustment_reasons=list(reasons),
            pedagogical_role=role,
            role_source="deterministic_content_evidence",
            is_concrete_classroom_material=concrete,
            concreteness_reasons=list(concreteness_reasons),
            skill_evidence_reason=skill_reason,
            rank=index,
        ) for index, (result, score, role_adjustment, concrete_adjustment, level_adjustment, skill_adjustment, reasons, role, concrete, concreteness_reasons, skill_reason) in enumerate(scored, start=1)]

    def desired_candidate_exists(self, candidates: Iterable[RetrievalResult], request: PedagogicalRankingRequest) -> bool | None:
        preferences = _ROLE_PREFERENCES.get(request.intent)
        return any(self.role(candidate) in preferences for candidate in candidates) if preferences else None

    def union_diagnostics(self, candidates: Iterable[RetrievalResult], request: PedagogicalRankingRequest) -> dict[str, object]:
        items = list(candidates)
        preferences = _ROLE_PREFERENCES.get(request.intent, ())
        desired = [item for item in items if self.role(item) in preferences]
        concrete = [item for item in items if self.concreteness(item)[0]]
        desired_and_concrete = [item for item in desired if self.concreteness(item)[0]]
        best = min(desired_and_concrete, key=lambda item: item.fused_rank or item.rank, default=None)
        return {
            "desired_role_candidate_exists_in_union": bool(desired) if preferences else None,
            "concrete_candidate_exists_in_union": bool(concrete),
            "desired_and_concrete_candidate_exists_in_union": bool(desired_and_concrete) if preferences else None,
            "best_desired_and_concrete": None if best is None else {
                "fused_rank": best.fused_rank or best.rank,
                "final_rank": best.final_rank,
                "role": best.pedagogical_role or self.role(best),
                "content_type": best.content_type,
            },
        }

    def _role_adjustment(self, role: str, intent: str) -> tuple[float, str | None]:
        preferences = _ROLE_PREFERENCES.get(intent)
        if not preferences or role not in preferences:
            return 0.0, None
        value = (self.weights.primary_role, self.weights.secondary_role, self.weights.tertiary_role, self.weights.tertiary_role)[min(preferences.index(role), 3)]
        return value, f"{intent}_role_match"

    def _concreteness_adjustment(self, concrete: bool, intent: str) -> tuple[float, str | None]:
        if intent not in _ROLE_PREFERENCES or not concrete:
            return 0.0, None
        return self.weights.concrete, "concrete_classroom_material"

    def concreteness(self, result: RetrievalResult) -> tuple[bool, tuple[str, ...]]:
        """Require positive classroom evidence; role labels are deliberately insufficient."""
        if result.content_type == "worksheet_exercise":
            return True, ("canonical_worksheet_exercise",)
        evidence = " ".join((*result.heading_context, result.content)).casefold()
        ignored: list[str] = []
        if re.search(r"\b(?:figure|profil)\s+\d+\s*[-:]", result.content, re.I) or re.search(r"\b(?:[a-zà-ÿ]+,\s*)?\d+\s*=|\b(?:pre-)?[a-c]\d\+?\s*=", result.content, re.I):
            ignored.append("serialized_table_enumeration_ignored")
        if re.search(r"\bpeut\s+\w+|\bcan\s+\w+", evidence, re.I):
            ignored.append("generic_descriptor_ignored")
        action = r"(?:écoutez|répondez|complétez|associez|discutez|présentez|jouez|demandez|lisez|écrivez|travaillez|أجب|استمع|أكمل|اكتب|اقرأ)"
        if re.search(rf"(?:^|\n)\s*\d+\s*[).:-]\s*{action}\b", result.content, re.I):
            return True, ("numbered_procedural_steps",)
        if re.search(r"(?:^|\n)\s*(?:enseignant|professeur|apprenant|élève|teacher|learner)\s*:", result.content, re.I):
            return True, ("dialogue_speaker_structure",)
        if re.search(rf"(?:consigne|instructions?|déroulement|matériel|التعليمات)\s*[:\-].{{0,100}}{action}\b", result.content, re.I) or re.search(rf"{action}\s+(?:en binômes|aux questions|le dialogue|la consigne|ثم)", evidence, re.I):
            return True, ("learner_instruction",)
        if re.search(r"(?:rôle\s*[:\-]|role\s*[:\-]).{0,100}(?:apprenant|élève|client|serveur|teacher|learner)", result.content, re.I):
            return True, ("role_play_structure",)
        if len(re.findall(r"\?", result.content)) >= 2 and re.search(r"(?:questions?|réponses?|سؤال|أجب)", evidence, re.I):
            return True, ("learner_question_set",)
        # Tables remain eligible: only explicit usable classroom structure above
        # distinguishes a concrete table from a capability/reference table.
        return False, tuple(ignored)

    def _is_concrete(self, result: RetrievalResult, role: str | None = None) -> bool:
        return self.concreteness(result)[0]

    def _level_adjustment(self, candidate_level: str | None, requested_level: str | None) -> tuple[float, str | None]:
        if not candidate_level or not requested_level:
            return 0.0, None
        try:
            distance = abs(_LEVELS.index(candidate_level.upper()) - _LEVELS.index(requested_level.upper()))
        except ValueError:
            return 0.0, None
        if distance == 0:
            return self.weights.exact_level, "exact_reliable_cefr_level"
        if distance == 1:
            return self.weights.adjacent_level, "adjacent_reliable_cefr_level"
        return self.weights.distant_level, "distant_reliable_cefr_level"

    def _skill_adjustment(self, result: RetrievalResult, requested_skills: tuple[str, ...]) -> tuple[float, str | None]:
        if not requested_skills:
            return 0.0, None
        heading = " ".join(result.heading_context).casefold()
        content = result.content.casefold()
        heading_skills = {skill for skill, markers in _SKILL_MARKERS.items() if any(marker in heading for marker in markers)}
        if any(skill in heading_skills for skill in requested_skills):
            return self.weights.matched_skill, "heading_skill_signal"
        if heading_skills:
            return self.weights.incompatible_skill, "incompatible_heading_skill_signal"
        content_skills = {skill for skill, markers in _SKILL_MARKERS.items() if any(marker in content for marker in markers)}
        if len(content_skills) > 1:
            return 0.0, "ambiguous_multiskill_chunk"
        if any(skill in content_skills for skill in requested_skills):
            return self.weights.matched_skill, "local_skill_signal"
        if content_skills:
            return self.weights.incompatible_skill, "incompatible_explicit_skill_signal"
        return 0.0, None
