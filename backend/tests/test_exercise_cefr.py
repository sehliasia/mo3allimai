"""Internal CEFR constraint layer: explicit vs inferred levels and the strict
no-relabelling rule (an explicitly-A2 exercise is never presented as A1)."""

import pytest

from app.services.exercise_cefr import (
    LEVEL_RULES,
    SUPPORTED_LEVELS,
    detect_explicit_level,
    estimate_level,
    has_arabic_script,
    normalize_level,
)


def test_supported_levels_are_a1_to_c2():
    assert SUPPORTED_LEVELS == ("A1", "A2", "B1", "B2", "C1", "C2")
    # Every level exposes pedagogical rules without falling back to invention.
    for level in SUPPORTED_LEVELS:
        assert "tasks" in LEVEL_RULES[level]
        assert "sentence" in LEVEL_RULES[level]


def test_normalize_level_accepts_canonical_and_rejects_junk():
    assert normalize_level("a1") == "A1"
    assert normalize_level("B2") == "B2"
    assert normalize_level(" PRE-A1 ") == "A1"
    assert normalize_level("A3") is None
    assert normalize_level("D1") is None
    assert normalize_level(None) is None


def test_explicit_level_from_indexed_metadata():
    assert detect_explicit_level(content="Exercice", indexed_cefr_level="A2") == "A2"


def test_explicit_level_from_heading():
    assert detect_explicit_level(
        content="Complète les phrases.", heading_context=["Niveau B1 — Vocabulaire"],
    ) == "B1"


def test_explicit_level_from_content():
    assert detect_explicit_level(content="Exercice niveau C1 : argumentation") == "C1"


def test_explicit_level_arabic_heading():
    assert detect_explicit_level(
        content="أكمل الفراغ", heading_context=["المستوى أ1"],
    ) == "A1"


def test_no_level_stated_means_none_not_invented():
    assert detect_explicit_level(content="Complète les phrases suivantes.") is None
    assert detect_explicit_level(
        content="1. هذا أبي\n2. هذه أمي", heading_context=["Exercice 1"],
    ) is None


@pytest.mark.parametrize("level", ["A1", "A2", "B1", "B2", "C1", "C2"])
def test_estimate_returns_within_grid_when_possible(level):
    # The estimator is deterministic: call on a generic text must return
    # either None (not enough signal) or one of the grid levels.
    estimated, confidence = estimate_level("Complète les phrases avec les mots corrects.")
    if estimated is not None:
        assert estimated in SUPPORTED_LEVELS
    assert 0.0 <= confidence <= 0.9


def test_estimate_marks_basic_text_as_low_level():
    estimated, confidence = estimate_level("C'est mon père. C'est ma mère. C'est la classe.")
    assert estimated in ("A1", "A2")
    assert 0.0 < confidence <= 0.9


def test_estimate_marks_argumentative_text_high():
    estimated, confidence = estimate_level(
        "À mon avis, cette situation mérite une analyse approfondie. Argumentez votre point de vue.",
    )
    assert estimated == "B2"


def test_explicit_a2_is_jammed_when_strictly_requesting_a1():
    # The pairing used by search: an explicit A2 must not be relabelled A1.
    explicit = detect_explicit_level(content="QCM", indexed_cefr_level="A2")
    assert explicit == "A2"
    assert explicit != "A1"


def test_has_arabic_script():
    assert has_arabic_script("أكمل الفراغ بالكلمة الصحيحة")
    assert not has_arabic_script("Complete the sentence with the right word")