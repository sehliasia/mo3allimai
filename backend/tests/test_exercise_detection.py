"""Deterministic multi-signal exercise detection and typing.

The core rule under test: the mere presence of "exercice"/"تمرين" is NEVER
sufficient; descriptive/preface paragraphs about exercises are rejected.
"""

import pytest

from app.services.exercise_detection import classify_exercise_type, score_exercise


def _result(text, heading=None, threshold=2.0):
    return score_exercise(text, [heading] if heading else None, threshold=threshold)


# -- Positive: real task structure -----------------------------------------

def test_exercise_title_and_instruction_detected():
    assert _result("Exercice 1 : Complète les phrases.").is_exercise is True


def test_activity_title_with_matching_verb_detected():
    assert _result("Activité 2 : Relie les mots aux images.").is_exercise is True


def test_consigne_with_mcq_hint_detected():
    assert _result("Consigne : Choisis la bonne réponse.").is_exercise is True


def test_numbered_arabic_items_detected():
    assert _result("1. هذا أبي\n2. هذه أمي").is_exercise is True


def test_choice_items_detected():
    assert _result("Choisis :\na) آ\nb) ب\nc) ج").is_exercise is True


def test_real_arabic_choices_block_detected_as_qcm():
    # Real Miftah / Sokkan alJarra worksheets list options under a
    # "الاختيارات:" header with hyphen-prefixed options. Such a block is a
    # genuine multiple-choice exercise stem, not a descriptive preface.
    text = "الاختيارات:\n- أحمر\n- أزرق\n- أخضر\nالاختيارات:\n- الأحد\n- الاثنين"
    result = _result(text)
    assert result.is_exercise is True
    assert classify_exercise_type(text) == "qcm"


def test_choices_block_with_title_detected_qcm():
    text = "تمرين 15\nالاختيارات:\n- 15\n- 14\n- 13"
    result = _result(text)
    assert result.is_exercise is True
    assert result.exercise_type == "qcm"


def test_arabic_title_alone_never_suffices():
    # "تمرين 4" alone (title without any item/option) is not an exercise.
    assert _result("تمرين 4\n4").is_exercise is False


def test_blank_fill_with_items_detected():
    text = "Exercice 3 : Complète les phrases.\n1. Mon ______ s'appelle Ahmed.\n2. Ma ______ s'appelle Fatima."
    result = _result(text)
    assert result.is_exercise is True
    assert result.confidence > 0.5


# -- Negative: descriptive/preface/methodology text ------------------------

@pytest.mark.parametrize("text", [
    "Ce manuel comprend plusieurs exercices.",
    "Les exercices proposés permettent aux élèves de consolider leurs acquis.",
    "Pour chaque niveau, des exercices de production sont proposés afin de consolider la progression.",
    "Nous avons conçu un cahier d'activités pour les années à venir.",
    "La présente méthode s'adresse aux apprenants débutants.",
])
def test_descriptive_text_is_never_an_exercise(text):
    result = _result(text)
    assert result.is_exercise is False
    assert result.exercise_type is None


def test_word_exercise_alone_never_suffices():
    assert _result("exercice exercice exercice").is_exercise is False
    assert _result("تمرين").is_exercise is False


def test_strict_threshold_can_filter_weak_signals():
    medium = "Consigne : Choisis la bonne réponse."
    assert _result(medium, threshold=3.0).is_exercise is False
    assert _result(medium, threshold=1.0).is_exercise is True


# -- Confidence is bounded [0,1] even for strong negatives ------------------

def test_confidence_bounded_for_negative_scores():
    result = _result("Pour chaque niveau, des exercices de production sont proposés.")
    assert 0.0 <= result.confidence <= 1.0


# -- Type classification ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Choisis la bonne réponse.\na) أ\nb) ب\nc) ج", "qcm"),
    ("Complète avec le mot correct.\n1. أكل ______ التفاحة.", "fill_blank"),
    ("Relie chaque mot à son image.\n1. البيت 2. المدرسة", "matching"),
    ("Vrai ou faux ?\n1. الشمس تشرق من الشرق.", "true_false"),
    ("Remets les phrases en ordre.\n1. التلميذ 2. يقرأ 3. الكتاب", "ordering"),
    ("Transforme ces phrases au pluriel.\n1. أنا تلميذ.", "transformation"),
])
def test_exercise_types(text, expected):
    assert classify_exercise_type(text) == expected


def test_unknown_structure_is_mixed():
    assert classify_exercise_type("Consigne : recopie la première leçon.") == "mixed"