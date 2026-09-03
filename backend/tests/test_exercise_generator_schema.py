"""HTTP contract tests for the exercise generator.

These prove the exact FastAPI/Pydantic boundary that produced a 422 when the
frontend sent `objective: ""`: the field is optional (str | None) but NON-EMPTY
when supplied. The frontend therefore sends `null` for empty optionals.
"""

import json

import pytest
from pydantic import ValidationError

from app.schemas.exercise_generator import ExerciseGenerateIn


def _valid_payload(**overrides):
    payload = {
        "level": "A1", "theme": "La famille", "objective": None,
        "skills": ["Vocabulaire"], "exercise_type": "QCM", "count": 8,
        "language": "ar", "adapt_with_ai": False, "special_instructions": None,
    }
    payload.update(overrides)
    return payload


# 1. Payload minimal valide (tous les champs optionnels absents).
def test_minimal_payload_is_valid():
    raw = json.dumps({"level": "A1", "theme": "La famille"})
    model = ExerciseGenerateIn.model_validate(json.loads(raw))
    assert model.objective is None
    assert model.count == 8
    assert model.adapt_with_ai is False
    assert model.language == "ar"


# 2. Payload complet valide.
def test_full_payload_is_valid():
    model = ExerciseGenerateIn.model_validate(_valid_payload(
        level="B1", objective="Développer le vocabulaire du voyage",
        skills=["Expression orale"], exercise_type="Vrai ou faux", count=12,
        language="fr", adapt_with_ai=True, special_instructions="Sans support audio",
    ))
    assert model.level == "B1"
    assert model.objective == "Développer le vocabulaire du voyage"
    assert model.count == 12
    assert model.adapt_with_ai is True


# 3. Objectif vide en tant que chaîne -> rejeté (cause exacte du 422).
def test_empty_objective_string_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(objective=""))
    errors = exc.value.errors(include_url=False)
    assert any(error["loc"] == ("objective",) and error["type"] == "string_too_short" for error in errors)


# 3b. Objectif vide en tant que null -> accepté (contrat attendu du frontend).
def test_empty_objective_null_is_valid():
    model = ExerciseGenerateIn.model_validate(_valid_payload(objective=None))
    assert model.objective is None


# 4. skills vide -> accepté (aucune contrainte de longueur).
def test_empty_skills_are_valid():
    model = ExerciseGenerateIn.model_validate(_valid_payload(skills=[]))
    assert model.skills == []


# 5 / 6. adapt_with_ai false / true.
def test_adapt_with_ai_false_is_valid():
    assert ExerciseGenerateIn.model_validate(_valid_payload(adapt_with_ai=False)).adapt_with_ai is False


def test_adapt_with_ai_true_is_valid():
    assert ExerciseGenerateIn.model_validate(_valid_payload(adapt_with_ai=True)).adapt_with_ai is True


# 7 / 8. Bornes de count.
def test_count_one_is_valid():
    assert ExerciseGenerateIn.model_validate(_valid_payload(count=1)).count == 1


def test_count_twenty_is_valid():
    assert ExerciseGenerateIn.model_validate(_valid_payload(count=20)).count == 20


def test_count_zero_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(count=0))
    assert any(error["type"] == "greater_than_equal" for error in exc.value.errors(include_url=False))


def test_count_above_twenty_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(count=21))
    assert any(error["type"] == "less_than_equal" for error in exc.value.errors(include_url=False))


# 9. exercise_type : champ libre (chaîne), accepte aussi les types personnalisés.
def test_exercise_type_free_string_is_valid():
    for value in ("QCM", "Vrai ou faux", "Dictée", "Mes propres consignes"):
        model = ExerciseGenerateIn.model_validate(_valid_payload(exercise_type=value))
        assert model.exercise_type == value


# 10. exercise_type invalide -> 422 (non-chaîne ou trop long).
def test_exercise_type_non_string_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(exercise_type=123))
    assert any(error["type"] == "string_type" for error in exc.value.errors(include_url=False))


def test_exercise_type_too_long_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(exercise_type="x" * 101))
    assert any(error["type"] == "string_too_long" for error in exc.value.errors(include_url=False))


# Contrôles stricts : level / language.
def test_level_literal_is_enforced():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(level="A3"))
    assert any(error["type"] == "literal_error" for error in exc.value.errors(include_url=False))


def test_language_literal_is_enforced():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(language="de"))
    assert any(error["type"] == "literal_error" for error in exc.value.errors(include_url=False))


# Champs inconnus interdits (extra="forbid").
def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError) as exc:
        ExerciseGenerateIn.model_validate(_valid_payload(exerciseType="QCM"))
    assert any(error["type"] == "extra_forbidden" for error in exc.value.errors(include_url=False))


# Le payload exact envoyé par le frontend corrigé valide.
def test_frontend_corrected_payload_is_valid():
    raw = json.dumps({
        "level": "A1", "theme": "La famille", "objective": None,
        "skills": ["Vocabulaire"], "exercise_type": "QCM", "count": 8,
        "language": "ar", "adapt_with_ai": False, "special_instructions": None,
    })
    model = ExerciseGenerateIn.model_validate(json.loads(raw))
    assert model.objective is None
    assert model.special_instructions is None