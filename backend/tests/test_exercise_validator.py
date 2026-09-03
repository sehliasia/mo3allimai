"""Tests for ExerciseValidator."""
from app.schemas.exercise_generator import ExerciseItem
from app.services.exercise_validator import ExerciseValidator


def _item(**overrides):
    base = {
        "title": "Exercice", "skill": "Vocabulaire", "exercise_type": "qcm",
        "prompt": "أكمل الجملة بالكلمة الصحيحة", "answer_expectation": "أب",
        "level": "A1", "options": ["أب", "أم", "أخ", "أخت"], "is_true": None,
        "pairs": [], "status": "ai_generated", "level_source": "generated",
    }
    base.update(overrides)
    return ExerciseItem.model_validate(base)


def _validate(validator, items, **req):
    defaults = {"request_level": "A1", "theme": "Famille", "language": "ar"}
    defaults.update(req)
    return validator.validate(items, **defaults)


def test_valid_item_passes():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item()])
    assert verdicts[0].ok is True


def test_empty_prompt_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(prompt="   ")])
    assert verdicts[0].ok is False
    assert any("consigne vide" in r for r in verdicts[0].reasons)


def test_wrong_level_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(level="B2")], request_level="A1")
    assert verdicts[0].ok is False
    assert any("niveau" in r for r in verdicts[0].reasons)


def test_non_arabic_prompt_flagged_when_requesting_arabic():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(prompt="Fill in the blank with the correct word.")])
    assert verdicts[0].ok is False


def test_strong_french_arabic_mix_flagged_when_requesting_arabic():
    v = ExerciseValidator()
    # Arabic present but with a large Latin (French) share: student content is no
    # longer natural Arabic → regenerate.
    verdicts = _validate(v, [_item(prompt="أريد أن partir إلى المدرسة مع ensemble الطلاب")])
    assert verdicts[0].ok is False
    assert any("mélange" in r for r in verdicts[0].reasons)


def test_light_latin_inside_arabic_is_tolerated():
    v = ExerciseValidator()
    # A little Latin (e.g. a proper noun or a short label) should not be rejected.
    verdicts = _validate(v, [_item(prompt="اقرأ النص عن الحياة اليومية وأجب عن الأسئلة")])
    assert verdicts[0].ok is True


def test_qcm_without_options_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(options=[])])
    assert verdicts[0].ok is False
    assert any("QCM sans options" in r for r in verdicts[0].reasons)


def test_qcm_duplicate_options_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(options=["أب", "أب", "أم", "أخت"])])
    assert any("options identiques" in r for r in verdicts[0].reasons)


def test_qcm_answer_missing_from_options_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(answer_expectation="جد", options=["أب", "أم", "أخ", "أخت"])])
    assert any("réponse correcte absente" in r for r in verdicts[0].reasons)


def test_true_false_requires_answer():
    v = ExerciseValidator()
    verdicts = _validate(v, [_item(exercise_type="true_false", is_true=None, answer_expectation=None)])
    assert any("vrai/faux sans réponse" in r for r in verdicts[0].reasons)


def test_duplicate_detection_marks_copies():
    v = ExerciseValidator()
    a = _item()
    dup = _item(prompt="أكمل الجملة بالكلمة الصحيحة من فضلك")
    indices = v.find_duplicate_indices([a, dup])
    assert 1 in indices


def test_no_false_duplicate_for_distinct_prompts():
    v = ExerciseValidator()
    a = _item(prompt="أكمل : عائلة أحمد")
    b = _item(prompt="رتب الكلمات لتكون جملة مفيدة")
    indices = v.find_duplicate_indices([a, b])
    assert indices == []


def test_ordering_rejectable_when_essential_word_missing():
    v = ExerciseValidator()
    # Correct phrase needs "إلى", which is missing from the shuffled words.
    verdicts = _validate(v, [
        _item(
            exercise_type="ordering",
            prompt='المدرسة · الطلاب · يذهبون · في · الصباح',
            answer_expectation="يذهب الطلاب إلى المدرسة في الصباح",
        ),
    ])
    assert verdicts[0].ok is False
    assert any("mot essentiel manquant" in r for r in verdicts[0].reasons)


def test_ordering_valid_when_all_answer_words_present():
    v = ExerciseValidator()
    verdicts = _validate(v, [
        _item(
            exercise_type="ordering",
            prompt='المدرسة · إلى · الطلاب · يذهبون · في · الصباح',
            answer_expectation="الطلاب يذهبون إلى المدرسة في الصباح",
        ),
    ])
    assert verdicts[0].ok is True


def test_ordering_without_answer_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [
        _item(exercise_type="ordering", answer_expectation=None),
    ])
    assert verdicts[0].ok is False
    assert any("réponse attendue manquante" in r for r in verdicts[0].reasons)


def test_french_arabic_mix_in_options_flagged():
    v = ExerciseValidator()
    # Latin embedded in an Arabic option (learner-facing) must be rejected.
    verdicts = _validate(v, [_item(options=["مدرسة", "بيت paru", "مستشفى", "مطعم"])])
    assert verdicts[0].ok is False
    assert any("mélange" in r for r in verdicts[0].reasons)


def test_french_arabic_mix_in_pairs_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [
        _item(exercise_type="matching", options=[], pairs=[
            {"left": "لكن", "right": "وقت ensemble"},
            {"left": "وقت", "right": "ترك"},
        ]),
    ])
    assert verdicts[0].ok is False
    assert any("mélange" in r for r in verdicts[0].reasons)


def test_french_arabic_mix_in_answer_flagged():
    v = ExerciseValidator()
    verdicts = _validate(v, [
        _item(exercise_type="complete", options=[], answer_expectation="يذهب إلى المدرسة chaque matin"),
    ])
    assert verdicts[0].ok is False
    assert any("mélange" in r for r in verdicts[0].reasons)


def test_metadata_fields_stay_allowed_in_french_when_arabic_requested():
    v = ExerciseValidator()
    # Internal metadata (title, skill, difficulty) may remain French/English and
    # must not be rejected — only learner-facing fields are checked.
    verdicts = _validate(v, [
        _item(title="Vocabulaire – l'école", skill="Vocabulaire", prompt="رتّب الكلمات لتكوين جملة"),
    ])
    assert verdicts[0].ok is True
