import json

import pytest

from app.schemas.lesson_plan import LessonPlanGenerateIn
from app.services.lesson_plan_generation_service import LessonPlanGenerationError, LessonPlanGenerationService
from app.services.llm_provider import FakeLLMProvider
from app.services.pedagogical_knowledge_service import PedagogicalContext


def _context():
    return PedagogicalContext({"cefr_level": "A1", "language": "ar"}, [], [], [], 0, 0, [], [], 0)


def _plan(durations=(10, 15, 15, 10, 10)):
    phases = ["Découverte", "Compréhension", "Pratique guidée", "Production", "Évaluation"]
    return {
        "title": "أسرتي", "level": "A1", "theme": "La famille", "duration": 60, "audience": "Adultes",
        "session_type": "Découverte", "skills": ["speaking"], "age_approximation": "6 à 10 ans", "communicative_objectives": ["Présenter un membre de sa famille"], "linguistic_objectives": ["Utiliser هذا / هذه avec les possessifs"], "general_objective": "Nommer sa famille",
        "specific_objectives": ["Dire هذا أبي"], "prerequisites": [],
        "linguistic_content": {"vocabulary": ["الأب", "الأم", "الأخ", "الأخت"], "grammar": ["هذا أبي"]},
        "materials": ["tableau et marqueur", "six images imprimées : père, mère, frère, sœur, grand-père, grand-mère"],
        "lesson_flow": [{"phase": phase, "duration": duration, "objective": "Objectif clair", "teacher_role": "Montre les images puis modèle la phrase.", "learner_activity": "Observe, répète puis associe l'image au mot.", "instructions": "انظر إلى الصورة وسمِّ أفراد الأسرة.", "materials": ["images imprimées"], "work_mode": "binômes", "example": "L'enseignant montre الأب ; les apprenants disent : هذا أبي.", "expected_result": "L'apprenant associe correctement l'image et la phrase."} for phase, duration in zip(phases, durations)],
        "assessment": {"assessment_type": "formative", "moment": "Fin de séance", "criteria": ["Vocabulaire"], "method": "observation", "activity": "Identifier quatre images", "instructions": "من هذا؟ من هذه؟", "success_indicators": ["nomme quatre personnes"], "rubric": [{"criterion": "Vocabulaire", "achieved": "Identifie quatre personnes", "to_reinforce": "Révise avec les images"}]},
        "differentiation": {"support": ["Donner un modèle de phrase et quatre images."], "extension": ["Présenter six membres de la famille."]},
        "extension": {"homework": "Réviser", "follow_up": "Jeu"},
    }


def test_a1_family_plan_is_validated_and_prompt_contains_quality_constraints():
    llm = FakeLLMProvider(json.dumps(_plan(), ensure_ascii=False))
    request = LessonPlanGenerateIn(level="A1", theme="La famille", general_objective="Apprendre le vocabulaire de la famille", duration_minutes=60, language="ar")
    result = LessonPlanGenerationService(llm=llm).generate(request, _context())
    assert result.lesson_flow[0].instructions == "انظر إلى الصورة وسمِّ أفراد الأسرة."
    assert "هذا أبي" in llm.calls[0]["system_prompt"]
    assert "somme de ses durées" in llm.calls[0]["system_prompt"]
    assert "lesson_flow est une liste de 5 objets" in llm.calls[0]["system_prompt"]
    assert "à A1, phrases courtes" in llm.calls[0]["system_prompt"]
    assert "expected_result" in llm.calls[0]["system_prompt"]
    assert "differentiation" in llm.calls[0]["system_prompt"]
    assert llm.calls[0]["max_tokens"] == 4096
    assert llm.calls[0]["generation_options"].reasoning_effort == "low"


def test_rejects_incoherent_phase_durations():
    service = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(_plan((10, 10, 10, 10, 10)))))
    with pytest.raises(LessonPlanGenerationError, match="somme des durées"):
        service.generate(LessonPlanGenerateIn(theme="La famille", general_objective="Vocabulaire", duration_minutes=60), _context())


def test_rejects_a_header_duration_that_differs_from_the_request():
    plan = _plan()
    plan["duration"] = 45
    service = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False)))

    with pytest.raises(LessonPlanGenerationError, match="durée annoncée"):
        service.generate(LessonPlanGenerateIn(theme="La famille", general_objective="Vocabulaire", duration_minutes=60), _context())


def test_rejects_a_plan_without_the_required_pedagogical_progression():
    plan = _plan()
    plan["lesson_flow"][2]["phase"] = "Interaction libre"
    service = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False)))

    with pytest.raises(LessonPlanGenerationError, match="phases"):
        service.generate(LessonPlanGenerateIn(theme="La famille", general_objective="Vocabulaire", duration_minutes=60), _context())


def test_rejects_an_empty_material_list_for_a_lesson_step():
    plan = _plan()
    plan["lesson_flow"][0]["materials"] = []
    service = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False)))

    with pytest.raises(LessonPlanGenerationError, match="matériel"):
        service.generate(LessonPlanGenerateIn(theme="La famille", general_objective="Vocabulaire", duration_minutes=60), _context())


def test_extracts_a_complete_json_object_from_json_markdown_and_trailing_text():
    raw = "Réponse :\n```json\n" + json.dumps(_plan(), ensure_ascii=False) + "\n```\nFin."
    parsed = LessonPlanGenerationService._json_object(raw)

    assert parsed["title"] == "أسرتي"


def test_rejects_an_incomplete_json_object():
    with pytest.raises(LessonPlanGenerationError, match="incomplet ou invalide"):
        LessonPlanGenerationService._json_object('{"title": "أسرتي"')


def test_logs_raw_response_and_pydantic_field_errors_for_an_invalid_llm_shape(caplog):
    invalid = {"titre": "أسرتي", "objectifs": []}
    service = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(invalid, ensure_ascii=False)))

    with pytest.raises(LessonPlanGenerationError, match="format attendu"):
        service.generate(LessonPlanGenerateIn(level="A1", theme="الأسرة", general_objective="تعلم المفردات", duration_minutes=60, language="ar"), _context())

    assert "lesson_plan_schema_validation_failed" in caplog.text
    assert "title" in caplog.text and "أسرتي" in caplog.text


def test_normalizes_duration_strings_before_pydantic_validation():
    plan = _plan()
    plan["duration"] = "60 minutes"
    for step in plan["lesson_flow"]:
        step["duration"] = f'{step["duration"]} minutes'

    result = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False))).generate(
        LessonPlanGenerateIn(theme="La famille", general_objective="Vocabulaire", duration_minutes=60), _context()
    )

    assert result.duration == 60
    assert [step.duration for step in result.lesson_flow] == [10, 15, 15, 10, 10]


def test_normalizes_extension_string_lists_before_pydantic_validation():
    plan = _plan()
    plan["extension"] = {
        "homework": ["Dessiner sa famille"],
        "follow_up": ["Présenter le dessin"],
    }

    result = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False))).generate(
        LessonPlanGenerateIn(theme="La famille", general_objective="Vocabulaire", duration_minutes=60), _context()
    )

    assert result.extension.homework == "Dessiner sa famille"
    assert result.extension.follow_up == "Présenter le dessin"


def test_rejects_strings_for_all_list_fields_in_the_a1_family_plan():
    plan = _plan()
    plan.update({"skills": "expression orale", "communicative_objectives": "présenter", "linguistic_objectives": "هذا / هذه", "specific_objectives": "nommer", "prerequisites": "mots de base", "materials": "cartes"})
    plan["linguistic_content"] = {"Point grammatical": "هذا / هذه"}
    plan["lesson_flow"][0]["materials"] = "cartes"
    plan["assessment"].update({"criteria": "vocabulaire", "success_indicators": "quatre réponses"})
    plan["differentiation"] = {"support": "modèle", "extension": "six phrases"}
    service = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False)))

    with pytest.raises(LessonPlanGenerationError, match="format attendu"):
        service.generate(LessonPlanGenerateIn(level="A1", theme="الأسرة", general_objective="تعلم المفردات", duration_minutes=60, language="ar"), _context())


def test_detailed_a1_arabic_family_request_contains_all_classroom_ready_sections():
    request = LessonPlanGenerateIn(
        level="A1", theme="الأسرة", duration_minutes=60,
        general_objective="تعلم مفردات الأسرة والتحدث عن أفراد العائلة بجمل بسيطة",
        audience="Enfants marocains vivant à l'étranger, débutants",
        skills=["Compréhension orale", "Expression orale"],
        prerequisites=["معرفة بعض الكلمات العربية الأساسية"],
        linguistic_points=["الأب، الأم، الأخ، الأخت، الجد، الجدة", "هذا أبي، هذه أمي، هذا أخي، هذه أختي"],
        special_instructions="استخدام أنشطة بسيطة وتفاعلية مناسبة للمستوى A1", language="ar",
    )
    plan = _plan()
    plan.update({"theme": "الأسرة", "audience": request.audience, "skills": request.skills, "prerequisites": request.prerequisites, "general_objective": request.general_objective})
    plan["linguistic_content"]["Vocabulaire"] = ["الأب — père", "الأم — mère", "الأخ — frère", "الأخت — sœur", "الجد — grand-père", "الجدة — grand-mère"]
    result = LessonPlanGenerationService(llm=FakeLLMProvider(json.dumps(plan, ensure_ascii=False))).generate(request, _context())

    assert sum(step.duration for step in result.lesson_flow) == 60
    assert result.materials and result.communicative_objectives and result.linguistic_objectives
    assert all(step.example and step.expected_result and step.instructions for step in result.lesson_flow)
    assert result.assessment.activity and result.assessment.instructions and result.assessment.criteria
    assert result.differentiation.support and result.differentiation.extension
    assert result.extension.homework
