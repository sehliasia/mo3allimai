import json

import pytest

from app.schemas.activity_generator import ActivityGenerateIn
from app.services.activity_generation_service import ActivityGenerationError, ActivityGenerationService
from app.services.llm_provider import FakeLLMProvider
from app.services.pedagogical_knowledge_service import PedagogicalContext


def _context():
    return PedagogicalContext({"cefr_level": "A1", "language": "ar"}, [], [], [], 0, 0, [], [], 0)


def _activity(level="B1", duration=30, title="Jeu de rôle : À l'aéroport", steps=None):
    if steps is None:
        steps = [(1, "Préparation", 5), (2, "Jeu de rôle", 15), (3, "Mise en commun", 10)]
    return {
        "title": title, "level": level, "theme": "Le voyage", "activity_type": "Jeu de rôle",
        "duration": duration, "objective": "Développer l'interaction orale",
        "skills": ["Expression orale", "Interaction orale"],
        "materials": ["Cartes de rôles"],
        "instructions": "قَدّم نَفْسَكَ في المَطار.",
        "procedure": [{"step": step, "title": title, "duration": step_duration, "description": "Description concrète de l'étape."} for step, title, step_duration in steps],
        "teacher_role": "Distribue les cartes de rôles puis modèle la consigne.",
        "learner_role": "Joue son rôle en binôme en utilisant les structures données.",
        "expected_outcome": "Chaque binôme réalise un échange court à l'aéroport.",
        "assessment": {"criteria": ["Utilise le lexique du voyage", "Formule une question simple"]},
        "differentiation": {"support": "Fournir une liste de phrases modèles.", "standard": "Échange guidé par les cartes.", "advanced": "Ajouter un imprévu à gérer."},
    }


def test_a1_vocabulary_orale_fr_activity_is_validated():
    llm = FakeLLMProvider(json.dumps(_activity(level="A1", duration=15, steps=[(1, "Échauffement", 5), (2, "Activité", 8), (3, "Clôture", 2)]), ensure_ascii=False))
    request = ActivityGenerateIn(level="A1", theme="Famille", objective="Apprendre du vocabulaire", skills=["Expression orale"], activity_type="Activité orale", duration_minutes=15, language="fr")
    result = ActivityGenerationService(llm=llm).generate(request, _context())
    assert result.duration == 15
    assert result.level == "A1"
    assert sum(step.duration for step in result.procedure) == 15
    assert "rà request.duration_minutes" in llm.calls[0]["system_prompt"] or "duration" in llm.calls[0]["system_prompt"]
    assert llm.calls[0]["max_tokens"] == 3072
    assert llm.calls[0]["generation_options"].reasoning_effort == "low"


def test_b1_voyage_interaction_role_play_fr_is_validated():
    llm = FakeLLMProvider(json.dumps(_activity(level="B1", duration=30), ensure_ascii=False))
    request = ActivityGenerateIn(level="B1", theme="Voyage", objective="Développer l'interaction orale", skills=["Interaction orale"], activity_type="Jeu de rôle", duration_minutes=30, language="fr")
    result = ActivityGenerationService(llm=llm).generate(request, _context())
    assert result.level == "B1"
    assert result.duration == 30
    assert sum(step.duration for step in result.procedure) == 30
    assert result.activity_type == "Jeu de rôle"


def test_a1_family_arabic_activity_is_validated():
    activity = _activity(level="A1", duration=15, title="نشاط شفهي: الأسرة", steps=[(1, "تقديم", 5), (2, "النشاط", 8), (3, "الخاتمة", 2)])
    activity.update({"theme": "الأسرة", "objective": "تعلم مفردات الأسرة", "activity_type": "نشاط شفهي"})
    llm = FakeLLMProvider(json.dumps(activity, ensure_ascii=False))
    request = ActivityGenerateIn(level="A1", theme="الأسرة", objective="تعلم مفردات الأسرة", skills=["Expression orale"], activity_type="نشاط شفهي", duration_minutes=15, language="ar")
    result = ActivityGenerationService(llm=llm).generate(request, _context())
    assert result.theme == "الأسرة"
    assert result.title == "نشاط شفهي: الأسرة"
    assert sum(step.duration for step in result.procedure) == int(request.duration_minutes)


def test_rejects_incomplete_activity_form():
    with pytest.raises(Exception):
        ActivityGenerateIn()  # missing required theme/objective at Pydantic level


def test_rejects_an_incomplete_json_object():
    with pytest.raises(ActivityGenerationError, match="incomplet ou invalide"):
        ActivityGenerationService._json_object('{"title": "نشاط"')


def test_works_without_relevant_rag_documents():
    llm = FakeLLMProvider(json.dumps(_activity(), ensure_ascii=False))
    context = PedagogicalContext({"cefr_level": "B1", "language": "ar"}, [], [], [], 0, 0, [], ["No relevant document"], 0)
    result = ActivityGenerationService(llm=llm).generate(
        ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", skills=["Interaction orale"], activity_type="Jeu de rôle", duration_minutes=30),
        context,
    )
    assert result.rag_sources_used == 0


def test_rejects_invalid_llm_json_shape(caplog):
    invalid = {"titre": "Activité", "clés": ["invalides"]}
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(invalid, ensure_ascii=False)))
    with pytest.raises(ActivityGenerationError, match="format attendu"):
        service.generate(
            ActivityGenerateIn(level="A1", theme="الأسرة", objective="تعلم المفردات", duration_minutes=15, language="ar"),
            _context(),
        )
    assert "activity_schema_validation_failed" in caplog.text


def test_rejects_an_llm_timeout_as_a_provider_error():
    from app.services.llm_provider import LLMProviderError

    class TimedOutLLM(FakeLLMProvider):
        def generate(self, **_kwargs):
            raise LLMProviderError("provider timed out", provider_message="Timeout du fournisseur")

    service = ActivityGenerationService(llm=TimedOutLLM())
    with pytest.raises(ActivityGenerationError, match="Timeout du fournisseur"):
        service.generate(
            ActivityGenerateIn(level="A1", theme="Famille", objective="Vocabulaire", duration_minutes=15),
            _context(),
        )


def test_rejects_incoherent_step_durations():
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(_activity(steps=[(1, "Préparation", 5), (2, "Jeu de rôle", 15), (3, "Mise en commun", 20)]), ensure_ascii=False)))
    with pytest.raises(ActivityGenerationError, match="somme des durées"):
        service.generate(
            ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", activity_type="Jeu de rôle", duration_minutes=30),
            _context(),
        )


def test_rejects_a_header_duration_that_differs_from_the_request():
    activity = _activity(duration=20)
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(activity, ensure_ascii=False)))
    with pytest.raises(ActivityGenerationError, match="durée annoncée"):
        service.generate(
            ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", activity_type="Jeu de rôle", duration_minutes=30),
            _context(),
        )


def test_rejects_a_level_mismatch():
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(_activity(level="B2"), ensure_ascii=False)))
    with pytest.raises(ActivityGenerationError, match="niveau annoncé"):
        service.generate(
            ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", activity_type="Jeu de rôle", duration_minutes=30),
            _context(),
        )


def test_normalizes_duration_strings_before_pydantic_validation():
    activity = _activity()
    activity["duration"] = "30 minutes"
    for step in activity["procedure"]:
        step["duration"] = f'{step["duration"]} minutes'
    result = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(activity, ensure_ascii=False))).generate(
        ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", activity_type="Jeu de rôle", duration_minutes=30),
        _context(),
    )
    assert result.duration == 30
    assert [step.duration for step in result.procedure] == [5, 15, 10]


def test_extracts_a_complete_json_object_from_json_markdown_and_trailing_text():
    raw = "Voici :\n```json\n" + json.dumps(_activity(), ensure_ascii=False) + "\n```\nFin."
    parsed = ActivityGenerationService._json_object(raw)
    assert parsed["title"] == "Jeu de rôle : À l'aéroport"


def _run(activity, request=None):
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(activity, ensure_ascii=False)))
    return service.generate(
        request or ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", activity_type="Jeu de rôle", duration_minutes=30),
        _context(),
    )


def test_string_fields_remain_strings_when_already_strings():
    result = _run(_activity())
    assert isinstance(result.teacher_role, str)
    assert isinstance(result.learner_role, str)
    assert isinstance(result.expected_outcome, str)
    assert isinstance(result.differentiation.support, str)
    assert isinstance(result.differentiation.standard, str)
    assert isinstance(result.differentiation.advanced, str)


def test_single_element_string_lists_are_normalized_to_string():
    activity = _activity()
    activity.update({
        "teacher_role": ["L'enseignant présente les images et guide le dialogue."],
        "learner_role": ["Les apprenants jouent en binômes."],
        "expected_outcome": ["Les apprenants produisent un court échange."],
        "differentiation": {
            "support": ["Fournir un modèle de phrase simplifié."],
            "standard": ["Échange guidé par les cartes."],
            "advanced": ["Ajouter un imprévu à gérer."],
        },
    })
    result = _run(activity)
    strict = [
        result.teacher_role, result.learner_role, result.expected_outcome,
        result.differentiation.support, result.differentiation.standard, result.differentiation.advanced,
    ]
    assert all(isinstance(value, str) for value in strict)
    assert result.teacher_role == "L'enseignant présente les images et guide le dialogue."
    assert result.differentiation.support == "Fournir un modèle de phrase simplifié."


def test_multi_element_string_lists_are_merged_without_losing_information():
    activity = _activity()
    activity["teacher_role"] = ["Présente le lexique.", "Modèle un mini-dialogue.", "Corrige les erreurs."]
    result = _run(activity)
    assert result.teacher_role == "Présente le lexique.\nModèle un mini-dialogue.\nCorrige les erreurs."
    assert len(result.teacher_role.split("\n")) == 3


def test_fully_arabic_response_with_bare_arrays_is_validated():
    activity = _activity(level="B1", duration=30, title="حوار في المطار", steps=[(1, "تمهيد", 5), (2, "الحوار", 15), (3, "مناقشة", 10)])
    activity.update({
        "theme": "رحلة", "activity_type": "نشاط شفهي", "objective": "التفاعل الشفهي",
        "teacher_role": ["يُقدّم الصور والأسئلة، يُوجه الحوار، يُصحح الأخطاء، يُسجل العبارات الجيدة"],
        "learner_role": ["يُكتب الإجابات، يشارك في الحوار مع شريك، يشارك في التلخيص، يكتب جملة ختامية"],
        "expected_outcome": ["الطلاب يستخدمون عبارات الاستفسار والاقتراح في سياق رحلة، ويظهرون قدرة على التفاعل مع شريك"],
        "differentiation": {
            "support": ["نموذج جملة مبسطة"],
            "standard": ["الطلاب يبدؤون بطرح سؤال"],
            "advanced": ["الطلاب يضيفون تفاصيل إضافية"],
        },
    })
    result = _run(activity)
    assert result.title == "حوار في المطار"
    assert result.teacher_role.startswith("يُقدّم الصور")
    assert result.learner_role.startswith("يُكتب الإجابات")
    assert result.expected_outcome.startswith("الطلاب يستخدمون")
    assert result.differentiation.advanced.startswith("الطلاب يضيفون")


def test_skills_remains_a_list():
    result = _run(_activity())
    assert isinstance(result.skills, list)
    assert result.skills == ["Expression orale", "Interaction orale"]


def test_materials_remains_a_list():
    result = _run(_activity())
    assert isinstance(result.materials, list)
    assert "Cartes de rôles" in result.materials


def test_procedure_remains_a_list():
    result = _run(_activity())
    assert isinstance(result.procedure, list)
    assert len(result.procedure) == 3
    assert result.procedure[0].title == "Préparation"


def test_assessment_criteria_remains_a_list():
    result = _run(_activity())
    assert isinstance(result.assessment.criteria, list)
    assert len(result.assessment.criteria) == 2


def test_invalid_json_behavior_is_preserved():
    with pytest.raises(ActivityGenerationError, match="incomplet ou invalide"):
        ActivityGenerationService._json_object('{"title": "نشاط"')


def test_normalizer_join_preserves_multiple_string_elements():
    payload = _activity()
    payload["teacher_role"] = ["A", "B", "C"]
    normalized = ActivityGenerationService._normalize_payload(payload)
    assert normalized["teacher_role"] == "A\nB\nC"


def test_normalizer_leaves_non_string_values_untouched():
    payload = _activity()
    payload["teacher_role"] = 42
    normalized = ActivityGenerationService._normalize_payload(payload)
    assert normalized["teacher_role"] == 42


def _quality_activity(*, title, level, theme, activity_type, duration, steps, objective, skills, teacher_role, learner_role, expected_outcome, instructions):
    return {
        "title": title, "level": level, "theme": theme, "activity_type": activity_type,
        "duration": duration, "objective": objective, "skills": skills,
        "materials": ["Tableau"], "instructions": instructions,
        "procedure": [{"step": step, "title": step_title, "duration": step_duration, "description": f"Consigne concrète de l'étape {step} avec modèle : « أين تسكن؟ » -> « أسكن في الرباط. »"} for step, step_title, step_duration in steps],
        "teacher_role": teacher_role, "learner_role": learner_role, "expected_outcome": expected_outcome,
        "assessment": {"criteria": ["Critère observable 1", "Critère observable 2"]},
        "differentiation": {"support": "Fournir un modèle de phrase.", "standard": "Tâche de niveau demandé.", "advanced": "Ajouter une contrainte supplémentaire."},
    }


def _generate(activity, request):
    return ActivityGenerationService(llm=FakeLLMProvider(json.dumps(activity, ensure_ascii=False))).generate(request, _context())


def test_quality_a1_family_vocabulary_oral_15min_is_simple_and_valid():
    activity = _quality_activity(
        title="نشاط: أفراد العائلة", level="A1", theme="Famille", activity_type="Activité orale", duration=15,
        steps=[(1, "المفردات", 5), (2, "التدريب", 8), (3, "الخاتمة", 2)],
        objective="Apprendre le vocabulaire de la famille",
        skills=["Expression orale"],
        teacher_role="يقدم المعلم صور العائلة، يشد التعليقات، ويقول أسماء أفراد العائلة.",
        learner_role="يسمّي المتعلمون أفراد العائلة عن الصور ويكررون الجمل مع الشريك.",
        expected_outcome="يسمّي المتعلمون أربعة أفراد من العائلة.",
        instructions="انظر إلى الصورة وسمِّ أفراد العائلة.",
    )
    result = _generate(activity, ActivityGenerateIn(level="A1", theme="Famille", objective="Apprendre le vocabulaire de la famille", skills=["Expression orale"], activity_type="Activité orale", duration_minutes=15, language="ar"))
    assert result.level == "A1"
    assert sum(step.duration for step in result.procedure) == 15
    assert all("جمل معقدة" not in getattr(result, field) for field in ("teacher_role", "learner_role", "expected_outcome"))


def test_quality_a2_school_expression_orale_20min_is_valid():
    activity = _quality_activity(
        title="نشاط: مدرستي", level="A2", theme="École", activity_type="Activité orale", duration=20,
        steps=[(1, "تقديم", 5), (2, "الممارسة", 10), (3, "الخاتمة", 5)],
        objective="Développer l'expression orale",
        skills=["Expression orale"],
        teacher_role="يطرح أسئلة بسيطة عن المدرسة ويساعد المتعلمين على وصفها.",
        learner_role="يصف المتعلمون مدرستهم بجمل بسيطة ويجيبون عن الأسئلة.",
        expected_outcome="يستطيع المتعلمون وصف مدرستهم بجمل بسيطة.",
        instructions="صف مدرستك لزميلك ثم أجب عن أسئلته.",
    )
    result = _generate(activity, ActivityGenerateIn(level="A2", theme="École", objective="Développer l'expression orale", skills=["Expression orale"], activity_type="Activité orale", duration_minutes=20, language="ar"))
    assert result.level == "A2"
    assert sum(step.duration for step in result.procedure) == 20


def test_quality_b1_travel_role_play_30min_avoids_complex_sentence_demand():
    activity = _quality_activity(
        title="التخطيط لرحلة ثقافية إلى المغرب", level="B1", theme="Le voyage", activity_type="Jeu de rôle", duration=30,
        steps=[(1, "التحضير", 5), (2, "لعب الأدوار", 20), (3, "المناقشة", 5)],
        objective="Développer l'interaction orale",
        skills=["Interaction orale"],
        teacher_role="يُشجع المعلم الطلاب على الإجابة بجمل واضحة ومترابطة، مع تقديم أسباب وتفاصيل مناسبة لمستوى B1.",
        learner_role="يخطط المتعلمون لرحلة، يتبادلون الأفكار ويجيبون عن أسئلة المجموعات الأخرى.",
        expected_outcome="يستطيع المتعلمون تقديم خطة سفر قصيرة وطرح أسئلة حولها والإجابة عنها.",
        instructions="اعملوا في مجموعات، خططوا لرحلة ثقافية، ثم اعرضوا خطتكم وأجيبوا عن أسئلة زملائكم.",
    )
    result = _generate(activity, ActivityGenerateIn(level="B1", theme="Voyage", objective="Développer l'interaction orale", skills=["Interaction orale"], activity_type="Jeu de rôle", duration_minutes=30, language="ar"))
    assert result.level == "B1"
    assert sum(step.duration for step in result.procedure) == 30
    assert "جمل معقدة" not in result.teacher_role
    assert "جمل واضحة ومترابطة" in result.teacher_role


def test_quality_b2_moroccan_culture_debate_45min_is_valid():
    activity = _quality_activity(
        title="نقاش: الثقافة المغربية", level="B2", theme="Culture marocaine", activity_type="Débat", duration=45,
        steps=[(1, "تقديم الإشكالية", 5), (2, "النقاش", 30), (3, "الخلاصة", 10)],
        objective="Argumenter et défendre une position",
        skills=["Interaction orale"],
        teacher_role="يقدم الإشكالية، يوزع الأدوار ويشجع على الحجج والردود.",
        learner_role="يتخذ المتعلمون موقفًا، يقدمون حججًا ويجيبون عن اعتراضات الآخرين.",
        expected_outcome="يستطيع المتعلمون عرض موقف ودفاعه بحجج منظمة.",
        instructions="خذ موقفًا حول الموضوع، قدم حججك وأجب عن اعتراضات زملائك.",
    )
    result = _generate(activity, ActivityGenerateIn(level="B2", theme="Culture marocaine", objective="Argumenter et défendre une position", skills=["Interaction orale"], activity_type="Débat", duration_minutes=45, language="ar"))
    assert result.level == "B2"
    assert result.activity_type == "Débat"
    assert sum(step.duration for step in result.procedure) == 45


def test_quality_arabic_a1_family_vocabulary_oral_15min_is_valid():
    activity = _quality_activity(
        title="نشاط شفهي: الأسرة", level="A1", theme="الأسرة", activity_type="نشاط شفهي", duration=15,
        steps=[(1, "مفردات", 5), (2, "التدريب", 7), (3, "إنتاج", 3)],
        objective="تعلم مفردات الأسرة", skills=["التعبير الشفهي"],
        teacher_role="يقدم المفردات ويصحح النطق.",
        learner_role="يسمي أفراد الأسرة ويكرر الجمل.",
        expected_outcome="يسمي المتعلمون أفراد الأسرة.",
        instructions="سمِّ أفراد الأسرة في الصورة.",
    )
    result = _generate(activity, ActivityGenerateIn(level="A1", theme="الأسرة", objective="تعلم مفردات الأسرة", skills=["التعبير الشفهي"], activity_type="نشاط شفهي", duration_minutes=15, language="ar"))
    assert result.theme == "الأسرة"
    assert sum(step.duration for step in result.procedure) == 15


def test_quality_arabic_b1_travel_role_play_30min_is_valid():
    activity = _quality_activity(
        title="لعب الأدوار: في المطار", level="B1", theme="السفر", activity_type="لعب الأدوار", duration=30,
        steps=[(1, "تحضير", 5), (2, "أدوار", 20), (3, "تقييم", 5)],
        objective="التفاعل الشفهي", skills=["التفاعل الشفهي"],
        teacher_role="يوجه الحوار، يقدم أسئلة مساعدة ويقدم تغذية راجعة.",
        learner_role="يلعب المتعلمون دور المسافر وموظف الاستقبال ويتبادلون الأسئلة.",
        expected_outcome="يطرح المتعلمون أسئلة عن الرحلة ويجيبون عنها بجمل واضحة.",
        instructions="تحدث مع شريكك: أحدهما مسافر والآخر موظف استقبال.",
    )
    result = _generate(activity, ActivityGenerateIn(level="B1", theme="السفر", objective="التفاعل الشفهي", skills=["التفاعل الشفهي"], activity_type="لعب الأدوار", duration_minutes=30, language="ar"))
    assert result.level == "B1"
    assert result.activity_type == "لعب الأدوار"
    assert sum(step.duration for step in result.procedure) == 30


def test_rejects_a_complex_sentence_requirement_at_b1():
    activity = _activity(level="B1", duration=30)
    activity["teacher_role"] = "يُشجع المعلم الطلاب على الرد باستخدام جمل معقدة"
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(activity, ensure_ascii=False)))
    with pytest.raises(ActivityGenerationError, match="phrases complexes"):
        service.generate(
            ActivityGenerateIn(level="B1", theme="Voyage", objective="Interagir", activity_type="Jeu de rôle", duration_minutes=30),
            _context(),
        )


def test_complex_sentence_requirement_still_allowed_above_b1():
    activity = _activity(level="B2", duration=30)
    activity["teacher_role"] = "Encourage les apprenants à formuler des phrases complexes et nuancées."
    service = ActivityGenerationService(llm=FakeLLMProvider(json.dumps(activity, ensure_ascii=False)))
    result = service.generate(
        ActivityGenerateIn(level="B2", theme="Culture", objective="Argumenter", activity_type="Débat", duration_minutes=30),
        _context(),
    )
    assert result.level == "B2"


def test_prompt_contains_cefr_adaptation_and_corrected_b1_phrasing():
    prompt = ActivityGenerationService._build_system_prompt()
    assert "expert en didactique de la langue arabe, en CECRL" in prompt
    assert "جمل واضحة ومترابطة" in prompt
    assert "جمل معقدة" in prompt
    assert "interaction orale" in prompt.lower()
    assert "التخطيط لرحلة ثقافية إلى المغرب" in prompt
    assert "A2" in prompt and "B2" in prompt and "C1" in prompt and "C2" in prompt
