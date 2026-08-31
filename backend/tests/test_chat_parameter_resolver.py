from app.services.chat_parameter_resolver import ChatParameterResolver


def resolve(message, **kwargs):
    return ChatParameterResolver().resolve(message=message, cefr_level=kwargs.get("level"), skills=kwargs.get("skills", ()), language=kwargs.get("language"), user_history=kwargs.get("history", ()))


def test_historical_semantic_text_removes_only_recognized_control_signals():
    text = "Propose-moi une activité orale A1 sur la famille avec Amina."

    assert ChatParameterResolver.historical_semantic_text(text) == "Propose-moi une sur la famille avec Amina."


def test_french_family_speaking_request_resolves_level_skill_and_language():
    result = resolve("Propose-moi une activité orale A1 sur la famille.")
    assert (result.cefr_level, result.skills, result.response_language) == ("A1", ("speaking",), "fr")


def test_specific_skill_phrases_and_arabic_target_are_not_confused():
    result = resolve("Propose une activité A1 en arabe.")
    assert result.cefr_level == "A1" and result.skills == () and result.response_language == "fr"
    assert resolve("Réponds en arabe.").response_language == "ar"
    assert resolve("compréhension orale A2").skills == ("listening",)
    assert resolve("production écrite B1").skills == ("writing",)


def test_skill_phrases_are_specific_and_multiple_skills_keep_canonical_order():
    assert resolve("lecture A2").skills == ("reading",)
    assert resolve("production orale B1").skills == ("speaking",)
    assert resolve("activité de lecture puis expression écrite").skills == ("reading", "writing")


def test_explicit_response_language_commands_override_message_language_detection():
    assert resolve("Réponds en français.").response_language == "fr"
    assert resolve("Answer in English.").response_language == "en"
    assert resolve("Responde en español.").response_language == "es"
    assert resolve("أجب بالعربية.").response_language == "ar"


def test_multilingual_and_no_implicit_level():
    assert resolve("أريد نشاطا للمستوى A1 في التعبير الشفهي").response_language == "ar"
    assert resolve("I need a speaking activity B1").skills == ("speaking",)
    assert resolve("Quiero una actividad de lectura A2").skills == ("reading",)
    assert resolve("activité débutant").cefr_level is None


def test_api_values_and_current_user_message_override_user_history_only():
    history = ("activité orale A1", "rien d'autre")
    carried = resolve("Rends-la plus facile.", history=history)
    assert (carried.cefr_level, carried.skills) == ("A1", ("speaking",))
    current = resolve("Maintenant niveau A2 et lecture", history=history)
    assert (current.cefr_level, current.skills) == ("A2", ("reading",))
    explicit = resolve("activité de lecture A1", level="B1", skills=("speaking",), language="es", history=history)
    assert (explicit.cefr_level, explicit.skills, explicit.response_language) == ("B1", ("speaking",), "es")


def test_message_language_is_carried_from_user_history_and_ambiguous_input_stays_unresolved():
    result = resolve("Rends-la plus facile.", history=("I need a speaking activity A1",))
    assert result.response_language == "fr"
    assert result.language_source == "message_detection"

    inherited = resolve("Plus court.", history=("I need a speaking activity A1",))
    assert inherited.response_language == "en"
    assert inherited.language_source == "history"

    ambiguous = resolve("A1")
    assert ambiguous.response_language is None
