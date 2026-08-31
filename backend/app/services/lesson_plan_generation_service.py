"""One-call, validated lesson-plan generation using the existing RAG boundary."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pydantic import ValidationError
from app.core.config import Settings, get_settings
from app.schemas.lesson_plan import LessonPlanGenerateIn, LessonPlanOut
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError
from app.services.pedagogical_knowledge_service import PedagogicalContext

logger = logging.getLogger(__name__)


class LessonPlanGenerationError(RuntimeError):
    pass


class LessonPlanGenerationService:
    _REQUIRED_PHASES = (
        "découverte",
        "compréhension",
        "pratique guidée",
        "production",
        "évaluation",
    )

    def __init__(self, *, llm: LLMProvider, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    @staticmethod
    def _json_object(raw: str) -> dict:
        text = raw.lstrip("\ufeff \t\r\n")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Providers occasionally add prose or ```json fences despite the
            # instruction. Decode the first complete object instead of relying
            # on the last brace, which may belong to a trailing explanation.
            decoder = json.JSONDecoder()
            start = text.find("{")
            while start >= 0:
                try:
                    value, _ = decoder.raw_decode(text, start)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    pass
                start = text.find("{", start + 1)
            raise LessonPlanGenerationError("Le JSON de la fiche pédagogique est incomplet ou invalide.")

    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        """Normalize only known LLM type slips before strict Pydantic validation."""
        normalized = payload.copy()

        def duration_as_int(value: object) -> object:
            if isinstance(value, str):
                match = re.fullmatch(r"\s*(\d+)\s+minutes?\s*", value, flags=re.IGNORECASE)
                if match:
                    return int(match.group(1))
            return value

        normalized["duration"] = duration_as_int(normalized.get("duration"))
        lesson_flow = normalized.get("lesson_flow")
        if isinstance(lesson_flow, list):
            normalized["lesson_flow"] = [
                {**step, "duration": duration_as_int(step.get("duration"))}
                if isinstance(step, dict) else step
                for step in lesson_flow
            ]

        extension = normalized.get("extension")
        if isinstance(extension, dict):
            normalized_extension = extension.copy()
            for field in ("homework", "follow_up"):
                value = normalized_extension.get(field)
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    normalized_extension[field] = "\n".join(value)
            normalized["extension"] = normalized_extension
        return normalized

    @staticmethod
    def _validate_pedagogical_consistency(plan: LessonPlanOut, request: LessonPlanGenerateIn) -> None:
        """Reject only objective output defects that a JSON schema cannot express."""
        if plan.duration != request.duration_minutes:
            raise LessonPlanGenerationError(
                f"La durée annoncée ({plan.duration} min) doit correspondre à la durée demandée ({request.duration_minutes} min)."
            )
        total = sum(step.duration for step in plan.lesson_flow)
        if total != request.duration_minutes:
            raise LessonPlanGenerationError(
                f"La somme des durées des étapes ({total} min) doit correspondre à la durée demandée ({request.duration_minutes} min)."
            )
        def normalized(value: str) -> str:
            return " ".join(
                "".join(
                    character for character in unicodedata.normalize("NFD", value.casefold())
                    if unicodedata.category(character) != "Mn"
                ).split()
            )

        expected_phases = tuple(normalized(phase) for phase in LessonPlanGenerationService._REQUIRED_PHASES)
        actual_phases = tuple(normalized(step.phase) for step in plan.lesson_flow)
        if actual_phases != expected_phases:
            raise LessonPlanGenerationError(
                "La fiche doit suivre, dans cet ordre, les phases : découverte, compréhension, pratique guidée, production et évaluation."
            )

        generic = {"guide", "facilitateur", "discussion", "activite", "activity"}
        for step in plan.lesson_flow:
            required_fields = {
                "objectif": step.objective,
                "rôle de l’enseignant": step.teacher_role,
                "activité des apprenants": step.learner_activity,
                "consigne": step.instructions,
                "modalité de travail": step.work_mode,
                "exemple": step.example,
                "résultat attendu": step.expected_result,
            }
            empty_field = next((name for name, value in required_fields.items() if not value.strip()), None)
            if empty_field:
                raise LessonPlanGenerationError(f"La fiche générée ne précise pas {empty_field} pour une étape.")
            if normalized(step.teacher_role) in generic or normalized(step.learner_activity) in generic:
                raise LessonPlanGenerationError("La fiche générée contient une activité trop générique. Réessayez la génération.")
            if not step.materials or any(not item.strip() for item in step.materials):
                raise LessonPlanGenerationError("Chaque étape doit indiquer un matériel ou une ressource concret(e).")
        if not plan.materials or any(not item.strip() for item in plan.materials):
            raise LessonPlanGenerationError("La fiche doit lister le matériel global réellement nécessaire.")
        if not plan.communicative_objectives or not plan.linguistic_objectives:
            raise LessonPlanGenerationError("La fiche doit séparer les objectifs communicatifs et linguistiques.")
        if not plan.assessment.activity.strip() or not plan.assessment.instructions.strip() or not plan.assessment.moment.strip():
            raise LessonPlanGenerationError("L’évaluation doit préciser son moment, son activité et sa consigne.")

    def generate(self, request: LessonPlanGenerateIn, context: PedagogicalContext) -> LessonPlanOut:
        sources = [
            {"title": block.document_title, "pages": [block.page_start, block.page_end], "content": block.content[:700]}
            for block in context.resource_blocks[:2]
        ]
        system = """Tu es un concepteur expert de séquences pédagogiques pour l'enseignement de la langue arabe, spécialisé dans le CECRL et dans l'adaptation aux enfants, adolescents et adultes. Produis une fiche directement exploitable en classe : ce n'est jamais un formulaire générique rempli par une IA.
Retourne uniquement un objet JSON valide, sans Markdown, sans bloc ```json et sans aucun texte avant ou après le JSON. Respecte exactement le schéma Pydantic existant.
Utilise exactement ces clés anglaises (ne les traduis pas) : title, level, theme, duration, audience, session_type, skills, age_approximation, communicative_objectives, linguistic_objectives, general_objective, specific_objectives, prerequisites, linguistic_content, materials, lesson_flow, assessment, differentiation, extension.
lesson_flow est une liste de 5 objets avec exactement : phase, duration, objective, teacher_role, learner_activity, instructions, materials, work_mode, example, expected_result.
assessment contient : assessment_type, moment, method, activity, instructions, criteria, success_indicators, rubric. Chaque objet rubric contient : criterion, achieved, to_reinforce. differentiation contient : support, extension. extension contient : homework, follow_up.
Règles de type impératives : duration est un entier JSON uniquement, par exemple 60, jamais "60 minutes". lesson_flow[].duration est un entier JSON uniquement, par exemple 10, jamais "10 minutes". extension.homework est une chaîne JSON, jamais un tableau ; extension.follow_up est une chaîne JSON, jamais un tableau.
Règle JSON impérative : chaque champ de liste est toujours un tableau JSON de chaînes, même avec un seul élément ou aucun élément. N'envoie jamais une chaîne seule pour un champ de liste.
Les champs obligatoirement list[str] sont : skills: ["expression orale"], communicative_objectives: ["..."], linguistic_objectives: ["..."], specific_objectives: ["..."], prerequisites: ["..."], materials: ["Cartes d'images"], lesson_flow[].materials: ["Cartes d'images"], assessment.criteria: ["..."], assessment.success_indicators: ["..."], differentiation.support: ["..."], differentiation.extension: ["..."].
linguistic_content est un objet dont chaque valeur est list[str], par exemple {"Vocabulaire": ["الأب — père"], "Point grammatical": ["هذا / هذه"]}. lesson_flow et assessment.rubric sont aussi des tableaux JSON d'objets, jamais des objets ou chaînes seuls.
Utilise réellement tous les paramètres de request : niveau CECRL, âge ou audience, thème, objectif général, compétences, durée, type de séance, prérequis, points linguistiques, instructions spéciales, langue et contexte fourni. Ne laisse aucun paramètre pertinent sans effet sur la difficulté, le lexique, les structures, les activités, le matériel, la production ou l'évaluation. Formule des objectifs communicatifs, linguistiques et spécifiques observables, mesurables et cohérents entre eux ; évite les objectifs vagues.
Construis une progression didactique cohérente et non répétitive selon la chaîne obligatoire « objectif → contenu linguistique → activités → production → évaluation ». Toute structure grammaticale ou lexicale introduite doit être réutilisée dans les activités, mobilisée dans la production et évaluée à la fin. lesson_flow contient exactement, dans cet ordre : Découverte, Compréhension, Pratique guidée, Production, Évaluation ; attribue à chacune une fonction pédagogique distincte, adaptée à la durée et aux compétences visées. La somme de ses durées égale exactement request.duration_minutes et chaque durée est réaliste. Pour les débutants, particulièrement les enfants, privilégie image → écoute → répétition → compréhension → manipulation → interaction → production orale, puis lecture ou écriture courte seulement si pertinente. Évite les phases longues et uniformes.
Chaque étape contient objective, work_mode, materials réels, teacher_role et learner_activity précis, instructions directement prononçables, example concret et expected_result observable. Indique quoi faire, comment, avec quel support, quelle langue les apprenants produisent et quel résultat est attendu. Interdis les formulations vagues comme « faire une activité ludique », « les élèves participent », « l'enseignant explique », « guide » ou « discussion ». Choisis des activités variées et adaptées : cartes-images, association, loto, memory, mime, devinettes, classement, dialogue, jeu de rôle, enquête orale, dessin, description ou mini-défi, sans les répéter mécaniquement. N'invente jamais des ressources présumées disponibles : privilégie tableau, cartes-images, feuilles imprimées, fiches, ardoise et matériel créé par l'enseignant. Une ressource audio, numérique, vidéo, plateforme, lien ou livre spécifique ne peut être proposée que comme optionnelle et explicitement signalée comme telle.
La fiche enseigne réellement l'arabe : le contenu arabe doit être riche, naturel, réutilisable et présent dans les activités, pas seulement quelques mots isolés. Privilégie les exemples, consignes, questions, réponses, expressions et activités en arabe ; ne fournis une traduction française que lorsqu'elle apporte une aide pédagogique claire. linguistic_content emploie, seulement si pertinent, les catégories « المفردات », « التراكيب اللغوية », « العبارات التواصلية », « الأسئلة », « الإجابات », « أجوبة نموذجية », « أمثلة », « حوار قصير », « إنتاج المتعلم », « تعليمات بسيطة », « التقويم », « Trace écrite », « Point grammatical » et « Prononciation ». Chaque élément arabe a une fonction pédagogique et est réemployé dans une activité. Organise l'acquisition : mot, groupe nominal simple, phrase modèle, question, réponse, interaction puis courte production personnelle. Adapte la richesse lexicale au niveau : à A1, lexique fréquent, concret et simple, phrases courtes et forte guidance ; lorsque le thème et la durée le permettent, vise généralement 8 à 12 éléments lexicaux pertinents, sans ajout artificiel. Aux niveaux supérieurs, enrichis progressivement vocabulaire, structures et autonomie sans dépasser le CECRL.
Dans teacher_role, learner_activity, instructions, example, expected_result, assessment et extension, donne des modèles arabes immédiatement utilisables lorsque l'activité est linguistique : questions de l'enseignant, réponses attendues, phrases modèles, court dialogue ou production de l'apprenant. Pour toute séance communicative, lorsque pertinent, inclue des questions en arabe, leurs réponses modèles, une mini-interaction enseignant/apprenant ou apprenant/apprenant, puis une production personnelle. Pour A1 et les thèmes communicatifs, utilise des formulations simples, naturelles et réellement utiles à l'apprenant ; par exemple pour présenter la famille, privilégie « مَنْ هٰذَا؟ / هٰذَا أَبِي. » et « مَنْ هٰذِهِ؟ / هٰذِهِ أُمِّي. », plutôt que de faire de « هو + اسم / هي + اسم » la structure principale si elle ne correspond pas à l'objectif. Si request.language est ar, écris les consignes et contenus destinés aux apprenants en arabe standard naturel et correct. Vérifie mentalement toute langue arabe avant la sortie : orthographe, grammaire, accords, genre, nombre, pronoms, démonstratifs, conjugaison, prépositions et formulation naturelle des questions. La vocalisation est admise à A1 si elle aide réellement l'apprentissage, sans surcharger.
Si le public est 6–8 ans, choisis des activités courtes, dynamiques, visuelles, manipulables et avec mouvement, répétition, encouragement et productions brèves ; évite les longues explications grammaticales. Pour adolescents et adultes, adapte naturellement les thèmes, l'autonomie, les interactions et les productions. Si le contexte MRE ou marocain est explicitement pertinent au thème, intègre un élément culturel utile ; distingue toujours الفصحى d'une éventuelle darija et ne les mélange pas sans objectif pédagogique.
Les prérequis sont justifiés : si aucun n'est réellement nécessaire, indique « Aucun prérequis spécifique » dans la liste. assessment est alignée aux objectifs et vérifie exclusivement une compétence, un lexique et les structures réellement travaillés ; elle précise moment, activité, consigne, critères, indicateurs de réussite et niveau attendu avec, lorsque pertinent, question en arabe, réponse attendue et grille simple. N'introduis jamais dans l'évaluation une structure plus difficile ou différente de celle enseignée. differentiation propose des aides concrètes liées aux activités (cartes-images, modèle, choix, répétition, aide phonétique ou vocalisation) et des extensions concrètes (lexique, question, mini-dialogue ou production plus complexe). La « Trace écrite », si pertinente, est courte, adaptée au niveau et reprend le vocabulaire, les structures essentielles et des exemples arabes. extension réemploie le contenu arabe et propose un prolongement utile, jamais un devoir vague.
Les sources RAG sont un appui à sélectionner, synthétiser et adapter lorsqu'elles sont pertinentes : ne copie pas aveuglément, n'invente aucune provenance ni ressource. Reste structuré et lisible, sans longues explications théoriques : produis des contenus directement utilisables en classe. Avant de répondre, contrôle silencieusement : objectifs alignés aux activités, structures enseignées réutilisées et évaluées, CECRL et âge respectés, A1 simple et naturel, durée exacte, progression logique, arabe correct et suffisamment présent, activités concrètes et communicatives, production apprenant claire, évaluation alignée, différenciation et prolongement pertinents, ressources réalistes et aucune contradiction. En cas de conflit, priorise pertinence, cohérence, exactitude linguistique, adaptation au niveau, exploitabilité en classe et richesse fonctionnelle de l'arabe."""
        user = json.dumps({"request": request.model_dump(), "rag_resources": sources}, ensure_ascii=False)
        try:
            result = self.llm.generate(
                system_prompt=system, user_prompt=user, temperature=0.2,
                max_tokens=self.settings.lesson_plan_max_output_tokens,
                generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
            )
        except LLMProviderError as exc:
            raise LessonPlanGenerationError(exc.provider_message) from exc
        if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
            raise LessonPlanGenerationError("La génération a été tronquée. Réduisez les consignes puis réessayez.")
        try:
            payload = self._json_object(result.text)
        except LessonPlanGenerationError:
            logger.warning("lesson_plan_json_parse_failed provider=%s raw_response=%r", result.model, result.text)
            raise
        try:
            plan = LessonPlanOut.model_validate(self._normalize_payload(payload))
        except ValidationError as exc:
            logger.warning(
                "lesson_plan_schema_validation_failed provider=%s errors=%s raw_response=%r",
                result.model, exc.errors(include_url=False), result.text,
            )
            raise LessonPlanGenerationError("La fiche générée ne respecte pas le format attendu.") from exc
        self._validate_pedagogical_consistency(plan, request)
        return plan.model_copy(update={"rag_sources_used": len(sources), "provider_model": result.model})
