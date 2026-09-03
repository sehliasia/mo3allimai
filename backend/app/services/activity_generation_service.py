"""One-call, validated activity generation using the existing RAG boundary."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pydantic import ValidationError
from app.core.config import Settings, get_settings
from app.schemas.activity_generator import ActivityGenerateIn, ActivityOut
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError
from app.services.pedagogical_knowledge_service import PedagogicalContext

logger = logging.getLogger(__name__)


class ActivityGenerationError(RuntimeError):
    pass


class ActivityGenerationService:
    def __init__(self, *, llm: LLMProvider, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    @staticmethod
    def _json_object(raw: str) -> dict:
        text = raw.lstrip("\ufeff \t\r\n")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
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
            raise ActivityGenerationError("Le JSON de l'activité est incomplet ou invalide.")

    @staticmethod
    def _join_or_string(value: object) -> object:
        """Return a non-empty list of strings as a single string (information-preserving)."""
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return "\n".join(value)
        return value

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
        procedure = normalized.get("procedure")
        if isinstance(procedure, list):
            normalized["procedure"] = [
                {**step, "duration": duration_as_int(step.get("duration"))}
                if isinstance(step, dict) else step
                for step in procedure
            ]
        for field in ("teacher_role", "learner_role", "expected_outcome"):
            normalized[field] = ActivityGenerationService._join_or_string(normalized.get(field))

        differentiation = normalized.get("differentiation")
        if isinstance(differentiation, dict):
            normalized_differentiation = differentiation.copy()
            for field in ("support", "standard", "advanced"):
                normalized_differentiation[field] = ActivityGenerationService._join_or_string(
                    normalized_differentiation.get(field)
                )
            normalized["differentiation"] = normalized_differentiation
        return normalized

    @staticmethod
    def _normalized(value: str) -> str:
        return " ".join(
            "".join(
                character for character in unicodedata.normalize("NFD", value.casefold())
                if unicodedata.category(character) != "Mn"
            ).split()
        )

    def _validate_pedagogical_consistency(self, activity: ActivityOut, request: ActivityGenerateIn) -> None:
        """Reject only objective output defects that a JSON schema cannot express."""
        if activity.level != request.level:
            raise ActivityGenerationError(
                f"Le niveau annoncé ({activity.level}) doit correspondre au niveau demandé ({request.level})."
            )
        if activity.duration != request.duration_minutes:
            raise ActivityGenerationError(
                f"La durée annoncée ({activity.duration} min) doit correspondre à la durée demandée "
                f"({request.duration_minutes} min)."
            )
        total = sum(step.duration for step in activity.procedure)
        if total != request.duration_minutes:
            raise ActivityGenerationError(
                f"La somme des durées des étapes ({total} min) doit correspondre à la durée demandée "
                f"({request.duration_minutes} min)."
            )
        if not activity.title.strip():
            raise ActivityGenerationError("L'activité générée doit avoir un titre.")
        if not activity.instructions.strip():
            raise ActivityGenerationError("L'activité générée doit fournir une consigne.")
        if not activity.objective.strip():
            raise ActivityGenerationError("L'activité générée doit préciser un objectif.")
        if not activity.skills:
            raise ActivityGenerationError("L'activité générée doit indiquer au moins une compétence.")
        if not activity.teacher_role.strip() or not activity.learner_role.strip():
            raise ActivityGenerationError("L'activité générée doit préciser le rôle de l'enseignant et celui des apprenants.")
        if not activity.assessment.criteria or any(not criterion.strip() for criterion in activity.assessment.criteria):
            raise ActivityGenerationError("L'activité générée doit fournir des critères d'évaluation observables.")
        for step in activity.procedure:
            if not step.title.strip() or not step.description.strip():
                raise ActivityGenerationError("Chaque étape doit comporter un titre et une description.")
        if request.level in {"A1", "A2", "B1"} and self._demands_complex_sentences(
            activity.teacher_role, activity.learner_role, activity.expected_outcome, activity.instructions
        ):
            raise ActivityGenerationError(
                "L'activité exige à tort des phrases complexes pour ce niveau CECRL. Précisez un attendu simple, clair et accessible."
            )

    @staticmethod
    def _demands_complex_sentences(*values: str) -> bool:
        """Detect LLM slippage demanding 'complex sentences' at low to mid CEFR levels."""
        forbidden = {
            ActivityGenerationService._normalized(pattern)
            for pattern in ("جمل معقدة", "جمل صعبة", "عبارات معقدة", "phrases complexes", "جمل معقدة ومعقدة")
        }
        return any(
            any(forbidden_item in ActivityGenerationService._normalized(value) for forbidden_item in forbidden)
            for value in values
            if value
        )

    def generate(self, request: ActivityGenerateIn, context: PedagogicalContext) -> ActivityOut:
        sources = [
            {"title": block.document_title, "pages": [block.page_start, block.page_end], "content": block.content[:700]}
            for block in context.resource_blocks[:2]
        ]
        system = self._build_system_prompt()
        user = json.dumps({"request": request.model_dump(), "rag_resources": sources}, ensure_ascii=False)
        try:
            result = self.llm.generate(
                system_prompt=system, user_prompt=user, temperature=0.2,
                max_tokens=self.settings.activity_max_output_tokens,
                generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
            )
        except LLMProviderError as exc:
            raise ActivityGenerationError(exc.provider_message) from exc
        if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
            raise ActivityGenerationError("La génération a été tronquée. Réduisez les consignes puis réessayez.")
        try:
            payload = self._json_object(result.text)
        except ActivityGenerationError:
            logger.warning("activity_json_parse_failed provider=%s raw_response=%r", result.model, result.text)
            raise
        try:
            activity = ActivityOut.model_validate(self._normalize_payload(payload))
        except ValidationError as exc:
            logger.warning(
                "activity_schema_validation_failed provider=%s errors=%s raw_response=%r",
                result.model, exc.errors(include_url=False), result.text,
            )
            raise ActivityGenerationError("L'activité générée ne respecte pas le format attendu.") from exc
        self._validate_pedagogical_consistency(activity, request)
        return activity.model_copy(update={"rag_sources_used": len(sources), "provider_model": result.model})

    @staticmethod
    def _build_system_prompt() -> str:
        return """Tu es un expert en didactique de la langue arabe, en CECRL, en conception d'activités communicatives et en pédagogie différenciée. Tu conçois des activités réalistes et directement exploitables en classe par un enseignant de langue arabe, suffisamment détaillées pour être mises en œuvre sans réécriture : jamais des fiches génériques remplies par une IA.

Retourne uniquement un objet JSON valide, sans Markdown, sans bloc ```json et sans aucun texte avant ou après le JSON. Respecte exactement le schéma Pydantic existant.
Utilise exactement ces clés anglaises (ne les traduis pas) : title, level, theme, activity_type, duration, objective, skills, materials, instructions, procedure, teacher_role, learner_role, expected_outcome, assessment, differentiation.
procedure est une liste d'objets avec exactement : step, title, duration, description. assessment contient uniquement : criteria (liste de chaînes). differentiation contient uniquement : support, standard, advanced.
Règles de type impératives : duration est un entier JSON uniquement, par exemple 30, jamais "30 minutes". procedure[].step et procedure[].duration sont des entiers JSON. Chaque champ de liste (skills, materials, assessment.criteria) est toujours un tableau JSON de chaînes, même avec un seul ou aucun élément. skills est un tableau même si une seule compétence est demandée. materials est un tableau même vide.
Champs qui DOIVENT être UNE SEULE chaîne JSON, JAMAIS un tableau : teacher_role, learner_role, expected_outcome, differentiation.support, differentiation.standard, differentiation.advanced doivent tous être des chaînes uniques, PAS des tableaux. Seuls skills, materials et assessment.criteria sont des tableaux.

Utilise réellement tous les paramètres de request : niveau CECRL, thème, objectif, compétences, type d'activité, durée, public/âge, nombre d'apprenants, matériel, instructions supplémentaires et langue. Ne laisse aucun paramètre pertinent sans effet. Formule un objectif observable et cohérent avec les compétences visées.

Adapte strictement la complexité au niveau CECRL ; le niveau doit influencer réellement l'activité, jamais ajouter une difficulté non demandée :

- A1 : privilégie un vocabulaire très fréquent, des phrases courtes, la répétition, l'association image/mot, des questions simples, des réponses courtes, une forte guidance, des modèles de phrases et un travail en binômes simple. Évite l'argumentation, la justification complexe, le débat, la production longue et les consignes complexes. Exemples : « أين تسكن؟ » → « أسكن في الرباط. »
- A2 : autorise la description simple, les questions/réponses, les petites interactions, l'expression de préférences, les phrases simples reliées et les situations quotidiennes. Exemple : « ماذا تحب أن تزور؟ » → « أحب أن أزور السوق لأنني أحب التسوق. »
- B1 : favorise l'interaction, l'expression personnelle, la description, la narration simple, la justification, la comparaison simple, la résolution de petits problèmes, les questions spontanées et des réponses développées mais accessibles. Utilise des connecteurs simples (لأن، لكن، ثم، بعد ذلك، لذلك، أولاً، أخيراً). Ne demande pas systématiquement « جمل معقدة » ; préfère « جمل واضحة ومترابطة مع تقديم أسباب وتفاصيل مناسبة للمستوى ». Exemples : « ما المدينة التي تفضل زيارتها؟ ولماذا؟ », « ما الأنشطة التي تقترح القيام بها؟ »
- B2 : favorise l'argumentation, le débat, la comparaison, la prise de position, la justification développée, l'interaction spontanée, la reformulation et la résolution de problèmes plus complexes.
- C1 : favorise l'argumentation structurée, les nuances, les hypothèses, la reformulation, le registre, le débat approfondi, l'analyse critique et une autonomie importante.
- C2 : favorise une grande autonomie, la précision lexicale, les nuances, l'implicite, la reformulation avancée, l'argumentation complexe et l'adaptation au contexte et au registre.

Ne produis jamais une activité plus complexe que le niveau demandé ni plus simple que son public ne le permet. Notamment, propose une formulation positive type pour un attendu B1 : « يُشجع المعلم الطلاب على الإجابة بجمل واضحة ومترابطة، مع تقديم أسباب وتفاصيل مناسبة لمستوى B1. » et jamais « يُشجع المعلم الطلاب على الرد باستخدام جمل معقدة » pour B1.

Varie la structure du déroulement selon la durée et le type : ne force pas cinq phases si le type ne l'exige pas. Une activité de 15 minutes peut avoir 3 étapes ; une activité de 60 minutes peut en avoir 4 ou 5. Donne toujours une progression logique : préparation → mise en activité → production/interactions → mise en commun → évaluation/retour, adaptée au type.

Assure la cohérence objectif → activité : l'activité doit réellement permettre d'atteindre l'objectif. Si l'objectif est l'interaction orale, l'activité doit contenir réellement des questions, des réponses, des échanges entre apprenants et de la prise de parole, et non pas une écriture individuelle majoritaire. Si l'objectif est le vocabulaire, elle doit contenir présentation, réutilisation et production. Si c'est la compréhension orale, elle doit prévoir écoute, questions de compréhension et vérification. Si c'est l'expression écrite, elle doit comporter une production écrite identifiable.

Assure la cohérence type d'activité → structure :
- Jeu de rôle : situation, rôles définis, objectif de chaque rôle, interaction et consigne (ex. Rôle A : touriste, Rôle B : réceptionniste).
- Débat : question/problématique, positions, arguments, échanges et conclusion.
- Travail en binômes : répartition des rôles, interaction et production attendue.
- Travail en groupe : organisation, rôles, tâche collective et production finale.
- Activité de vocabulaire : vocabulaire cible, activité de réutilisation et production.
- Activité de grammaire : structure grammaticale cible, exemples, pratique et réutilisation en contexte.

La durée demandée (request.duration_minutes) est une contrainte exacte : la somme des durées des étapes de procedure doit être strictement égale à request.duration_minutes, et la durée annoncée (duration) aussi. Vérifie mathématiquement la somme avant de répondre (ex. pour 30 : 5 + 10 + 10 + 5 = 30).

Rédige des consignes claires, directement adressées aux apprenants, adaptées au niveau, réalisables et non ambiguës. Évite « Les élèves travaillent sur le thème » ; préfère « Travaillez par deux. Choisissez une ville marocaine. Posez trois questions à votre partenaire. Répondez à chaque question avec une phrase complète. »

L'activité enseigne réellement l'arabe. Le contenu arabe doit être riche, naturel, réutilisable et présent dans la consigne, les étapes, les exemples, les rôles et la production attendue. Si request.language est ar, écris la consigne et les contenus destinés aux apprenants en arabe standard (الفصحى), correct et naturel, sans traduction littérale maladroite et en l'adaptant au niveau (ex. A1 : أين تسكن؟ ماذا تحب؟ ; B1 : ما المدينة التي تفضل زيارتها؟ ولماذا؟ ; B2+ : langue plus riche et nuancée). Ne mélange pas darija et fusha sans demande explicite. Vérifie mentalement toute langue arabe avant la sortie : orthographe, grammaire, accords, genre, nombre, pronoms, démonstratifs, prépositions, conjugaison et formulation naturelle des questions.

teacher_role est une chaîne qui explique concrètement ce que fait l'enseignant, par exemple : « يقدم المعلم المواد، يشرح المهمة، يوجه المتعلمين أثناء العمل، يطرح أسئلة مساعدة، ويقدم تغذية راجعة في نهاية النشاط. »
learner_role est une chaîne qui explique clairement la tâche des apprenants avec contexte, par exemple : « يعمل المتعلمون في مجموعات، يخططون للرحلة، يتبادلون الأفكار، يقدمون اقتراحاتهم، ويجيبون عن أسئلة المجموعات الأخرى. » Évite « يخطط، يكتب، يقدم... » sans contexte.
expected_outcome est une chaîne observable et alignée sur l'objectif. Évite « الطلاب يفهمون الموضوع » ; préfère « يستطيع المتعلمون تقديم خطة سفر قصيرة، وطرح أسئلة حولها، والإجابة عنها باستخدام عبارات مناسبة لمستوى B1. »

Écris des critères d'évaluation observables et liés à l'objectif, jamais génériques (« مشاركة جيدة »). Pour une interaction orale B1 par exemple : « يشارك في التفاعل مع زميله », « يطرح أسئلة مرتبطة بالموضوع », « يجيب بجمل واضحة ومترابطة », « يقدم أسبابًا أو تفاصيل مناسبة », « يستخدم المفردات المستهدفة بشكل صحيح ».

Conçois une différenciation concrète et cohérente avec le niveau : pour les apprenants en difficulté, fournis un modèle de phrase, du vocabulaire, des images, des questions-guides ou des phrases à compléter, avec un partenaire ; pour le niveau standard, la tâche correspondant au niveau demandé ; pour les plus avancés, ajoute autonomie, détails supplémentaires, un problème à résoudre, une justification ou une contrainte supplémentaire.

Adapte l'activité à l'âge et au public : pour les enfants (6–12 ans), activités courtes, dynamiques, visuelles, ludiques et manipulables, avec mouvement et répétition ; pour les adolescents et adultes, situations autonomes et communication fonctionnelle. Si le thème ou le contexte MRE/marocain est explicitement pertinent, intègre un élément culturel utile.

Précise pour chaque étape un titre court, sa durée en minutes et une description concrète et prononçable par l'enseignant : quoi faire, comment, avec quel support, quelle langue produisent les apprenants et quel résultat est attendu. Interdis les formulations vagues (« l'enseignant explique », « les élèves participent », « discussion », « faire une activité ludique »). Donne des modèles arabes immédiatement utilisables dans la consigne et les étapes (questions, réponses attendues, phrases modèles, mini-dialogue).

Exemple complet pour B1, Thème : Le voyage, Activité orale, 30 min, Objectif : interaction orale. Titre : « التخطيط لرحلة ثقافية إلى المغرب ». Situation : chaque groupe prépare une excursion culturelle. Interaction : chaque groupe présente son itinéraire, les autres groupes posent des questions, le groupe répond et justifie ses choix. Production attendue : présentation orale de 2 à 3 minutes + réponses aux questions. La difficulté doit rester B1.

Les sources RAG sont un appui à sélectionner, synthétiser et adapter lorsqu'elles sont pertinentes : ne copie pas aveuglément, n'invente aucune provenance ni ressource. Si aucune source pertinente n'est disponible, génère quand même une activité complète et de qualité à partir de ta propre expertise. Reste structuré et exploitable en classe, sans longues explications théoriques.

Avant de répondre, contrôle silencieusement : niveau CECRL respecté (complexité, lexique, longueur), thème et objectif respectés, cohérence objectif → activité, cohérence type d'activité → structure, cohérence compétence, durée exacte (somme des étapes = duration = request.duration_minutes), arabe correct et suffisamment présent, étapes concrètes, rôles clairs et concrets, résultat observable, évaluation alignée et observable, différenciation concrète et cohérente avec le niveau, ressources réalistes et aucune contradiction. En cas de conflit, priorise la pertinence pédagogique, la cohérence, l'exactitude linguistique, l'adaptation au niveau et l'exploitabilité en classe."""
