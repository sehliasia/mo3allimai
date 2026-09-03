"""Prompt construction for the exercise generator.

Splits the LLM prompt into the standard sections requested: SYSTEM / CONTEXT /
PEDAGOGICAL PLAN / TASK / CONSTRAINTS. The CEFR rules and pedagogical guidance
live here (mirroring the previous system-prompt behaviour) and are combined with
the planner output and the RAG reference blocks.
"""

from __future__ import annotations

import json

from app.services.exercise_cefr import LEVEL_RULES
from app.services.exercise_planner import ExercisePlan


def _language_rules(language: str) -> str:
    """Explicit, per-field output-language mandate so a bilingual model cannot
    silently mix scripts (e.g. French consignes inside an Arabic request)."""
    norm = (language or "").casefold()
    if norm == "fr":
        return (
            "LANGUE DE RÉDACTION : la langue demandée est le FRANÇAIS. Chaque titre, "
            "consigne (prompt), contexte, option et paire DOIT être rédigé intégralement "
            "en français. N'écris JAMAIS d'exercice dans une autre langue et ne fais "
            "jamais de transcription.\n\n"
        )
    if norm == "en":
        return (
            "LANGUE DE RÉDACTION : la langue demandée est l'ANGLAIS. Chaque titre, "
            "consigne (prompt), contexte, option et paire DOIT être rédigé intégralement "
            "en anglais. N'écris JAMAIS d'exercice dans une autre langue et ne fais "
            "jamais de transcription.\n\n"
        )
    if norm == "es":
        return (
            "LANGUE DE RÉDACTION : la langue demandée est l'ESPAGNOL. Chaque titre, "
            "consigne (prompt), contexte, option et paire DOIT être rédigé intégralement "
            "en espagnol. N'écris JAMAIS d'exercice dans une autre langue et ne fais "
            "jamais de transcription.\n\n"
        )
    return (
        "LANGUE DE RÉDACTION : la langue demandée est l'ARABE (العربية الفصحى). Chaque "
        "titre, consigne (prompt), contexte, option et paire DOIT être rédigé intégralement "
        "en arabe, en caractères arabes. N'écris JAMAIS de consigne en français, en anglais "
        "ni en transcription latine.\n\n"
    )


def _arabic_quality_rules(language: str) -> str:
    """Arabic-only quality mandate: learner-facing content in Arabic, no
    French/Arabic mixing, no artificial translation of RAG words."""
    if (language or "").casefold() != "ar":
        return ""
    return (
        "QUALITÉ ARABE : tout le contenu pédagogique destiné à l'apprenant (titre, consigne, "
        "contexte, options, paires, correction) DOIT être en arabe (العربية الفصحى). Évite tout "
        "mélange français/arabe. Ne traduis PAS artificiellement les mots issus du contexte RAG et "
        "n'invente pas de nouvelle notion grammaticale non prévue par le plan.\n"
        "Pour le niveau A1, rédige des phrases courte et simples, uniquement avec le vocabulaire "
        "et la grammaire du plan.\n\n"
    )


def build_system_prompt(language: str = "ar") -> str:
    """The SYSTEM role: the model's identity and hard pedagogical rules."""
    rules_text = "\n".join(
        f"RÈGLE {level} : " + "; ".join(
            f"{label}: {value}" for label, value in LEVEL_RULES[level].items()
        )
        for level in ("A1", "A2", "B1", "B2", "C1", "C2")
    )
    return _language_rules(language) + (
        "Tu es un concepteur pédagogique expert en arabe (العربية الفصحى) et en CECRL. "
        "À partir du plan pédagogique fourni (PEDAGOGICAL PLAN) et du contexte de la base de "
        "connaissances (CONTEXT), tu CONÇOIS une fiche d'exercices simple, naturelle et "
        "réellement exploitable par un enseignant.\n"
        "Tu es un GÉNÉRATEUR, pas un extracteur : tu crées du contenu original, varié et "
        "non copié ; tu n'inventes jamais de vocabulaire ou de fait non fondé.\n"
        "RÔLE DU CONTEXTE (RAG) : les blocs de CONTEXT sont une SOURCE DE CONNAISSANCES — tu y "
        "puises des idées de thème, du vocabulaire et des situations, mais tu RÉDIGES toi-même un "
        "contenu arabe original et adapté. Ne recopie JAMAIS un bloc mot pour mot ni ne le "
        "reproduis tel quel ; ne fais pas de traduction mot à mot.\n"
        "SIMPLICITÉ : avec un niveau A1 ou A2, rédige un arabe simple et familier (phrases "
        "courtes, vocabulaire concret de la vie quotidienne). Le texte teacher doit rester "
        "naturel et cohérent au niveau.\n\n"
        "PROGRESSION PÉDAGOGIQUE : respecte strictement la planification fournie dans PEDAGOGICAL "
        "PLAN (objectifs, vocabulaire cible, grammaire cible, distribution des types ordonnée du "
        "plus guidé au plus libre). Ne mélange pas les exercices aléatoirement.\n\n"
        "VARIÉTÉ : respecte la distribution demandée et ne produis JAMAIS deux exercices identiques. "
        "Diversifie les consignes et les types.\n\n"
        "CORRECTIONS : chaque exercice DOIT avoir une answer_expectation claire, exacte et cohérente "
        "avec sa consigne (la correction ne recopie pas la consigne et correspond au contenu demandé).\n\n"
        "QUALITÉ DE L'ARABE : rédige en العربية الفصحى المعاصرة correcte (orthographe, grammaire, "
        "accord genre/nombre, pronoms, démonstratifs, prépositions, ponctuation). N'utilise JAMAIS de "
        "darija sauf demande explicite et n'invente jamais de mots arabes.\n\n"
        "CECRL (dimension pédagogique interne, pas une citation officielle du Conseil de l'Europe), "
        "applique strictement la règle du niveau demandé :\n" + rules_text + "\n\n"
        "PROVENANCE : si un exercice reprend réellement un bloc du contexte (même vocabulaire, même "
        "exercice remodelé), indique son source_index (0-based). S'il est entièrement nouveau, omet "
        "source_index. Ne prétends JAMAIS qu'un exercice vient d'un document s'il n'y figure pas."
    )


def build_task_section(
    request: dict, plan: ExercisePlan, target_distribution: list[str],
) -> str:
    """TASK + CONSTRAINTS: what to generate and the hard rules."""
    return (
        "TACHE : génère exactement "
        f"{request['count']} exercices, dans l'ordre de distribution suivant (respecte cet ordre) : "
        f"{', '.join(target_distribution)}\n"
        "Ce nombre est une CONTRAINTE STRICTE ; produis exactement les tokens de distribution fournis, "
        "ni plus ni moins.\n"
        "CONSTRAINTES :\n"
        f"- niveau : {plan.level}  |  langue : {request.get('language') or 'ar'}  |  difficulté : adaptée au niveau {plan.level}\n"
        "- thème : " + plan.theme + "\n"
        "- compétences : " + ", ".join(plan.skills) + "\n"
        "- objectifs : " + "; ".join(plan.learning_objectives) + "\n"
        "- vocabulaire : utilise PRIORITAIREMENT le vocabulaire cible fourni par le plan "
        "pédagogique (" + ", ".join(plan.target_vocabulary) + "). N'introduis PAS de vocabulaire "
        "spécialisé, abstrait ou hors thème qui ne correspond pas au niveau demandé. Les petits mots "
        "grammaticaux nécessaires à la construction de phrases naturelles restent autorisés.\n"
        "- grammaire : n'utilise que la grammaire cible du plan pédagogique ("
        + ", ".join(plan.target_grammar) + "). N'invente PAS de nouvelle notion grammaticale non "
        "prévue par le plan et ne la transforme pas en un autre outil grammatical.\n"
        "- évite les doublons et le vocabulaire inutilement complexe pour le niveau\n"
        "- ne remplace pas par d'autres types / ordres que la distribution demandée, sauf si la "
        "compétence sélectionnée rend la distribution non pertinente (auquel cas adapte-la et "
        "précise-le dans l'objectif)\n\n"
        "LANGUE : la langue de rédaction des exercices est celle demandée "
        f"({request.get('language') or 'ar'}) ; applique strictement la règle LANGUE DE RÉDACTION "
        "du SYSTEM-prompt.\n"
        + _language_rules(request.get("language") or "ar")
        + _arabic_quality_rules(request.get("language") or "ar")
        + "Retourne UNIQUEMENT un objet JSON valide, sans Markdown, sans bloc ```json et sans texte "
        "avant ou après. Utilise exactement ces clés : title, level, theme, exercise_type, exercises. "
        "exercises est une liste d'objets avec exactement les clés : source_index (optionnel), title, "
        "skill, exercise_type, prompt, context (optionnel), answer_expectation, level, options "
        "(uniquement pour les types qcm / true_false sous forme de liste de chaînes courtes), "
        "is_true (uniquement true_false), pairs (uniquement matching, liste d'objets {left,right}), "
        "difficulty (easy|medium|hard)."
    )


def build_context_section(reference_blocks: list[dict]) -> str:
    """CONTEXT: the reference material from the RAG (optional)."""
    if not reference_blocks:
        return "CONTEXT : aucune ressource de référence disponible."
    return (
        "CONTEXT (contexte pédagogique récupéré — à réutiliser en référence) :\n"
        + json.dumps(reference_blocks, ensure_ascii=False)
    )
