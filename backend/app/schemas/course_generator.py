"""HTTP contracts for the teacher course (contenu pédagogique) generator.

A course answers: "what is the teacher going to teach and what content must the
learner acquire?" It is intentionally distinct from the lesson plan, which
answers "how does the teacher organise and run the session?", and from an
activity, which is a single communicative task.

The generated structure follows the pedagogical progression
Découvrir → Comprendre → Observer → Apprendre → Pratiquer → Réutiliser →
Produire → Bilan, adapted to the requested CEFR level (A1 to C2).
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CourseGenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "A1"
    theme: str = Field(min_length=1, max_length=255)
    objective: str | None = Field(default=None, min_length=1, max_length=1000)
    skills: list[str] = Field(default_factory=lambda: ["Expression orale"])
    duration_minutes: int = Field(default=60, ge=15, le=240)
    audience: str | None = Field(default=None, max_length=255)
    learner_count: int | None = Field(default=None, ge=1, le=100)
    language: Literal["ar", "fr", "en", "es"] = "ar"
    special_instructions: str | None = Field(default=None, max_length=1500)


class CourseExample(BaseModel):
    """A short worked example with a display title and the Arabic text."""

    title: str = "Exemple"
    body: str


class CourseSection(BaseModel):
    """A named explanatory block (a grammar point or a content block) with
    optional worked examples. Reused for both grammar and content so the LLM
    contract stays uniform."""

    title: str
    body: str
    examples: list[CourseExample] = Field(default_factory=list)


class CourseExercise(BaseModel):
    """One practice/production item: a task title, the learner briefing and an
    optional model answer."""

    title: str
    instructions: str = ""
    example: CourseExample | None = None


class CourseDialogue(BaseModel):
    """A model conversation: an optional context line then labelled turns."""

    context: str = ""
    lines: list[str] = Field(default_factory=list)


class CourseOut(BaseModel):
    """Complete course content. Fields are relaxed (defaults/optionals) so a
    section is never artificially mandatory: the pedagogical consistency
    validator enforces the level-appropriate requirements instead."""

    title: str
    level: str
    theme: str
    duration: int
    objectives: list[str] = Field(min_length=1)
    skills: list[str]
    vocabulary: list[str] = Field(min_length=1)
    expressions: list[str] = Field(default_factory=list)
    introduction: str = Field(min_length=1)
    grammar: list[CourseSection] = Field(default_factory=list)
    content: list[CourseSection] = Field(min_length=1)
    dialogue: CourseDialogue | None = None
    comprehension: list[CourseExercise] = Field(default_factory=list)
    guided_practice: list[CourseExercise] = Field(default_factory=list)
    communicative_practice: list[CourseExercise] = Field(default_factory=list)
    production: list[CourseExercise] = Field(default_factory=list)
    summary: list[str]
    homework: str | None = None
    rag_sources_used: int = 0
    provider_model: str | None = None