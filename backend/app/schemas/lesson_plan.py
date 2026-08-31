"""HTTP contracts for the teacher lesson-plan generator."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class LessonPlanGenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "A1"
    theme: str = Field(min_length=1, max_length=255)
    general_objective: str = Field(min_length=1, max_length=1000)
    duration_minutes: int = Field(default=60, ge=15, le=240)
    skills: list[str] = Field(default_factory=lambda: ["speaking"])
    audience: str = Field(default="Adolescents et adultes", max_length=255)
    learner_count: int | None = Field(default=None, ge=1, le=100)
    session_type: str = Field(default="Découverte et pratique", max_length=100)
    prerequisites: list[str] = Field(default_factory=list)
    linguistic_points: list[str] = Field(default_factory=list)
    special_instructions: str | None = Field(default=None, max_length=1500)
    language: Literal["ar", "fr", "en", "es"] = "fr"


class LessonPlanFlowStep(BaseModel):
    phase: str
    duration: int = Field(ge=1, le=120)
    objective: str
    teacher_role: str
    learner_activity: str
    instructions: str
    materials: list[str] = Field(default_factory=list)
    work_mode: str
    example: str
    expected_result: str


class LessonPlanAssessment(BaseModel):
    assessment_type: str
    moment: str
    criteria: list[str] = Field(default_factory=list)
    method: str
    activity: str
    instructions: str
    success_indicators: list[str] = Field(default_factory=list)
    rubric: list["LessonPlanRubricItem"] = Field(default_factory=list)


class LessonPlanRubricItem(BaseModel):
    criterion: str
    achieved: str
    to_reinforce: str


class LessonPlanExtension(BaseModel):
    homework: str | None = None
    follow_up: str | None = None


class LessonPlanDifferentiation(BaseModel):
    support: list[str] = Field(default_factory=list)
    extension: list[str] = Field(default_factory=list)


class LessonPlanOut(BaseModel):
    title: str
    level: str
    theme: str
    duration: int
    audience: str
    session_type: str
    skills: list[str]
    age_approximation: str | None = None
    communicative_objectives: list[str] = Field(default_factory=list)
    linguistic_objectives: list[str] = Field(default_factory=list)
    general_objective: str
    specific_objectives: list[str]
    prerequisites: list[str]
    linguistic_content: dict[str, list[str]]
    materials: list[str]
    lesson_flow: list[LessonPlanFlowStep] = Field(min_length=1)
    assessment: LessonPlanAssessment
    differentiation: LessonPlanDifferentiation
    extension: LessonPlanExtension
    rag_sources_used: int = 0
    provider_model: str | None = None
