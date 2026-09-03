"""HTTP contracts for the teacher activity generator."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ActivityGenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "A1"
    theme: str = Field(min_length=1, max_length=255)
    objective: str = Field(min_length=1, max_length=1000)
    skills: list[str] = Field(default_factory=lambda: ["Expression orale"])
    activity_type: str = Field(default="Autre", max_length=100)
    duration_minutes: int = Field(default=15, ge=5, le=120)
    audience: str | None = Field(default=None, max_length=255)
    learner_count: int | None = Field(default=None, ge=1, le=100)
    materials: list[str] = Field(default_factory=list)
    special_instructions: str | None = Field(default=None, max_length=1500)
    language: Literal["ar", "fr", "en", "es"] = "ar"


class ActivityProcedureStep(BaseModel):
    step: int = Field(ge=1)
    title: str
    duration: int = Field(ge=1, le=120)
    description: str


class ActivityAssessment(BaseModel):
    criteria: list[str] = Field(default_factory=list)


class ActivityDifferentiation(BaseModel):
    support: str
    standard: str
    advanced: str


class ActivityOut(BaseModel):
    title: str
    level: str
    theme: str
    activity_type: str
    duration: int
    objective: str
    skills: list[str]
    materials: list[str]
    instructions: str
    procedure: list[ActivityProcedureStep] = Field(min_length=1)
    teacher_role: str
    learner_role: str
    expected_outcome: str
    assessment: ActivityAssessment
    differentiation: ActivityDifferentiation
    rag_sources_used: int = 0
    provider_model: str | None = None
