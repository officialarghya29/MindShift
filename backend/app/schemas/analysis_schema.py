"""Pydantic response schemas (PS-01 §31 schemas/analysis_schema.py)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionProbs(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class BinarySignal(BaseModel):
    probability: float
    confidence: float | None = None
    prediction: int | None = None
    supporting_signals: list[str] = Field(default_factory=list)


class MessageAnalysis(BaseModel):
    message_id: int
    speaker_id: str
    text: str
    sentiment: PredictionProbs
    emotion: PredictionProbs
    tone: PredictionProbs
    tension: float
    sarcasm: BinarySignal
    irony: BinarySignal
    passive_aggression: BinarySignal
    confidence: float | None = None


class TurningPoint(BaseModel):
    message_id: int
    before: dict
    after: dict
    tension_change: float
    trigger_text: str
    speaker: str


class ConversationReport(BaseModel):
    conversation_id: str
    n_messages: int
    summary: dict
    messages: list[MessageAnalysis]
    emotional_arc: dict
    emotion_transitions: list[dict]
    turning_points: list[TurningPoint]
    escalation: dict
    speaker_profiles: dict
    explanations: list[dict]
    disclaimer: str
