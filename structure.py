from typing import List

from pydantic.v1 import BaseModel, Field


class SolvingConcept(BaseModel):
    concept: str = Field(description="A problem-solving concept name in traditional chinese to recognize.")
    description: str = Field(description="Concept description in traditional chinese.")
    sample_questions: List[str] = Field(
        description="Questions based on the presented problem-solving concept.")


class Process(BaseModel):
    concepts: List[SolvingConcept] = Field(description="A comprehensive problem-solving process.",
                                           default=[])


class AugmentedQuestions(SolvingConcept):
    questions: List[str] = Field(
        description="Augmented questions in traditional chinese related questions based on the presented "
                    "problem-solving concept.")
