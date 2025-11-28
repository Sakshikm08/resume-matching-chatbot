from pydantic import BaseModel
from typing import List, Optional

class SkillOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class ResumeOut(BaseModel):
    id: int
    candidate_name: str
    title: Optional[str] = None
    years_experience: Optional[float] = None
    location: Optional[str] = None

    # IMPORTANT: make this a list of strings, not SkillOut
    skills: List[str] = []          # <-- CHANGE HERE

    match_score: float
    skill_match_score: float
    explanation: str

    class Config:
        from_attributes = True
