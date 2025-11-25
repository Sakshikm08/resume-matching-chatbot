from pydantic import BaseModel
from typing import List, Optional

class SkillOut(BaseModel):
    id: int
    name: str
    class Config:
        orm_mode = True

class ResumeOut(BaseModel):
    id: int
    candidate_name: str
    title: Optional[str]
    years_experience: Optional[float]
    location: Optional[str]
    skills: List[SkillOut]
    match_score: float
    skill_match_score: float
    explanation: str

    class Config:
        orm_mode = True
