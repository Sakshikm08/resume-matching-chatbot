from sqlalchemy.orm import Session
from .models import Resume, Skill

def get_or_create_skill(db: Session, name: str) -> Skill:
    name_norm = name.strip().lower()
    skill = db.query(Skill).filter(Skill.name == name_norm).first()
    if not skill:
        skill = Skill(name=name_norm)
        db.add(skill)
        db.commit()
        db.refresh(skill)
    return skill

def add_resume(
    db: Session,
    candidate_name: str,
    raw_text: str,
    skills: list[str],
    title: str | None = None,
    years_experience: float | None = None,
    location: str | None = None,
):
    resume = Resume(
        candidate_name=candidate_name,
        title=title,
        years_experience=years_experience,
        location=location,
        raw_text=raw_text,
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)

    for s in skills:
        skill_obj = get_or_create_skill(db, s)
        resume.skills.append(skill_obj)

    db.commit()
    return resume
