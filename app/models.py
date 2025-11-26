from sqlalchemy import Column, Integer, String, Text, Float, Table, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

resume_skill = Table(
    "resume_skill",
    Base.metadata,
    Column("resume_id", Integer, ForeignKey("resumes.id"), primary_key=True),
    Column("skill_id", Integer, ForeignKey("skills.id"), primary_key=True),
)

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    candidate_name = Column(String(200), nullable=False)
    title = Column(String(200), nullable=True)
    years_experience = Column(Float, nullable=True)
    location = Column(String(200), nullable=True)
    raw_text = Column(Text, nullable=False)
    embedding_dim = Column(Integer, nullable=True)

    skills = relationship("Skill", secondary=resume_skill, back_populates="resumes")

class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)

    resumes = relationship("Resume", secondary=resume_skill, back_populates="skills")
