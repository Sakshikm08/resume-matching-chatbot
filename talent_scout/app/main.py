from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List

from .db import SessionLocal, init_db
from .models import Resume
from .schemas import ResumeOut
from .embeddings import search_similar, build_index
from .utils_resume import add_resume

app = FastAPI(title="Talent Scout Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
from fastapi import File, UploadFile, Form
import pdfplumber

@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    candidate_name: str = Form(...),
    title: str = Form(""),
    years_experience: float = Form(0),
    location: str = Form(""),
    skills: str = Form(...),
    resume_file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    # Parse file (if uploaded)
    raw_text = ""
    if resume_file:
        if resume_file.filename.endswith(".pdf"):
            with pdfplumber.open(await resume_file.read()) as pdf:
                raw_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif resume_file.filename.endswith(".txt"):
            raw_text = (await resume_file.read()).decode()
    if not raw_text:
        # As fallback, use inputted skills/info
        raw_text = f"{candidate_name}, {title}, {skills}"

    skills_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
    add_resume(
        db=db,
        candidate_name=candidate_name,
        raw_text=raw_text,
        skills=skills_list,
        title=title,
        years_experience=years_experience,
        location=location,
    )
    build_index()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": [], "query": "", "message": "Resume uploaded!"},
    )


@app.on_event("startup")
def startup_event():
    init_db()
    build_index()

def simple_extract_requirements(message: str):
    text = message.lower()
    must_have = []
    for kw in ["python", "flask", "fastapi", "react", "node", "aws", "docker"]:
        if kw in text:
            must_have.append(kw)
    min_years = 0
    for y in range(1, 16):
        if f"{y} year" in text or f"{y} years" in text:
            min_years = y
    return {"must_have_skills": must_have, "min_years": min_years}

def compute_skill_match(resume, must_have_skills):
    resume_skill_names = {s.name for s in resume.skills}
    must = set(must_have_skills)
    if not must:
        return 1.0
    overlap = must.intersection(resume_skill_names)
    return len(overlap) / len(must)

def build_explanation(resume, must_have_skills, sim_score, skill_match_score):
    matched = [s for s in must_have_skills if s in {x.name for x in resume.skills}]
    missing = [s for s in must_have_skills if s not in matched]
    parts = []
    if matched:
        parts.append(f"Matches: {', '.join(matched)}")
    if missing:
        parts.append(f"Missing: {', '.join(missing)}")
    parts.append(f"Similarity {sim_score:.2f}, skill match {skill_match_score:.2f}")
    return " | ".join(parts)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": [], "query": "", "message": ""},
    )

@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    message: str = Form(...),
    top_k: int = Form(5),
    db: Session = Depends(get_db),
):
    req = simple_extract_requirements(message)
    must_have = req["must_have_skills"]
    min_years = req["min_years"]

    semantic_results = search_similar(message, top_k=top_k * 3)
    ids = [r_id for r_id, _ in semantic_results]
    id_to_sim = {r_id: score for r_id, score in semantic_results}

    resumes = db.query(Resume).filter(Resume.id.in_(ids)).all()
    out: List[ResumeOut] = []
    for r in resumes:
        if min_years and r.years_experience is not None and r.years_experience < min_years:
            continue
        skill_match = compute_skill_match(r, must_have)
        sim_score = id_to_sim.get(r.id, 0.0)
        final_score = 0.6 * sim_score + 0.4 * skill_match
        explanation = build_explanation(r, must_have, sim_score, skill_match)

        out.append(
            ResumeOut(
                id=r.id,
                candidate_name=r.candidate_name,
                title=r.title,
                years_experience=r.years_experience,
                location=r.location,
                skills=list(r.skills),
                match_score=round(final_score * 100, 2),
                skill_match_score=round(skill_match * 100, 2),
                explanation=explanation,
            )
        )

    out_sorted = sorted(out, key=lambda x: x.match_score, reverse=True)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": out_sorted[:top_k],
            "query": message,
            "message": "Showing best matches",
        },
    )

# optional simple seed endpoint
@app.get("/seed")
def seed(db: Session = Depends(get_db)):
    add_resume(
        db,
        candidate_name="Alice Johnson",
        title="Senior Backend Engineer",
        years_experience=5,
        location="Remote",
        raw_text="Python backend engineer with Flask, FastAPI, Docker and AWS experience.",
        skills=["python", "flask", "fastapi", "docker", "aws"],
    )
    add_resume(
        db,
        candidate_name="Bob Smith",
        title="Frontend Developer",
        years_experience=3,
        location="Bangalore",
        raw_text="React and Node developer with some AWS knowledge.",
        skills=["react", "node", "aws"],
    )
    build_index()
    return {"status": "seeded"}
