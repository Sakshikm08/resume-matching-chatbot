from fastapi import FastAPI, Request, Depends, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List
import os
import tempfile
import pdfplumber
import re

from .db import SessionLocal, init_db
from .models import Resume
from .schemas import ResumeOut
from .embeddings import search_similar, build_index
from .utils_resume import add_resume

app = FastAPI(title="Talent Scout Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- DB dependency ----------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Startup ----------

@app.on_event("startup")
def startup_event():
    init_db()
    db = SessionLocal()
    try:
        dedupe_resumes(db)
    finally:
        db.close()
    build_index()


# ---------- Helper functions ----------

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


def compute_skill_match(resume: Resume, must_have_skills: List[str]) -> float:
    resume_skill_names = {s.name for s in resume.skills}
    must = set(must_have_skills)
    if not must:
        return 1.0
    overlap = must.intersection(resume_skill_names)
    return len(overlap) / len(must)


def build_explanation(resume: Resume, must_have_skills: List[str], sim_score: float, skill_match_score: float) -> str:
    resume_skill_names = {x.name for x in resume.skills}
    matched = [s for s in must_have_skills if s in resume_skill_names]
    missing = [s for s in must_have_skills if s not in matched]
    parts = []
    if matched:
        parts.append(f"Matches: {', '.join(matched)}")
    if missing:
        parts.append(f"Missing: {', '.join(missing)}")
    parts.append(f"Similarity {sim_score:.2f}, skill match {skill_match_score:.2f}")
    return " | ".join(parts)

from sqlalchemy import func

def dedupe_resumes(db: Session):
    """
    Keep one resume per identical raw_text, delete the others.
    """
    # Find min(id) for each raw_text
    subq = (
        db.query(func.min(Resume.id).label("keep_id"))
        .group_by(Resume.raw_text)
        .subquery()
    )

    # Delete all resumes whose id is not in the keep set
    deleted = (
        db.query(Resume)
        .filter(~Resume.id.in_(db.query(subq.c.keep_id)))
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted



def extract_resume_info(raw_text: str):
    """
    Robust extraction of candidate_name, title, years_experience, location, skills.
    """
    raw_text = raw_text.strip()
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    raw_lower = raw_text.lower()

    # ----- Name -----
    candidate_name = "Resume Candidate"
    for line in lines[:8]:
        if 2 <= len(line) <= 100 and any(c.isalpha() for c in line):
            words = line.split()
            if 1 <= len(words) <= 5:
                if not any(kw in line.lower() for kw in ["resume", "curriculum", "cv", "contact", "email", "phone"]):
                    candidate_name = line
                    break
    if candidate_name == "Resume Candidate" and lines:
        candidate_name = lines[0]

    # ----- Title -----
    title = ""
    title_keywords = [
        "engineer", "developer", "scientist", "analyst", "manager", "designer",
        "architect", "consultant", "specialist", "lead", "director", "coordinator",
        "administrator", "technician", "programmer", "qa", "tester",
    ]
    for line in lines[:20]:
        lower = line.lower()
        if any(w in lower for w in ["skills", "experience:", "education", "summary", "objective"]):
            continue
        if any(k in lower for k in title_keywords) and len(line) < 100:
            title = line
            break

    # ----- Years of experience -----
    years_experience = 0.0
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience",
        r"experience[:\s]+(\d+(?:\.\d+)?)\+?\s*years?",
        r"(\d+(?:\.\d+)?)\+?\s*yrs?\s+(?:of\s+)?(?:experience|exp)",
        r"(\d+(?:\.\d+)?)\s*years?",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw_lower)
        if m:
            try:
                val = float(m.group(1))
                if 0 < val <= 50:
                    years_experience = val
                    break
            except ValueError:
                pass

    # ----- Location -----
    location = ""
    loc_patterns = [
        r"location[:\s]+([^\n]+)",
        r"address[:\s]+([^\n]+)",
        r"based in[:\s]+([^\n]+)",
    ]
    for pattern in loc_patterns:
        m = re.search(pattern, raw_lower)
        if m:
            location = m.group(1).strip().title()
            break
    if not location:
        cities = [
            "bangalore", "bengaluru", "mumbai", "delhi", "ncr", "gurgaon", "gurugram",
            "noida", "pune", "hyderabad", "chennai", "kolkata", "ahmedabad", "jaipur",
            "kochi", "coimbatore", "chandigarh", "indore", "bhopal", "remote", "india",
        ]
        for city in cities:
            if city in raw_lower:
                idx = raw_lower.index(city)
                location = raw_text[idx:idx + len(city)]
                break

    # ----- Skills -----
    skills_list: List[str] = []
    all_skills = [
        "python", "java", "javascript", "typescript", "c++", "cpp", "c#", "go",
        "react", "reactjs", "angular", "vue", "html", "css", "flask", "django",
        "fastapi", "node", "nodejs", "express", "docker", "kubernetes", "aws",
        "azure", "gcp", "postgresql", "postgres", "mysql", "mongodb", "redis",
        "sql", "microservices", "rest", "api",
    ]
    for skill in all_skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", raw_lower):
            if skill not in skills_list:
                skills_list.append(skill)
    skills_list = skills_list[:25]

    return {
        "candidate_name": candidate_name,
        "title": title,
        "years_experience": years_experience,
        "location": location,
        "skills": skills_list,
    }


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": [], "query": "", "message": ""},
    )


@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    resume_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        raw_text = ""

        if resume_file.filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await resume_file.read()
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name
            with pdfplumber.open(tmp_path) as pdf:
                raw_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        elif resume_file.filename.endswith(".txt"):
            content = await resume_file.read()
            raw_text = content.decode("utf-8", errors="ignore")
        else:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "results": [],
                    "query": "",
                    "message": "❌ Unsupported file type. Use PDF or TXT.",
                },
            )

        if not raw_text.strip():
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "results": [],
                    "query": "",
                    "message": "❌ Could not extract text from file. File may be empty or corrupted.",
                },
            )

        if len(raw_text.strip()) < 30:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "results": [],
                    "query": "",
                    "message": "❌ Resume text too short. Please upload a complete resume (minimum 30 characters).",
                },
            )

        info = extract_resume_info(raw_text)

        if not info["candidate_name"] or info["candidate_name"] == "Resume Candidate":
            filename_name = (
                resume_file.filename.replace(".pdf", "").replace(".txt", "")
            )
            filename_name = filename_name.replace("_", " ").replace("-", " ").title()
            info["candidate_name"] = (
                filename_name if filename_name.strip() else "Resume Candidate"
            )

        add_resume(
            db=db,
            candidate_name=info["candidate_name"],
            raw_text=raw_text,
            skills=info["skills"],
            title=info["title"],
            years_experience=info["years_experience"],
            location=info["location"],
        )
        dedupe_resumes(db)  # <- remove duplicates
        
        build_index()

        success_parts = [f"✅ Resume uploaded: {info['candidate_name']}"]
        if info["title"]:
            success_parts.append(f"Title: {info['title']}")
        if info["years_experience"] > 0:
            success_parts.append(f"Experience: {info['years_experience']} years")
        if info["skills"]:
            success_parts.append(f"Skills: {len(info['skills'])} found")
        success_message = " | ".join(success_parts)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "results": [], "query": "", "message": success_message},
        )
        
       

    except Exception as e:
        import traceback

        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": [],
                "query": "",
                "message": f"❌ Upload failed: {str(e)}",
            },
        )


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    message: str = Form(...),
    top_k: int = Form(5),
    db: Session = Depends(get_db),
):
    try:
        req = simple_extract_requirements(message)
        must_have = req["must_have_skills"]
        min_years = req["min_years"]

        semantic_results = search_similar(message, top_k=top_k * 3)
        ids = [r_id for r_id, _ in semantic_results]
        id_to_sim = {r_id: score for r_id, score in semantic_results}

        resumes = db.query(Resume).filter(Resume.id.in_(ids)).all()
        out: List[ResumeOut] = []

        for r in resumes:
            if (
                min_years
                and r.years_experience is not None
                and r.years_experience < min_years
            ):
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
                    skills=[s.name for s in r.skills],  # List[str] for Pydantic
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
    except Exception as e:
        import traceback

        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": [],
                "query": message,
                "message": f"❌ Search failed: {str(e)}",
            },
        )


@app.get("/view_resumes", response_class=HTMLResponse)
def view_resumes(request: Request, db: Session = Depends(get_db)):
    resumes = db.query(Resume).all()
    return templates.TemplateResponse(
        "view_resumes.html", {"request": request, "resumes": resumes}
    )


@app.post("/delete_resume/{resume_id}")
def delete_resume(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if resume:
        db.delete(resume)
        db.commit()
        build_index()
        return {"success": True, "message": "Resume deleted"}
    return {"success": False, "message": "Resume not found"}


@app.get("/manage_resumes", response_class=HTMLResponse)
def manage_resumes(request: Request, db: Session = Depends(get_db)):
    resumes = db.query(Resume).all()
    return templates.TemplateResponse(
        "manage_resumes.html", {"request": request, "resumes": resumes}
    )
