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

import re

def extract_resume_info(raw_text: str):
    """
    Extract title, experience, location, and skills from resume text.
    Robust to varied formatting and works for plain text resumes like Alice Johnson's.
    """
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    raw_lower = raw_text.lower()

    # --- Name ---
    candidate_name = "Resume Candidate"
    for line in lines[:6]:
        if any(c.isalpha() for c in line) and len(line.split()) in (2, 3, 4):
            candidate_name = line
            break
    # Fallback: use filename
    if candidate_name in ['resume', 'curriculum vitae', 'cv', 'profile']:
        candidate_name = lines[1] if len(lines) > 1 else "Resume Candidate"

    # --- Title ---
    title = ""
    for line in lines[:10]:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['engineer', 'developer', 'scientist', 'analyst',
                                           'manager', 'designer', 'architect', 'lead', 'specialist']):
            if 'experience' not in line_lower and 'skill' not in line_lower and len(line) < 60:
                title = line
                break
    # Fallback: find "Title:" label
    if not title:
        match = re.search(r'title:\s*(.*)', raw_text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()

    # --- Experience ---
        # --- Experience ---
    years_experience = 0

    # Robust patterns for experience extraction
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?[^\n]*experience',            # "5 years experience"
        r'experience[^\d]*(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?',            # "Experience: 5 years"
        r'over\s+(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?',                     # "over 5 years"
        r'(\d+)\s*years?[^.]*at',                                         # "5 years at Company"
        r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*yrs?[^\n]*exp',                     # "5 yrs exp"
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            years_experience = float(match.group(1))
            break

    # Fallback: Scan description lines for year-like expressions
    if years_experience == 0:
        for line in lines[:15]:
            if 'year' in line.lower():
                nums = re.findall(r'\d+(?:\.\d+)?', line)
                for num in nums:
                    try:
                        num_float = float(num)
                        if 0 < num_float < 50:  # reasonable range
                            years_experience = num_float
                            break
                    except:
                        continue


    # --- Location ---
    location = ""
    match = re.search(r'Location:\s*([^\n]+)', raw_text, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
    else:
        # Try searching for city/remote phrases
        for city in ['bangalore', 'mumbai', 'delhi', 'pune', 'hyderabad', 'chennai', 'kolkata', 'remote', 'india']:
            if city in raw_lower:
                location = city.title()
                break

    # --- Skills ---
    skill_keywords = [
        'python', 'flask', 'fastapi', 'docker', 'aws', 'postgresql', 'api',
        'microservices', 'rest', 'sql', 'react', 'node', 'mongodb', 'html', 'css',
        'tailwind', 'bootstrap', 'svelte', 'typescript', 'angular', 'vue'
    ]
    skills_list = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', raw_lower):
            skills_list.append(skill)

    # Remove duplicates and sort
    skills_list = sorted(list(set(skills_list)), key=lambda x: raw_lower.index(x) if x in raw_lower else 9999)

    # Final result
    return {
        'candidate_name': candidate_name,
        'title': title,
        'years_experience': years_experience,
        'location': location,
        'skills': skills_list
    }


from fastapi import Request, File, UploadFile, HTTPException

@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    resume_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        raw_text = ""
        # Read file based on type
        if resume_file.filename.endswith(".pdf"):
            import tempfile, pdfplumber, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await resume_file.read()
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name
            with pdfplumber.open(tmp_path) as pdf:
                raw_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        elif resume_file.filename.endswith(".txt"):
            content = await resume_file.read()
            raw_text = content.decode("utf-8", errors="ignore")
        else:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "results": [], "query": "", "message": "❌ Unsupported file type. Use PDF or TXT."},
            )

        if not raw_text.strip():
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "results": [], "query": "", "message": "❌ Could not extract text from file. File may be empty or corrupted."},
            )

        if len(raw_text.strip()) < 30:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "results": [], "query": "", "message": "❌ Resume text too short. Please upload a complete resume (minimum 30 characters)."},
            )

        info = extract_resume_info(raw_text)

        if not info['candidate_name'] or info['candidate_name'] == "Resume Candidate":
            filename_name = resume_file.filename.replace('.pdf', '').replace('.txt', '')
            filename_name = filename_name.replace('_', ' ').replace('-', ' ').title()
            info['candidate_name'] = filename_name if filename_name.strip() else "Resume Candidate"

        add_resume(
            db=db,
            candidate_name=info['candidate_name'],
            raw_text=raw_text,
            skills=info['skills'] if info['skills'] else [],
            title=info['title'] if info['title'] else "",
            years_experience=info['years_experience'] if info['years_experience'] else 0,
            location=info['location'] if info['location'] else "",
        )
        build_index()

        success_parts = [f"✅ Resume uploaded: {info['candidate_name']}"]
        if info['title']:
            success_parts.append(f"Title: {info['title']}")
        if info['years_experience'] > 0:
            success_parts.append(f"Experience: {info['years_experience']} years")
        if info['skills']:
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
            {"request": request, "results": [], "query": "", "message": f"❌ Upload failed: {str(e)}"},
        )

    
@app.get("/view_resumes", response_class=HTMLResponse)
def view_resumes(request: Request, db: Session = Depends(get_db)):
    resumes = db.query(Resume).all()
    return templates.TemplateResponse(
        "view_resumes.html",
        {"request": request, "resumes": resumes}
    )


@app.post("/delete_resume/{resume_id}")
def delete_resume(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if resume:
        db.delete(resume)
        db.commit()
        build_index()  # Rebuild FAISS index after deletion
        return {"success": True, "message": "Resume deleted"}
    return {"success": False, "message": "Resume not found"}

@app.get("/manage_resumes", response_class=HTMLResponse)
def manage_resumes(request: Request, db: Session = Depends(get_db)):
    resumes = db.query(Resume).all()
    return templates.TemplateResponse(
        "manage_resumes.html",
        {"request": request, "resumes": resumes}
    )




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
                skills=[s.name for s in r.skills],  # <--- here!
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

