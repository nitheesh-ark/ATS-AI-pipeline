"""
matcher.py  (v2 — upgraded)
---------------------------
Section-by-section ATS scoring of resumes against HR requirements.

Install:
    pip install rapidfuzz sentence-transformers scikit-learn numpy

Usage:
    from matcher import ResumeMatcher
    matcher = ResumeMatcher(model)
    matcher.set_requirements(HR_REQUIREMENTS)
    results = matcher.match_all(resume_db)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    from rapidfuzz import fuzz, process as rfuzz_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import warnings
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False
    warnings.warn(
        "rapidfuzz not installed — falling back to difflib.\n"
        "For better fuzzy skill matching: pip install rapidfuzz",
        stacklevel=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SCORE SCALING
# ─────────────────────────────────────────────────────────────────────────────

SCORE_MIN = 0.38
SCORE_MAX = 0.92


def _rescale(sim: float) -> float:
    """Map raw cosine similarity → 0–100 readable percentage."""
    c = max(SCORE_MIN, min(SCORE_MAX, float(sim)))
    return (c - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY SKILL MATCHING
# ─────────────────────────────────────────────────────────────────────────────

FUZZY_THRESHOLD = 82


def _fuzzy_skill_match(required_skill: str, candidate_skill: str) -> bool:
    a = required_skill.lower().strip()
    b = candidate_skill.lower().strip()

    if a == b:
        return True
    if a in b or b in a:
        return True

    if RAPIDFUZZ_AVAILABLE:
        if fuzz.token_set_ratio(a, b) >= FUZZY_THRESHOLD:
            return True
        if fuzz.partial_ratio(a, b) >= 90:
            return True
        if fuzz.WRatio(a, b) >= FUZZY_THRESHOLD:
            return True
        return False

    return SequenceMatcher(None, a, b).ratio() >= (FUZZY_THRESHOLD / 100)


def _skill_found(skill: str, candidate_skills_lower: list, raw_text: str) -> bool:
    skill_l = skill.lower().strip()

    if skill_l in raw_text:
        return True

    if RAPIDFUZZ_AVAILABLE:
        match = rfuzz_process.extractOne(
            skill_l,
            candidate_skills_lower,
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        return match is not None

    return any(_fuzzy_skill_match(skill_l, cs) for cs in candidate_skills_lower)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_skills(hr_req: dict, parsed: dict) -> tuple:
    required  = hr_req.get("required_skills", [])
    preferred = hr_req.get("preferred_skills", [])
    cand_lower = [s.lower() for s in parsed.get("skills", [])]
    raw_text   = " ".join(parsed.get("raw_sections", {}).values()).lower()

    req_matched  = [s for s in required  if _skill_found(s, cand_lower, raw_text)]
    pref_matched = [s for s in preferred if _skill_found(s, cand_lower, raw_text)]

    req_score  = (len(req_matched)  / len(required))  * 100 if required  else 100.0
    pref_score = (len(pref_matched) / len(preferred)) * 100 if preferred else 0.0

    extra = len(parsed.get("skills", [])) - len(req_matched) - len(pref_matched)
    bonus = min(5.0, extra * 0.5)

    combined = (req_score * 0.75) + (pref_score * 0.20) + bonus
    return (
        round(min(100.0, combined), 1),
        len(req_matched), len(required),
        len(pref_matched), len(preferred),
    )


def _score_experience(hr_req: dict, parsed: dict, hr_emb: dict, resume_emb: dict) -> tuple:
    is_fresher  = parsed.get("is_fresher", False)
    freshers_ok = hr_req.get("experience_required", {}).get("freshers_allowed", True)

    if is_fresher and not freshers_ok:
        return 0.0, "Fresher — not accepted for this role"
    if is_fresher and freshers_ok:
        return 45.0, "Fresher (fresher-friendly role)"

    all_req = (
        [s.lower() for s in hr_req.get("required_skills", [])]
        + [s.lower() for s in hr_req.get("preferred_skills", [])]
    )
    raw      = parsed.get("raw_sections", {})
    exp_lower = (raw.get("experience", "") + " " + raw.get("other", "")).lower()

    skills_in_exp = sum(1 for s in all_req if s in exp_lower)
    keyword_score = (skills_in_exp / len(all_req) * 100) if all_req else 50.0

    sem_score = 50.0
    if "experience" in hr_emb and "experience" in resume_emb:
        sim = cosine_similarity(
            hr_emb["experience"].reshape(1, -1),
            resume_emb["experience"].reshape(1, -1),
        )[0][0]
        sem_score = _rescale(sim)

    final = (keyword_score * 0.5) + (sem_score * 0.5)
    return round(final, 1), f"{len(parsed.get('experience',[]))} job(s) | {skills_in_exp}/{len(all_req)} skills in exp"


def _score_projects(hr_req: dict, parsed: dict, hr_emb: dict, resume_emb: dict) -> tuple:
    min_count  = hr_req.get("projects_required", {}).get("min_count", 0)
    cand_projs = parsed.get("projects", [])
    raw        = parsed.get("raw_sections", {})

    verbs = ["built", "created", "developed", "designed", "implemented",
             "launched", "spearheaded", "led", "delivered", "architected",
             "migrated", "deployed", "integrated", "automated"]
    combined = (raw.get("experience","") + " " + raw.get("projects","") + " " + raw.get("other","")).lower()
    evidence = sum(1 for v in verbs if v in combined)
    has_ev   = evidence >= 3

    if not cand_projs and not has_ev:
        return 0.0, "No projects found"

    effective   = len(cand_projs) if cand_projs else min(3, evidence // 2)
    count_score = min(100.0, (effective / min_count) * 100) if min_count > 0 else (80.0 if has_ev else 100.0)

    sem_score = 50.0
    if "projects" in hr_emb:
        tgt = resume_emb.get("projects", resume_emb.get("experience"))
        if tgt is not None:
            sim = cosine_similarity(hr_emb["projects"].reshape(1,-1), tgt.reshape(1,-1))[0][0]
            sem_score = _rescale(sim)

    final = (count_score * 0.4) + (sem_score * 0.6)
    src   = "dedicated section" if cand_projs else "inferred from experience"
    return round(final, 1), f"{effective} project(s) ({src})"


def _score_education(hr_req: dict, parsed: dict) -> tuple:
    edu_req  = hr_req.get("education_required", {})
    cand_edu = parsed.get("education", {})
    gpa      = parsed.get("gpa")
    score    = 70.0
    notes: list = []

    min_gpa = edu_req.get("min_gpa", 0)
    if gpa and min_gpa > 0:
        try:
            gpa_val = float(gpa)
            score   = min(100.0, score + (20 if gpa_val >= min_gpa else -20))
            notes.append(f"GPA {gpa_val}")
        except ValueError:
            pass
    elif not gpa:
        notes.append("GPA not found")

    req_field = edu_req.get("field", "").lower()
    deg_text  = (cand_edu.get("degree","") + " " + cand_edu.get("field","")).lower()
    if req_field and any(w in deg_text for w in req_field.split()):
        score = min(100.0, score + 10)
        notes.append("Field matched ✅")

    if cand_edu.get("institution"):
        score = min(100.0, score + 5)
        notes.append(cand_edu["institution"][:30])

    return round(score, 1), " | ".join(notes) if notes else "Education found"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "skills":     0.37,
    "experience": 0.15,
    "projects":   0.20,
    "education":  0.4,
    "summary":    0.30,
}


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ResumeMatcher:
    """
    Matches resume_db against HR requirements.

    Example:
        from matcher import ResumeMatcher
        matcher = ResumeMatcher(parser.model)
        matcher.set_requirements(HR_REQUIREMENTS)
        results = matcher.match_all(resume_db)
    """

    def __init__(self, model: SentenceTransformer, weights: dict = None):
        self.model   = model
        self.weights = weights or DEFAULT_WEIGHTS
        self.hr_req: dict = {}
        self.hr_emb: dict = {}
        if RAPIDFUZZ_AVAILABLE:
            print("✅ rapidfuzz fuzzy skill matching active")
        else:
            print("⚠️  rapidfuzz not installed — using difflib fallback (pip install rapidfuzz)")

    def set_requirements(self, hr_req: dict) -> None:
        self.hr_req = hr_req
        self.hr_emb = self._embed_requirements(hr_req)
        print(f"✅ Requirements set: {hr_req.get('job_role','Unknown')}  |  sections: {list(self.hr_emb.keys())}")

    def _embed_requirements(self, hr_req: dict) -> dict:
        emb: dict = {}
        all_skills = hr_req.get("required_skills", []) + hr_req.get("preferred_skills", [])
        if all_skills:
            emb["skills"] = self.model.encode(
                "Required skills: " + ", ".join(all_skills), normalize_embeddings=True
            )
        exp_desc = hr_req.get("experience_required", {}).get("description", "")
        if exp_desc:
            emb["experience"] = self.model.encode(exp_desc, normalize_embeddings=True)
        proj_desc = hr_req.get("projects_required", {}).get("description", "")
        if proj_desc:
            emb["projects"] = self.model.encode(proj_desc, normalize_embeddings=True)
        role = hr_req.get("role_description", "")
        if role.strip():
            emb["summary"] = self.model.encode(role.strip(), normalize_embeddings=True)
        return emb

    def match(self, filename: str, resume_data: dict) -> dict:
        """Score one resume against HR requirements."""
        if not self.hr_req:
            raise RuntimeError("Call set_requirements() before match().")

        parsed     = resume_data["parsed"]
        resume_emb = resume_data["embeddings"]

        skill_score, req_m, req_t, pref_m, pref_t = _score_skills(self.hr_req, parsed)
        exp_score,   exp_note   = _score_experience(self.hr_req, parsed, self.hr_emb, resume_emb)
        proj_score,  proj_note  = _score_projects(self.hr_req, parsed, self.hr_emb, resume_emb)
        edu_score,   edu_note   = _score_education(self.hr_req, parsed)

        sum_score = 50.0
        if "summary" in self.hr_emb and "summary" in resume_emb:
            sim = cosine_similarity(
                self.hr_emb["summary"].reshape(1, -1),
                resume_emb["summary"].reshape(1, -1),
            )[0][0]
            sum_score = _rescale(sim)

        section_scores = {
            "skills":     skill_score,
            "experience": exp_score,
            "projects":   proj_score,
            "education":  edu_score,
            "summary":    round(sum_score, 1),
        }
        total = sum(section_scores[s] * self.weights.get(s, 0) for s in section_scores)

        return {
            "name":           parsed["candidate_name"],
            "filename":       filename,
            "total_score":    round(total, 1),
            "section_scores": section_scores,
            "details": {
                "skills":     f"{req_m}/{req_t} required, {pref_m}/{pref_t} preferred",
                "experience": exp_note,
                "projects":   proj_note,
                "education":  edu_note,
            },
            "parsed": parsed,
        }

    def match_all(self, resume_db: dict) -> list:
        """Score all resumes and return sorted list (highest score first)."""
        results = [self.match(fname, data) for fname, data in resume_db.items()]
        results.sort(key=lambda x: x["total_score"], reverse=True)
        return results

    @staticmethod
    def print_summary(results: list) -> None:
        print("\n📋 ATS Match Summary")
        print("-" * 65)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. {r['name']:<28} {r['total_score']:>5.1f}%"
                f"  | Skills: {r['details']['skills']}"
            )
