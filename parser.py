"""
parser.py  (v3 — Tesseract confirmed)
--------------------------------------
PDF → structured JSON + sentence-transformer embeddings.

Libraries:
  ✅ pdfminer.six   — better text extraction for complex/multi-column PDFs
  ✅ pdfplumber     — word-position spatial layout extraction
  ✅ PyMuPDF (fitz) — high-quality 300 DPI page rendering for OCR
  ✅ pytesseract    — OCR for scanned/image PDFs (Tesseract confirmed installed)
  ✅ Pillow         — image preprocessing before OCR (deskew, threshold, contrast)
  ✅ spaCy          — NER (PERSON/ORG/GPE) + POS tagging for skill cleaning
  ✅ dateparser     — robust date parsing (all formats, PREFER_DATES_FROM: past)
  ✅ re             — structured patterns (email, phone, GPA)

Install:
    pip install pdfplumber pdfminer.six PyMuPDF pytesseract pillow \
                spacy dateparser sentence-transformers torch numpy scikit-learn
    python -m spacy download en_core_web_md

    # Tesseract binary:
    # Ubuntu:  sudo apt install tesseract-ocr
    # Mac:     brew install tesseract
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import io
import warnings
import datetime
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageFilter, ImageOps
from sentence_transformers import SentenceTransformer

# ── pdfminer ──────────────────────────────────────────────────────────────────
from pdfminer.high_level import extract_text as pdfminer_extract

# ── PyMuPDF — required for high-quality OCR page rendering ───────────────────
import fitz  # PyMuPDF — pip install PyMuPDF

# ── pytesseract — confirmed installed ────────────────────────────────────────
import pytesseract

# Tesseract config: LSTM engine (oem 3) + uniform block of text (psm 6)
TESS_CONFIG = "--oem 3 --psm 6"

# ── spaCy ─────────────────────────────────────────────────────────────────────
try:
    import spacy
    for _model_name in ("en_core_web_md", "en_core_web_sm"):
        try:
            _nlp = spacy.load(_model_name)
            SPACY_MODEL = _model_name
            SPACY_AVAILABLE = True
            break
        except OSError:
            continue
    else:
        raise OSError("No spaCy model found")
except Exception:
    _nlp = None
    SPACY_MODEL = None
    SPACY_AVAILABLE = False
    warnings.warn(
        "spaCy model not found. Run:\n"
        "  python -m spacy download en_core_web_md\n"
        "Falling back to regex-based extraction.",
        stacklevel=2,
    )

# ── dateparser ────────────────────────────────────────────────────────────────
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    warnings.warn("dateparser not installed — pip install dateparser", stacklevel=2)

_DATEPARSER_SETTINGS = {
    "PREFER_DAY_OF_MONTH": "first",
    "PREFER_DATES_FROM":   "past",
    "RETURN_AS_TIMEZONE_AWARE": False,
    "STRICT_PARSING": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PROFICIENCY_WORDS = {
    "expert", "advanced", "intermediate", "beginner", "native", "fluent",
    "proficient", "familiar", "basic", "senior", "junior", "experienced",
    "skilled", "certified", "working knowledge",
}

NOISE_SKILLS = {
    "communication", "teamwork", "leadership", "time management",
    "problem solving", "attention to detail", "critical thinking",
    "interpersonal", "multitasking", "adaptability",
}

SECTION_HEADERS = {
    "summary":      ["summary", "objective", "profile", "about", "overview",
                     "career objective", "professional summary"],
    "skills":       ["skill", "technologies", "tools", "languages", "frameworks",
                     "competencies", "technical skills", "expertise", "stack",
                     "proficiencies"],
    "experience":   ["experience", "work history", "employment", "work experience",
                     "professional experience", "career history"],
    "projects":     ["projects", "portfolio", "personal projects", "academic projects",
                     "key projects"],
    "education":    ["education", "academic", "qualifications", "degrees",
                     "educational background"],
    "certificates": ["certificate", "certification", "licenses", "credentials",
                     "achievements"],
}

KNOWN_TECH = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "React", "Vue", "Angular", "Node.js", "Django", "Flask", "FastAPI",
    "PyTorch", "TensorFlow", "Keras", "NumPy", "pandas", "scikit-learn",
    "OpenCV", "YOLO", "BERT", "GPT", "Hugging Face", "CUDA",
    "CNN", "RNN", "LSTM", "Transformer",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Git",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Firebase",
    "HTML", "CSS", "SQL", "REST", "GraphQL",
    "MLflow", "ONNX", "Spark", "Kafka", "Spring Boot",
    "Flutter", "Swift", "Kotlin", "Android", "iOS",
]


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION  (3-stage pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _pdfplumber_text(filepath: str) -> str:
    lines = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            try:
                words = page.extract_words()
                if words:
                    rows: dict = {}
                    for w in words:
                        y = round(w["top"] / 8) * 8
                        rows.setdefault(y, []).append(w)
                    for y in sorted(rows):
                        lines.append(" ".join(
                            w["text"] for w in sorted(rows[y], key=lambda x: x["x0"])
                        ))
                else:
                    t = page.extract_text()
                    if t:
                        lines.extend(t.split("\n"))
            except Exception:
                t = page.extract_text()
                if t:
                    lines.extend(t.split("\n"))
    return "\n".join(lines)


def _pdfminer_text(filepath: str) -> str:
    try:
        return pdfminer_extract(filepath) or ""
    except Exception:
        return ""


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    min_width = 1800
    if img.width < min_width:
        scale = min_width / img.width
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 0 if p < 150 else 255, "1")
    return img


def _tesseract_ocr(filepath: str) -> str:
    pages: list = []
    try:
        doc = fitz.open(filepath)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = _preprocess_for_ocr(img)
            text = pytesseract.image_to_string(img, lang="eng", config=TESS_CONFIG)
            pages.append(text)
        doc.close()
        if pages:
            return "\n".join(pages)
    except Exception as e:
        print(f"  ⚠️  PyMuPDF OCR failed ({e}) — trying pdfplumber fallback...")

    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=300).original
                img = _preprocess_for_ocr(img)
                text = pytesseract.image_to_string(img, lang="eng", config=TESS_CONFIG)
                pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def extract_raw_text(filepath: str) -> str:
    """3-stage extraction: pdfplumber → pdfminer → Tesseract OCR."""
    text1 = _pdfplumber_text(filepath)
    if len(text1.strip()) >= 150:
        return text1

    text2 = _pdfminer_text(filepath)
    if len(text2.strip()) >= 150:
        return text2

    best = text1 if len(text1) >= len(text2) else text2

    if len(best.strip()) < 100:
        print(f"  ⚠️  Low text yield ({len(best)} chars) — running Tesseract OCR...")
        ocr = _tesseract_ocr(filepath)
        if len(ocr.strip()) > len(best.strip()):
            print("  ✅ OCR succeeded.")
            return ocr
        print("  ⚠️  OCR also returned low text — PDF may be corrupt or heavily encrypted.")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# SECTION SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def detect_section(line: str) -> str | None:
    clean = line.strip().lower().rstrip(":").rstrip()
    for section, keywords in SECTION_HEADERS.items():
        for kw in keywords:
            if clean == kw or clean.startswith(kw):
                return section
    return None


def _is_header_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 55:
        return False
    if s.upper() == s and 2 < len(s) and len(s.split()) <= 5:
        return True
    if s.endswith(":") and len(s.split()) <= 5:
        return True
    return bool(detect_section(s))


def split_into_sections(raw_text: str) -> dict:
    sections = {k: [] for k in SECTION_HEADERS}
    sections["other"] = []
    current = "other"
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_header_line(stripped):
            detected = detect_section(stripped)
            if detected:
                current = detected
                continue
        sections[current].append(stripped)
    return {k: "\n".join(v) for k, v in sections.items()}


# ─────────────────────────────────────────────────────────────────────────────
# spaCy-POWERED FIELD EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def extract_name(raw_text: str) -> str:
    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(raw_text[:600])
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 5:
                if not detect_section(ent.text):
                    return ent.text.title()

    for line in [l.strip() for l in raw_text.split("\n") if l.strip()][:6]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(
            w.replace("-", "").replace(".", "").isalpha() for w in words
        ):
            if not detect_section(line):
                return line.title()
    return "Unknown Candidate"


def extract_orgs(text: str) -> list:
    if not SPACY_AVAILABLE or not _nlp:
        return []
    doc = _nlp(text[:3000])
    return list(dict.fromkeys(
        ent.text.strip() for ent in doc.ents
        if ent.label_ in ("ORG", "FAC") and len(ent.text.strip()) > 2
    ))


def extract_contact(raw_text: str) -> dict:
    contact: dict = {}

    m = re.search(r"[\w.+-]+@[\w-]+\.\w+", raw_text)
    if m:
        contact["email"] = m.group()

    m = re.search(r"[\+\(]?\d[\d\s\-\(\)]{7,15}\d", raw_text)
    if m:
        contact["phone"] = m.group().strip()

    m = re.search(r"linkedin\.com/in/[\w\-]+", raw_text, re.I)
    if m:
        contact["linkedin"] = m.group()

    m = re.search(r"github\.com/[\w\-]+", raw_text, re.I)
    if m:
        contact["github"] = m.group()

    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(raw_text[:800])
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC") and "location" not in contact:
                contact["location"] = ent.text
    else:
        m = re.search(r"[A-Z][a-z]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+)", raw_text)
        if m:
            contact["location"] = m.group()

    return contact


def _clean_skill(raw: str) -> str | None:
    if SPACY_AVAILABLE and _nlp and len(raw.split()) > 1:
        doc = _nlp(raw.strip())
        tokens = list(doc)
        while tokens and (
            tokens[-1].pos_ in ("ADJ", "ADV")
            or tokens[-1].text.lower() in PROFICIENCY_WORDS
        ):
            tokens.pop()
        cleaned = " ".join(t.text for t in tokens).strip()
        return cleaned if len(cleaned) > 1 else None

    words = raw.strip().split()
    while words and words[-1].lower() in PROFICIENCY_WORDS:
        words.pop()
    cleaned = " ".join(words).strip()
    return cleaned if len(cleaned) > 1 else None


# ─────────────────────────────────────────────────────────────────────────────
# DATE PARSING
# ─────────────────────────────────────────────────────────────────────────────

_PRESENT_WORDS = {"present", "current", "now", "till date", "ongoing", "today"}

def _parse_date(s: str) -> datetime.datetime | None:
    if not s:
        return None
    if s.strip().lower() in _PRESENT_WORDS:
        return datetime.datetime.now()

    if DATEPARSER_AVAILABLE:
        result = dateparser.parse(s, settings=_DATEPARSER_SETTINGS)
        if result:
            return result

    yr = re.search(r"\b(19|20)\d{2}\b", s)
    if yr:
        return datetime.datetime(int(yr.group()), 1, 1)
    return None


_DATE_RANGE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|\d{1,2}/\d{4}|\d{4})"
    r"\s*(?:–|-|to|—)\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|\d{1,2}/\d{4}|\d{4}|present|current|now|till date|ongoing)",
    re.IGNORECASE,
)


def get_years_of_experience(exp_text: str) -> float:
    total_months = 0
    for m in _DATE_RANGE_RE.finditer(exp_text):
        start = _parse_date(m.group(1))
        end   = _parse_date(m.group(2))
        if start and end and end >= start:
            total_months += max(0, (end.year - start.year) * 12 + (end.month - start.month))

    if total_months > 0:
        return round(total_months / 12, 1)

    years = re.findall(r"\b(19|20)(\d{2})\b", exp_text)
    year_ints = [int(f"{a}{b}") for a, b in years if int(f"{a}{b}") >= 2000]
    return datetime.datetime.now().year - min(year_ints) if year_ints else 0


# ─────────────────────────────────────────────────────────────────────────────
# SKILL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(skills_text: str, full_text: str = "") -> list:
    skills: list = []

    if skills_text.strip():
        for token in re.split(r"[\n,•·\-–/|;]", skills_text):
            token = token.strip()
            if not token or len(token) < 2:
                continue
            if token.lower() in PROFICIENCY_WORDS:
                continue
            cleaned = _clean_skill(token)
            if cleaned and len(cleaned) > 1:
                if not any(n in cleaned.lower() for n in NOISE_SKILLS):
                    skills.append(cleaned)

    if not skills and full_text:
        ft_lower = full_text.lower()
        for tech in KNOWN_TECH:
            pattern = r"(?<![a-z0-9])" + re.escape(tech.lower()) + r"(?![a-z0-9])"
            if re.search(pattern, ft_lower):
                skills.append(tech)

    return list(dict.fromkeys(skills))


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_experience(exp_text: str) -> list:
    jobs: list = []
    if not exp_text.strip():
        return jobs

    orgs = extract_orgs(exp_text)
    lines = [l.strip() for l in exp_text.split("\n") if l.strip()]
    current_job: dict = {}
    current_bullets: list = []

    for line in lines:
        date_m = _DATE_RANGE_RE.search(line)
        if date_m:
            if current_job:
                current_job["responsibilities"] = current_bullets
                jobs.append(current_job)
            current_job = {
                "duration": date_m.group(0),
                "title": "",
                "company": "",
                "responsibilities": [],
            }
            current_bullets = []
            for org in orgs:
                if org.lower() in line.lower() and not current_job["company"]:
                    current_job["company"] = org

        elif line.startswith(("•", "-", "·", "*", "▪", "➤")) or (
            current_job and len(line) > 30
        ):
            current_bullets.append(re.sub(r"^[•\-·*▪➤]\s*", "", line).strip())

        elif current_job and not current_job.get("title"):
            current_job["title"] = line

        elif current_job and not current_job.get("company"):
            current_job["company"] = line if line in orgs else line

    if current_job:
        current_job["responsibilities"] = current_bullets
        jobs.append(current_job)

    return jobs


def detect_is_fresher(exp_text: str) -> bool:
    if len(exp_text.strip()) > 80:
        return False
    if re.search(r"\b(19|20)\d{2}\b", exp_text):
        return False
    job_kws = ["engineer", "developer", "designer", "analyst", "manager",
               "intern", "consultant", "researcher", "lead", "scientist"]
    return not any(k in exp_text.lower() for k in job_kws)


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_education(edu_text: str, raw_text: str) -> dict:
    edu_lines = [l for l in edu_text.split("\n") if l.strip()]
    edu: dict = {
        "degree":          edu_lines[0] if edu_lines else "",
        "field":           edu_lines[1] if len(edu_lines) > 1 else "",
        "institution":     "",
        "graduation_year": "",
        "gpa":             None,
    }

    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(edu_text[:1000])
        for ent in doc.ents:
            if ent.label_ == "ORG" and not edu["institution"]:
                edu["institution"] = ent.text.strip()

    if not edu["institution"]:
        for line in edu_lines:
            if any(kw in line.lower() for kw in ("university", "college", "institute", "school")):
                edu["institution"] = line.strip()
                break

    for line in edu_lines:
        yr = re.search(r"\b(19|20)\d{2}\b", line)
        if yr and not edu["graduation_year"]:
            edu["graduation_year"] = yr.group()

    for pattern in [
        r"(?:GPA|CGPA|grade)[:\s]+(\d+\.\d+)",
        r"(\d+\.\d+)\s*/\s*(?:4\.0|4)",
        r"(\d+\.\d+)\s*/\s*10",
    ]:
        m = re.search(pattern, raw_text, re.IGNORECASE)
        if m:
            edu["gpa"] = m.group(1)
            break

    return edu


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_resume(raw_text: str, filename: str) -> dict:
    """raw text → structured JSON dict."""
    sections = split_into_sections(raw_text)
    exp_txt  = sections.get("experience", "")
    edu      = extract_education(sections.get("education", ""), raw_text)

    return {
        "candidate_name":      extract_name(raw_text),
        "contact":             extract_contact(raw_text),
        "summary":             sections.get("summary", "")[:500].strip(),
        "skills":              extract_skills(sections.get("skills", ""), raw_text),
        "experience":          extract_experience(exp_txt),
        "projects":            [],
        "education":           edu,
        "gpa":                 edu.get("gpa"),
        "is_fresher":          detect_is_fresher(exp_txt),
        "years_of_experience": get_years_of_experience(exp_txt),
        "raw_sections":        sections,
        "source_file":         filename,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def build_embeddings(parsed: dict, model: SentenceTransformer) -> dict:
    emb: dict = {}

    if parsed.get("summary"):
        emb["summary"] = model.encode(parsed["summary"], normalize_embeddings=True)

    if parsed.get("skills"):
        emb["skills"] = model.encode(
            "Technical skills: " + ", ".join(parsed["skills"]),
            normalize_embeddings=True,
        )

    exp_text = parsed.get("raw_sections", {}).get("experience", "")
    if exp_text and not parsed.get("is_fresher"):
        emb["experience"] = model.encode(exp_text[:2000], normalize_embeddings=True)

    proj_text = parsed.get("raw_sections", {}).get("projects", "")
    if proj_text:
        emb["projects"] = model.encode(proj_text[:2000], normalize_embeddings=True)
    elif "experience" in emb:
        emb["projects"] = emb["experience"]

    edu = parsed.get("education", {})
    edu_str = f"{edu.get('degree','')} {edu.get('field','')} {edu.get('institution','')}".strip()
    if edu_str:
        emb["education"] = model.encode(edu_str, normalize_embeddings=True)

    full = " ".join(v for v in parsed.get("raw_sections", {}).values() if isinstance(v, str))[:3000]
    emb["full"] = model.encode(full, normalize_embeddings=True)

    return emb


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ResumeParser:
    """
    Loads model once, exposes process() and process_many().

    Example:
        from parser import ResumeParser
        parser     = ResumeParser()
        result     = parser.process("resume.pdf")
        parsed     = result["parsed"]
        embeddings = result["embeddings"]
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"⏳ Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model ready.")
        print(f"✅ spaCy NER      : {SPACY_MODEL if SPACY_AVAILABLE else 'UNAVAILABLE — python -m spacy download en_core_web_md'}")
        print(f"✅ Tesseract OCR  : active  ({TESS_CONFIG})")
        print(f"{'✅' if DATEPARSER_AVAILABLE else '⚠️ '} dateparser    : {'active (PREFER_DATES_FROM: past)' if DATEPARSER_AVAILABLE else 'unavailable — pip install dateparser'}")
        print(f"✅ PyMuPDF        : active  (300 DPI rendering)")
        print(f"✅ Image preproc  : greyscale → autocontrast → sharpen → binarise")

    def process(self, filepath: str) -> dict:
        """Parse a single PDF and return parsed + embeddings dict."""
        raw_text   = extract_raw_text(filepath)
        parsed     = parse_resume(raw_text, Path(filepath).name)
        embeddings = build_embeddings(parsed, self.model)
        return {"parsed": parsed, "embeddings": embeddings}

    def process_many(self, filepaths: list) -> dict:
        """Parse multiple PDFs and return a filename-keyed dict."""
        db: dict = {}
        for fp in filepaths:
            print(f"\n📄 Processing: {fp}")
            key     = Path(fp).name
            db[key] = self.process(fp)
            p       = db[key]["parsed"]
            print(f"   ✅ Name   : {p['candidate_name']}")
            print(f"   ✅ Skills : {p['skills'][:5]}")
            print(f"   ✅ Fresher: {p['is_fresher']}  |  YoE: {p['years_of_experience']}")
            print(f"   ✅ Jobs   : {len(p['experience'])}")
        return db
