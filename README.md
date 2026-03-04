# ATS Resume Screening System

PDF resume parser + ATS scorer + domain clusterer — converted from Google Colab to a clean, importable Python package.

---

## Project Structure

```
ats_system/
├── __init__.py          # Package init — import from here
├── parser.py            # PDF → structured JSON + embeddings
├── matcher.py           # ATS scoring against HR requirements
├── clusterer.py         # Zero-shot domain clustering
├── requirements.py      # Example HR requirement configs
└── main.py              # CLI entry point
```

---

## Installation

```bash
# Tesseract OCR binary (required for scanned PDFs)
# Ubuntu / Debian:
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract

# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# Python dependencies
pip install pdfplumber "pdfminer.six" PyMuPDF pytesseract Pillow \
            spacy dateparser rapidfuzz sentence-transformers \
            scikit-learn numpy torch

# spaCy language model (medium is better than small for NER)
python -m spacy download en_core_web_md
```

---

## CLI Usage

```bash
# Parse a folder of PDFs, score against the Web Dev preset, export JSON
python main.py --resumes ./resumes/ --role web --output results.json

# Deep Learning role
python main.py --resumes ./resumes/ --role dl

# Single resume file
python main.py --resumes candidate.pdf

# Run with no arguments for interactive prompts
python main.py
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--resumes` / `-r` | Path to a PDF file **or** a folder of PDFs |
| `--role` | HR preset: `web` or `dl` (default: `web`) |
| `--output` / `-o` | Optional JSON export path |

---

## Python API Usage

```python
from ats_system.parser    import ResumeParser
from ats_system.matcher   import ResumeMatcher
from ats_system.clusterer import ResumeClusterer

# 1. Initialise (loads embedding model once — ~1–2 min first time)
parser    = ResumeParser()
matcher   = ResumeMatcher(parser.model)
clusterer = ResumeClusterer(parser.model)

# 2. Parse resumes
resume_db = parser.process_many(["cv1.pdf", "cv2.pdf", "cv3.pdf"])

# 3. ATS matching
HR_REQUIREMENTS = {
    "job_role": "Backend Engineer",
    "role_description": "Python backend developer with FastAPI and PostgreSQL",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "preferred_skills": ["Redis", "AWS", "Kubernetes"],
    "projects_required": {"min_count": 1, "description": "API projects"},
    "education_required": {"field": "Computer Science"},
    "experience_required": {"freshers_allowed": False},
}

matcher.set_requirements(HR_REQUIREMENTS)
results = matcher.match_all(resume_db)    # sorted list, highest score first
matcher.print_summary(results)

# 4. Cluster by domain
clusters = clusterer.cluster(resume_db)
clusterer.print_summary(clusters)

# 5. Inspect one result
best = results[0]
print(best["name"])           # candidate name
print(best["total_score"])    # e.g. 78.4
print(best["section_scores"]) # {"skills": 85.0, "experience": 72.0, ...}
print(best["parsed"]["skills"])  # ["Python", "FastAPI", ...]
```

---

## Output Format

### ATS Result (one entry in `results` list)

```json
{
  "name": "Jane Doe",
  "filename": "jane_cv.pdf",
  "total_score": 78.4,
  "section_scores": {
    "skills": 85.0,
    "experience": 72.0,
    "projects": 60.0,
    "education": 75.0,
    "summary": 68.0
  },
  "details": {
    "skills": "8/10 required, 3/6 preferred",
    "experience": "2 job(s) | 6/10 skills in exp",
    "projects": "2 project(s) (inferred from experience)",
    "education": "Field matched ✅ | State University"
  }
}
```

### Cluster Result (one domain entry)

```json
{
  "🌐 Web / Full Stack": [
    {
      "name": "Jane Doe",
      "primary_score": 81.2,
      "secondary": "☁️ Cloud / DevOps",
      "secondary_score": 76.5,
      "richness": 72,
      "reason": "confident",
      "top_skills": ["React", "Node.js", "MySQL", "CSS", "JavaScript"]
    }
  ]
}
```

---

## Customising HR Requirements

Edit or extend `requirements.py`:

```python
MY_REQUIREMENTS = {
    "job_role":        "Data Analyst",
    "role_description": "...",
    "required_skills":  ["SQL", "Python", "Excel", "Power BI"],
    "preferred_skills": ["Tableau", "R", "Spark"],
    "projects_required":  {"min_count": 1, "description": "..."},
    "education_required": {"field": "Statistics or Computer Science"},
    "experience_required": {"freshers_allowed": True},
}
```

---

## Modules at a Glance

| Module | Key Class | What it does |
|--------|-----------|--------------|
| `parser.py` | `ResumeParser` | PDF → raw text (pdfplumber / pdfminer / Tesseract OCR) → spaCy NER → structured dict + sentence-transformer embeddings |
| `matcher.py` | `ResumeMatcher` | Fuzzy + semantic section-by-section scoring against HR requirements |
| `clusterer.py` | `ResumeClusterer` | Cosine similarity + rapidfuzz keyword boost → domain label |
| `requirements.py` | — | Pre-built HR requirement dicts (web, dl) |
| `main.py` | — | CLI entry point + interactive mode |



