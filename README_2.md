# 🎯 ATS Resume Screening System

> **A complete AI-powered Applicant Tracking System** — parses PDFs, scores resumes against HR requirements, and clusters candidates by domain.  
> Built with spaCy · sentence-transformers · Tesseract OCR · rapidfuzz · pdfplumber

---

## 📋 Table of Contents

1. [What This System Does](#-what-this-system-does)
2. [Project Structure](#-project-structure)
3. [System Requirements](#-system-requirements)
4. [Step-by-Step Installation](#-step-by-step-installation)
5. [Quick Start](#-quick-start)
6. [Running the System — Full Guide](#-running-the-system--full-guide)
7. [CLI Commands Reference](#-cli-commands-reference)
8. [Python API Usage](#-python-api-usage)
9. [HR Requirements Config](#-hr-requirements-config)
10. [Understanding the Output](#-understanding-the-output)
11. [Scoring Breakdown](#-scoring-breakdown)
12. [Domain Clusters](#-domain-clusters)
13. [Troubleshooting](#-troubleshooting)
14. [FAQ](#-faq)

---

## 🧠 What This System Does

```
PDF Resumes
    │
    ▼
┌─────────────────────────────────────────────┐
│  STAGE 1 — PARSER  (parser.py)              │
│  pdfplumber → pdfminer → Tesseract OCR      │
│  spaCy NER (names, orgs, locations)         │
│  → Structured JSON + sentence embeddings    │
└──────────────────────┬──────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐       ┌──────────────────────┐
│  STAGE 2        │       │  STAGE 3             │
│  MATCHER        │       │  CLUSTERER           │
│  (matcher.py)   │       │  (clusterer.py)      │
│                 │       │                      │
│  Scores each    │       │  Groups resumes by   │
│  resume against │       │  tech domain using   │
│  HR requirements│       │  cosine similarity   │
│  using fuzzy    │       │  + rapidfuzz keyword │
│  matching +     │       │  boosting            │
│  semantic search│       │                      │
└────────┬────────┘       └──────────┬───────────┘
         │                           │
         ▼                           ▼
   Ranked Candidate            Domain Clusters
   Scores (0–100%)             (Web / AI / Cloud...)
```

---

## 📁 Project Structure

```
ats_system/
│
├── 📄 main.py              ← Entry point — run this to use the system
├── 📄 parser.py            ← PDF extraction + NER + embeddings
├── 📄 matcher.py           ← ATS scoring engine
├── 📄 clusterer.py         ← Domain classification
├── 📄 requirements.py      ← HR requirement presets (web, dl)
├── 📄 __init__.py          ← Package definition (import from here)
├── 📄 install.sh           ← One-shot installer script
└── 📄 README.md            ← This file

your-project/
├── ats_system/             ← Drop the folder here
├── resumes/                ← Put your PDF resumes here
│   ├── alice_cv.pdf
│   ├── bob_cv.pdf
│   └── carol_cv.pdf
└── results.json            ← Output exported here
```

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 4 GB | 8 GB+ |
| Disk space | 5 GB | 10 GB (models cache) |
| GPU | Not required | CUDA GPU (speeds up embedding) |
| OS | Windows / macOS / Linux | Ubuntu 20.04+ |

> ⚠️ **First run downloads ~1.5 GB** of model files (BAAI/bge-large-en-v1.5 + spaCy en_core_web_md). This happens once and is cached locally.

---

## 🚀 Step-by-Step Installation

### Step 1 — Clone or copy the project

```bash
# If using git:
git clone <your-repo-url>
cd ats_system

# Or just copy the ats_system/ folder into your project
```

---

### Step 2 — Create a virtual environment (strongly recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On macOS / Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 — Install Tesseract OCR binary

Tesseract is a separate program (not a Python package) that must be installed on your system.

**Ubuntu / Debian / WSL:**
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

**macOS (with Homebrew):**
```bash
brew install tesseract
```

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer — install to `C:\Program Files\Tesseract-OCR\`
3. Add to PATH: Search "Environment Variables" → System Variables → Path → Add `C:\Program Files\Tesseract-OCR\`

**Verify Tesseract is installed:**
```bash
tesseract --version
# Should print: tesseract 5.x.x
```

---

### Step 4 — Install Python dependencies

```bash
pip install pdfplumber "pdfminer.six" PyMuPDF pytesseract Pillow \
            spacy dateparser rapidfuzz \
            sentence-transformers scikit-learn numpy torch
```

> ⏳ This will take 3–10 minutes depending on your internet speed.

---

### Step 5 — Download the spaCy language model

```bash
python -m spacy download en_core_web_md
```

> The `md` (medium) model is significantly better than `sm` for name/org extraction. ~50 MB download.

---

### Step 6 — Verify everything is working

```bash
python -c "
import pdfplumber, fitz, pytesseract, spacy, dateparser, rapidfuzz
from sentence_transformers import SentenceTransformer
print('All packages imported successfully')
print('Tesseract:', pytesseract.get_tesseract_version())
"
```

Expected output:
```
All packages imported successfully
Tesseract: 5.x.x
```

---

### OR: Use the one-shot installer script

```bash
chmod +x install.sh
bash install.sh
```

This auto-detects your OS and installs everything in one go.

---

## ⚡ Quick Start

```bash
# 1. Put your PDF resumes in a folder
mkdir resumes
# copy your .pdf files into resumes/

# 2. Run the system (interactive mode — no flags needed)
python main.py

# It will ask you:
#   - Path to your resumes folder
#   - Which HR role preset to use
#   - Whether to export results
```

---

## 🔧 Running the System — Full Guide

### STEP 1 — Prepare your resumes

Create a folder and put all your PDF resumes inside:
```
resumes/
├── alice_smith.pdf
├── bob_jones.pdf
└── carol_lee.pdf
```

> ✅ Any PDF works — text-based, scanned, image-only, multi-column layouts
> ⚠️ Only `.pdf` files are processed — Word docs must be exported to PDF first

---

### STEP 2 — Choose or create an HR requirements profile

**Option A — Use a built-in preset** (fastest)

| Preset | Job Role |
|--------|----------|
| `web`  | Junior Web / Full Stack Developer |
| `dl`   | Deep Learning / AI Engineer |

**Option B — Create your own** in `requirements.py`:

```python
MY_CUSTOM_REQUIREMENTS = {
    "job_role": "Backend Engineer",
    "role_description": "Python backend developer with FastAPI experience...",
    "required_skills":  ["Python", "FastAPI", "PostgreSQL", "Docker", "Git"],
    "preferred_skills": ["Redis", "AWS", "Kubernetes", "CI/CD"],
    "projects_required":  {"min_count": 2, "description": "API or backend projects"},
    "education_required": {"field": "Computer Science or related"},
    "experience_required": {"freshers_allowed": False},
}
```

Then import it in `main.py`:
```python
from requirements import MY_CUSTOM_REQUIREMENTS
ROLE_MAP["backend"] = MY_CUSTOM_REQUIREMENTS
```

---

### STEP 3 — Run the system

```bash
# Standard run — web dev role, results saved to JSON
python main.py --resumes ./resumes/ --role web --output results.json

# Deep learning role
python main.py --resumes ./resumes/ --role dl --output results.json

# Interactive mode (step-by-step prompts)
python main.py

# Single resume test
python main.py --resumes ./resumes/alice_smith.pdf --role web
```

---

### STEP 4 — Read the output

The terminal will print:

**1. Parsing progress** (one line per resume):
```
📄 Processing: alice_smith.pdf
   ✅ Name   : Alice Smith
   ✅ Skills : ['Python', 'React', 'PostgreSQL', 'Docker', 'Git']
   ✅ Fresher: False  |  YoE: 2.5
   ✅ Jobs   : 2
```

**2. ATS Ranking** (sorted best to worst):
```
═══════════════════════════════════════════════════════════════
🎯  ATS MATCH RESULTS
═══════════════════════════════════════════════════════════════

  🥇  Alice Smith  (alice_smith.pdf)
      Total Score   : 81.4%  🟢 Excellent
      Skills        : 85.0%  — 8/10 required, 4/6 preferred
      Experience    : 72.0%  — 2 job(s) | 6/10 skills in exp
      Projects      : 68.0%  — 2 project(s) (dedicated section)
      Education     : 80.0%  — Field matched ✅ | State University
      Summary       : 74.0%

  🥈  Bob Jones  (bob_jones.pdf)
      Total Score   : 63.2%  🟡 Good
```

**3. Domain Clusters**:
```
🗂️  Resume Cluster Summary  (3 resume(s))
═══════════════════════════════════════════
  🌐 Web / Full Stack  (2 resume(s))
    • Alice Smith     richness:  78/100  [81.4%]
    • Bob Jones       richness:  55/100  [63.2%]

  🧠 Deep Learning / AI  (1 resume(s))
    • Carol Lee       richness:  82/100  [79.8%]
```

---

## 📟 CLI Commands Reference

### Basic syntax
```bash
python main.py --resumes <path> --role <preset> --output <file.json>
```

### All flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--resumes` | `-r` | path | required | PDF file or folder of PDFs |
| `--role` | — | string | `web` | HR preset: `web`, `dl`, or `custom` |
| `--output` | `-o` | path | none | Save results as JSON |

### All command examples

```bash
# ── Most common ─────────────────────────────────────────────────

# Parse a folder, web role, save output
python main.py --resumes ./resumes/ --role web --output results.json

# Parse a folder, deep learning role
python main.py --resumes ./resumes/ --role dl --output dl_results.json

# Interactive mode (no args — prompts everything)
python main.py

# ── Single resume ────────────────────────────────────────────────

python main.py --resumes ./resumes/alice_smith.pdf --role web

# ── Custom role ──────────────────────────────────────────────────

python main.py --resumes ./resumes/ --role custom
# Then follow the prompts to enter requirements

# ── No JSON export (terminal only) ──────────────────────────────

python main.py --resumes ./resumes/ --role web

# ── Absolute paths also work ─────────────────────────────────────

python main.py --resumes /home/user/candidates/ --role dl --output /tmp/results.json
```

---

## 🐍 Python API Usage

Use the system inside your own Python project by importing the modules directly.

### Minimal example

```python
from ats_system.parser    import ResumeParser
from ats_system.matcher   import ResumeMatcher
from ats_system.clusterer import ResumeClusterer

# Step 1: Initialise (loads model — ~1 min first time)
parser    = ResumeParser()
matcher   = ResumeMatcher(parser.model)   # share same model
clusterer = ResumeClusterer(parser.model) # share same model

# Step 2: Parse resumes
resume_db = parser.process_many(["cv1.pdf", "cv2.pdf"])

# Step 3: Set HR requirements and score
HR_REQ = {
    "job_role":        "Backend Engineer",
    "role_description": "Python backend with FastAPI and PostgreSQL",
    "required_skills":  ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "preferred_skills": ["Redis", "AWS"],
    "projects_required":   {"min_count": 1, "description": "API projects"},
    "education_required":  {"field": "Computer Science"},
    "experience_required": {"freshers_allowed": False},
}
matcher.set_requirements(HR_REQ)
results = matcher.match_all(resume_db)  # sorted best to worst

# Step 4: Cluster by domain
clusters = clusterer.cluster(resume_db)

# Step 5: Use the results
for r in results:
    print(f"{r['name']:30} {r['total_score']:.1f}%")
```

---

### Accessing parsed resume data

```python
resume_db = parser.process_many(["alice.pdf"])
parsed = resume_db["alice.pdf"]["parsed"]

print(parsed["candidate_name"])       # "Alice Smith"
print(parsed["contact"]["email"])     # "alice@email.com"
print(parsed["contact"]["phone"])     # "+91 98765 43210"
print(parsed["contact"]["linkedin"])  # "linkedin.com/in/alice"
print(parsed["contact"]["github"])    # "github.com/alicesmith"
print(parsed["skills"])               # ["Python", "React", "PostgreSQL", ...]
print(parsed["years_of_experience"])  # 2.5
print(parsed["is_fresher"])           # False
print(parsed["gpa"])                  # "8.7" or None
print(parsed["education"])            # {"degree": "...", "institution": "...", ...}
print(parsed["experience"])           # [{"title": "...", "company": "...", ...}]
print(parsed["summary"])              # "Experienced developer..."
```

---

### Accessing match result data

```python
results = matcher.match_all(resume_db)
best    = results[0]  # highest scorer

print(best["name"])                         # "Alice Smith"
print(best["filename"])                     # "alice.pdf"
print(best["total_score"])                  # 81.4
print(best["section_scores"]["skills"])     # 85.0
print(best["section_scores"]["experience"]) # 72.0
print(best["section_scores"]["projects"])   # 68.0
print(best["section_scores"]["education"])  # 80.0
print(best["section_scores"]["summary"])    # 74.0
print(best["details"]["skills"])            # "8/10 required, 4/6 preferred"
print(best["details"]["experience"])        # "2 job(s) | 6/10 skills in exp"
```

---

### Processing a single resume

```python
result     = parser.process("single_resume.pdf")
parsed     = result["parsed"]
embeddings = result["embeddings"]
```

---

## ⚙️ HR Requirements Config

```python
HR_REQUIREMENTS = {

    # ── Required ─────────────────────────────────────────────────

    "job_role": "Junior Web Developer",
    # Name of the job (shown in output headers)

    "role_description": "Longer description of the role and ideal candidate.",
    # Used for semantic summary matching via embeddings

    "required_skills": ["HTML", "CSS", "JavaScript", "Java", "Spring Boot"],
    # MUST-HAVE skills. Weighted 75% of skills score.
    # Fuzzy matching handles: "React.js" = "ReactJS" = "react js"

    "preferred_skills": ["React.js", "REST API", "Docker"],
    # Nice-to-have. Weighted 20% of skills score.

    # ── Optional ──────────────────────────────────────────────────

    "projects_required": {
        "min_count": 2,
        "description": "Type of projects expected (used for semantic matching).",
    },

    "education_required": {
        "degree": "Bachelor of Computer Science",
        "field":  "Computer Science or related field",
        "min_gpa": 7.0,  # set 0 to skip GPA check
    },

    "experience_required": {
        "freshers_allowed": True,
        # True  → fresher gets 45/100 (fair chance)
        # False → fresher gets 0/100 (disqualified)
        "description": "Context for semantic experience matching.",
    },
}
```

---

## 📊 Understanding the Output

### Score grades

| Score | Grade | Meaning |
|-------|-------|---------|
| 80–100% | 🟢 Excellent | Strong match — shortlist immediately |
| 60–79% | 🟡 Good | Good match — worth interviewing |
| 45–59% | 🟠 Partial | Some gaps — consider with caution |
| 0–44% | 🔴 Weak | Significant skill gaps |

### JSON output structure

```json
{
  "job_role": "Junior Web Developer",
  "ats_results": [
    {
      "name": "Alice Smith",
      "filename": "alice_smith.pdf",
      "total_score": 81.4,
      "section_scores": {
        "skills": 85.0,
        "experience": 72.0,
        "projects": 68.0,
        "education": 80.0,
        "summary": 74.0
      },
      "details": {
        "skills":     "8/10 required, 4/6 preferred",
        "experience": "2 job(s) | 6/10 skills in exp",
        "projects":   "2 project(s) (dedicated section)",
        "education":  "GPA 8.7 | Field matched ✅ | State University"
      }
    }
  ],
  "clusters": {
    "🌐 Web / Full Stack": [
      {
        "name": "Alice Smith",
        "primary_score": 81.4,
        "secondary": "📊 Data Science / Analytics",
        "secondary_score": 72.1,
        "richness": 78,
        "top_skills": ["React", "JavaScript", "PostgreSQL", "Docker"]
      }
    ]
  }
}
```

---

## 📐 Scoring Breakdown

```
Total Score =
  (skills     × 0.37)
+ (experience × 0.15)
+ (projects   × 0.20)
+ (education  × 0.40)
+ (summary    × 0.30)
```

| Section | Weight | How it's calculated |
|---------|--------|---------------------|
| Skills | 37% | required_matched/total × 75% + preferred × 20% + bonus |
| Experience | 15% | keyword overlap 50% + semantic similarity 50% |
| Projects | 20% | count vs min_count 40% + semantic similarity 60% |
| Education | 40% | base 70 + field match +10 + GPA +/-20 + institution +5 |
| Summary | 30% | pure cosine similarity vs role description |

---

## 🗂️ Domain Clusters

| Domain | Typical Skills |
|--------|---------------|
| 🌐 Web / Full Stack | HTML, CSS, JS, React, Node.js, Spring Boot |
| 🧠 Deep Learning / AI | PyTorch, TensorFlow, CNN, BERT, GPT, YOLO |
| 📊 Data Science / Analytics | pandas, scikit-learn, SQL, Power BI, Spark |
| ☁️ Cloud / DevOps | AWS, Docker, Kubernetes, CI/CD, Terraform |
| 📱 Mobile Development | Flutter, Swift, Kotlin, Android, iOS |
| 🎓 Fresher / General | Sparse resume or low content richness |

**Richness score** (0–100): how content-rich the resume is.  
Below 25 → automatically placed in Fresher/General regardless of skills.

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `TesseractNotFoundError` | `sudo apt install tesseract-ocr` (Linux) or `brew install tesseract` (Mac) |
| `Can't find model 'en_core_web_md'` | `python -m spacy download en_core_web_md` |
| `No module named 'fitz'` | `pip install PyMuPDF` |
| `No module named 'rapidfuzz'` | `pip install rapidfuzz` |
| Low text yield / blank output | PDF is scanned — Tesseract will auto-activate. If still failing, PDF may be password-protected. |
| Wrong name extracted | Name on PDF not on its own line. Override: `resume_db["f.pdf"]["parsed"]["candidate_name"] = "Correct Name"` |
| Out of memory | Use smaller model: `ResumeParser(model_name="BAAI/bge-small-en-v1.5")` |
| Slow first run | Normal — downloading ~1.5 GB of models. Cached after first run. |

---

## ❓ FAQ

**Q: Can I process Word (.docx) files?**  
Export to PDF first: File → Save As → PDF in Microsoft Word.

**Q: How many resumes at once?**  
No hard limit. 50 resumes ≈ 3–5 min. 200+ resumes ≈ 15–20 min.

**Q: Does it work on scanned PDFs?**  
Yes. Falls back to Tesseract OCR with image preprocessing automatically.

**Q: Is the model downloaded every time?**  
No. Cached in `~/.cache/huggingface/` after first download.

**Q: Can I use a smaller/faster model?**  
```python
parser = ResumeParser(model_name="all-MiniLM-L6-v2")  # small, fast, less accurate
```

**Q: Can I embed this in a Flask/FastAPI app?**  
```python
from ats_system.parser  import ResumeParser
from ats_system.matcher import ResumeMatcher

parser  = ResumeParser()               # load ONCE at app startup
matcher = ResumeMatcher(parser.model)  # share same model

@app.post("/screen")
def screen(pdf_path: str, hr_req: dict):
    resume_db = parser.process_many([pdf_path])
    matcher.set_requirements(hr_req)
    return matcher.match_all(resume_db)
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pdfplumber` | Spatial word-position PDF text extraction |
| `pdfminer.six` | Fallback for complex font encodings |
| `PyMuPDF (fitz)` | 300 DPI page rendering for OCR |
| `pytesseract` | OCR engine for scanned PDFs |
| `Pillow` | Image preprocessing (greyscale, contrast, binarise) |
| `spaCy en_core_web_md` | NER (names, orgs, locations) + POS tagging |
| `dateparser` | Robust date parsing for work history |
| `rapidfuzz` | Fuzzy skill matching (React vs ReactJS etc.) |
| `sentence-transformers` | Semantic embeddings (BAAI/bge-large-en-v1.5) |
| `scikit-learn` | Cosine similarity |
| `numpy` | Vector math |

---

*MIT License — free to use in personal and commercial projects.*
