"""
clusterer.py  (v2 — upgraded)
------------------------------
Zero-shot domain clustering of parsed resumes.

Install:
    pip install rapidfuzz sentence-transformers scikit-learn numpy

Usage:
    from clusterer import ResumeClusterer
    clusterer = ResumeClusterer(model)
    clusters  = clusterer.cluster(resume_db)
    clusterer.print_summary(clusters)
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    from rapidfuzz import fuzz, process as rfuzz_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_ANCHORS: dict[str, str] = {
    "🌐 Web / Full Stack": (
        "HTML CSS JavaScript TypeScript React Angular Vue Node.js Express Django Flask "
        "REST API GraphQL frontend backend full stack responsive design UI UX Figma Tailwind "
        "Next.js Nuxt PHP Laravel Spring Boot web development"
    ),
    "🧠 Deep Learning / AI": (
        "deep learning neural networks CNN RNN LSTM transformer BERT GPT diffusion model "
        "PyTorch TensorFlow Keras computer vision NLP natural language processing object detection "
        "YOLO CUDA GPU training fine-tuning Hugging Face embeddings LLM generative AI"
    ),
    "📊 Data Science / Analytics": (
        "data science machine learning scikit-learn pandas numpy matplotlib seaborn "
        "statistics regression classification clustering feature engineering EDA "
        "data visualization Power BI Tableau SQL data analysis A/B testing business intelligence "
        "MLflow Spark big data ETL pipeline"
    ),
    "☁️ Cloud / DevOps": (
        "AWS Azure GCP cloud infrastructure Docker Kubernetes CI/CD Jenkins GitLab Terraform "
        "Ansible Linux bash scripting microservices serverless Lambda EC2 S3 networking DevOps "
        "SRE reliability monitoring Prometheus Grafana"
    ),
    "📱 Mobile Development": (
        "Android iOS Swift Kotlin React Native Flutter mobile app development "
        "Firebase push notifications Play Store App Store Jetpack Compose SwiftUI "
        "mobile UI UX performance optimization"
    ),
}

DOMAIN_KEYWORDS: dict[str, list] = {
    "🌐 Web / Full Stack":         ["html", "css", "javascript", "react", "vue", "angular",
                                     "node.js", "django", "flask", "rest api", "frontend", "backend",
                                     "spring boot", "typescript", "next.js"],
    "🧠 Deep Learning / AI":       ["pytorch", "tensorflow", "keras", "cnn", "rnn", "lstm",
                                     "transformer", "bert", "gpt", "yolo", "cuda", "hugging face",
                                     "nlp", "computer vision", "llm", "deep learning"],
    "📊 Data Science / Analytics": ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
                                     "data science", "machine learning", "sql", "tableau", "power bi",
                                     "spark", "etl", "regression", "classification"],
    "☁️ Cloud / DevOps":           ["aws", "azure", "gcp", "docker", "kubernetes", "terraform",
                                     "ci/cd", "jenkins", "linux", "devops", "microservices", "ansible"],
    "📱 Mobile Development":        ["android", "ios", "flutter", "swift", "kotlin", "react native",
                                     "firebase", "jetpack compose", "swiftui"],
}

FRESHER_LABEL        = "🎓 Fresher / General"
CONFIDENCE_THRESHOLD = 62.0
AMBIGUOUS_GAP        = 3.0
KEYWORD_BOOST        = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT RICHNESS
# ─────────────────────────────────────────────────────────────────────────────

def content_richness(parsed: dict) -> int:
    score  = 0
    score += min(30, len(parsed.get("skills",     [])) * 5)
    score += min(20, len(parsed.get("experience", [])) * 10)
    score += min(20, len(parsed.get("projects",   [])) * 10)
    score += min(15, len(parsed.get("summary", "").split()) // 5)
    score += min(10, len(parsed.get("certifications", [])) * 5)
    raw_words = sum(
        len(v.split()) for v in parsed.get("raw_sections", {}).values()
        if isinstance(v, str)
    )
    score += min(5, raw_words // 50)
    return min(100, score)


# ─────────────────────────────────────────────────────────────────────────────
# RAPIDFUZZ KEYWORD BOOST
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_boost_scores(parsed: dict) -> dict:
    if not RAPIDFUZZ_AVAILABLE:
        return {d: 0.0 for d in DOMAIN_KEYWORDS}

    candidate_skills = [s.lower() for s in parsed.get("skills", [])]
    full_text_lower  = " ".join(
        v.lower() for v in parsed.get("raw_sections", {}).values() if isinstance(v, str)
    )
    boosts: dict = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matched = 0
        for kw in keywords:
            hit = rfuzz_process.extractOne(
                kw, candidate_skills,
                scorer=fuzz.token_set_ratio,
                score_cutoff=82,
            )
            if hit:
                matched += 1
                continue
            if kw in full_text_lower:
                matched += 1

        boosts[domain] = min(20.0, matched * KEYWORD_BOOST)

    return boosts


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_resume(
    parsed: dict,
    embeddings: dict,
    anchor_embeddings: dict,
) -> dict:
    richness = content_richness(parsed)

    if richness < 25:
        return {
            "primary": FRESHER_LABEL, "primary_score": 100.0,
            "secondary": None,        "secondary_score": 0.0,
            "richness": richness,     "reason": "insufficient content to classify",
        }

    vecs = []
    if "full"   in embeddings: vecs.append((embeddings["full"],   0.5))
    if "skills" in embeddings: vecs.append((embeddings["skills"], 0.5))
    if not vecs:
        return {
            "primary": FRESHER_LABEL, "primary_score": 0.0,
            "secondary": None,        "secondary_score": 0.0,
            "richness": richness,     "reason": "no embeddings",
        }

    combo = sum(w * e for e, w in vecs)
    norm  = np.linalg.norm(combo)
    if norm == 0:
        return {
            "primary": FRESHER_LABEL, "primary_score": 0.0,
            "secondary": None,        "secondary_score": 0.0,
            "richness": richness,     "reason": "zero vector",
        }
    combo /= norm

    raw_scores: dict = {}
    for domain, anchor_emb in anchor_embeddings.items():
        sim = float(cosine_similarity(combo.reshape(1, -1), anchor_emb.reshape(1, -1))[0][0])
        raw_scores[domain] = round(((sim + 1) / 2) * 100, 1)

    boosts = _keyword_boost_scores(parsed)
    for domain in raw_scores:
        raw_scores[domain] = min(100.0, round(raw_scores[domain] + boosts.get(domain, 0.0), 1))

    ranked     = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = ranked[0]
    sec_domain, sec_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

    if sec_score and (top_score - sec_score) < AMBIGUOUS_GAP:
        return {
            "primary": top_domain,   "primary_score": top_score,
            "secondary": sec_domain, "secondary_score": sec_score,
            "richness": richness,    "reason": f"ambiguous ({top_score}% vs {sec_score}%)",
        }

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "primary": FRESHER_LABEL, "primary_score": top_score,
            "secondary": top_domain,  "secondary_score": sec_score,
            "richness": richness,     "reason": f"low confidence ({top_score}%)",
        }

    return {
        "primary": top_domain,   "primary_score": top_score,
        "secondary": sec_domain, "secondary_score": sec_score,
        "richness": richness,    "reason": "confident",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ResumeClusterer:
    """
    Groups resume_db into domain clusters.

    Example:
        from clusterer import ResumeClusterer
        clusterer = ResumeClusterer(parser.model)
        clusters  = clusterer.cluster(resume_db)
        clusterer.print_summary(clusters)
    """

    def __init__(self, model: SentenceTransformer, anchors: dict = None):
        self.model    = model
        self.anchors  = anchors or DOMAIN_ANCHORS
        self.anchor_embeddings: dict = {}
        self.flat_results: dict = {}
        self._encode_anchors()
        if RAPIDFUZZ_AVAILABLE:
            print("✅ rapidfuzz keyword boosting active for clustering")
        else:
            print("⚠️  rapidfuzz not installed — no keyword boost (pip install rapidfuzz)")

    def _encode_anchors(self) -> None:
        print("⚙️  Encoding domain anchors...")
        for domain, text in self.anchors.items():
            self.anchor_embeddings[domain] = self.model.encode(text, normalize_embeddings=True)
        print(f"✅ {len(self.anchor_embeddings)} domains ready.\n")

    def cluster(self, resume_db: dict) -> dict:
        """Classify all resumes and group by primary domain."""
        groups: dict = defaultdict(list)
        self.flat_results = {}

        for filename, data in resume_db.items():
            parsed = data["parsed"]
            embs   = data["embeddings"]

            result = classify_resume(parsed, embs, self.anchor_embeddings)
            result["name"]       = parsed["candidate_name"]
            result["filename"]   = filename
            result["top_skills"] = parsed.get("skills", [])[:6]
            result["is_fresher"] = parsed.get("is_fresher", False)

            self.flat_results[filename] = result
            groups[result["primary"]].append(result)

        for domain in groups:
            groups[domain].sort(key=lambda x: x["primary_score"], reverse=True)

        return dict(groups)

    def print_summary(self, clusters: dict) -> None:
        total = sum(len(v) for v in clusters.values())
        print(f"\n🗂️  Resume Cluster Summary  ({total} resume(s))")
        print("=" * 65)
        for domain, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
            print(f"\n  {domain}  ({len(members)} resume(s))")
            for m in members:
                warn = " ⚠️  sparse" if m["richness"] < 25 else ""
                sec  = (
                    f"  ↳ also {m['secondary']} ({m['secondary_score']}%)"
                    if m.get("secondary") and m["secondary"] != domain
                    else ""
                )
                print(
                    f"    • {m['name']:<30}  richness: {m['richness']:>3}/100"
                    f"  [{m['primary_score']}%]{warn}{sec}"
                )
