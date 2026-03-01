"""
main.py
-------
ATS Resume Screening System — CLI Entry Point

Usage:
    # Basic: parse PDFs in a folder and run full pipeline
    python main.py --resumes ./resumes/

    # With custom HR requirements
    python main.py --resumes ./resumes/ --role web

    # Export results to JSON
    python main.py --resumes ./resumes/ --role dl --output results.json

    # Single resume
    python main.py --resumes resume.pdf

Available --role presets: web, dl  (or extend requirements.py with your own)
"""

import argparse
import json
import sys
import os
from pathlib import Path

# ── Local modules ─────────────────────────────────────────────────────────────
from parser    import ResumeParser
from matcher   import ResumeMatcher
from clusterer import ResumeClusterer
from requirements import WEB_DEV_REQUIREMENTS, DL_REQUIREMENTS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ROLE_MAP = {
    "web": WEB_DEV_REQUIREMENTS,
    "dl":  DL_REQUIREMENTS,
}

BANNER = """
╔══════════════════════════════════════════════════════╗
║         ATS Resume Screening System  v3              ║
║  parser · matcher · clusterer                        ║
╚══════════════════════════════════════════════════════╝
"""


def collect_pdfs(path: str) -> list[str]:
    """Accept a file path or directory and return list of PDF paths."""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() == ".pdf":
            return [str(p)]
        print(f"⚠️  {path} is not a PDF — skipping.")
        return []
    if p.is_dir():
        pdfs = sorted(p.glob("*.pdf"))
        if not pdfs:
            print(f"⚠️  No PDF files found in {path}")
        return [str(f) for f in pdfs]
    print(f"❌  Path not found: {path}")
    return []


def print_ats_results(results: list) -> None:
    """Pretty-print ATS match results to terminal."""
    print("\n" + "=" * 65)
    print("🎯  ATS MATCH RESULTS")
    print("=" * 65)
    for i, r in enumerate(results, 1):
        sc = r["total_score"]
        if sc >= 80:
            grade = "🟢 Excellent"
        elif sc >= 60:
            grade = "🟡 Good"
        elif sc >= 45:
            grade = "🟠 Partial"
        else:
            grade = "🔴 Weak"

        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"  #{i}"
        print(f"\n  {medal}  {r['name']}  ({r['filename']})")
        print(f"      Total Score   : {sc:.1f}%  {grade}")
        print(f"      Skills        : {r['section_scores']['skills']:.1f}%  — {r['details']['skills']}")
        print(f"      Experience    : {r['section_scores']['experience']:.1f}%  — {r['details']['experience']}")
        print(f"      Projects      : {r['section_scores']['projects']:.1f}%  — {r['details']['projects']}")
        print(f"      Education     : {r['section_scores']['education']:.1f}%  — {r['details']['education']}")
        print(f"      Summary       : {r['section_scores']['summary']:.1f}%")
    print()


def export_json(
    resume_db: dict,
    results: list,
    clusters: dict,
    hr_req: dict,
    output_path: str,
) -> None:
    """Export full pipeline output to JSON."""
    output = {
        "job_role": hr_req.get("job_role", ""),
        "ats_results": [
            {k: v for k, v in r.items() if k != "parsed"}
            for r in results
        ],
        "clusters": clusters,
        "parsed_resumes": [
            {k: v for k, v in data["parsed"].items() if k != "raw_sections"}
            for data in resume_db.values()
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"✅ Results exported → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE  (no args — prompts user)
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode() -> argparse.Namespace:
    """Fallback interactive prompt when no CLI args are provided."""
    print(BANNER)
    print("No arguments provided — running in interactive mode.\n")

    resume_input = input("📂  Enter path to PDF file or folder of PDFs: ").strip()
    if not resume_input:
        print("❌  No path entered. Exiting.")
        sys.exit(1)

    print("\nAvailable HR requirement presets:")
    for key, req in ROLE_MAP.items():
        print(f"  [{key}]  {req['job_role']}")
    print("  [custom]  Enter your own job role details manually")

    role_choice = input("\n🎯  Choose preset (web / dl / custom): ").strip().lower()

    output_path = input("\n💾  Save results to JSON? (enter filename or leave blank): ").strip()

    return argparse.Namespace(
        resumes=resume_input,
        role=role_choice if role_choice in ROLE_MAP else None,
        custom=role_choice == "custom",
        output=output_path or None,
    )


def build_custom_requirements() -> dict:
    """Prompt user to enter custom HR requirements interactively."""
    print("\n📝  Enter custom HR requirements:")
    job_role    = input("   Job role title       : ").strip()
    description = input("   Role description     : ").strip()
    req_raw     = input("   Required skills (comma-separated): ").strip()
    pref_raw    = input("   Preferred skills (comma-separated): ").strip()
    freshers    = input("   Freshers allowed? (y/n): ").strip().lower() == "y"
    field       = input("   Education field required: ").strip()

    return {
        "job_role":        job_role,
        "role_description": description,
        "required_skills": [s.strip() for s in req_raw.split(",") if s.strip()],
        "preferred_skills": [s.strip() for s in pref_raw.split(",") if s.strip()],
        "experience_required": {"freshers_allowed": freshers},
        "education_required":  {"field": field},
        "projects_required":   {"min_count": 1, "description": ""},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Parse CLI args ────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        parser_cli = argparse.ArgumentParser(
            description="ATS Resume Screening System",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser_cli.add_argument(
            "--resumes", "-r", required=True,
            help="Path to a PDF file or a directory containing PDF resumes",
        )
        parser_cli.add_argument(
            "--role", default="web",
            choices=list(ROLE_MAP.keys()) + ["custom"],
            help="HR requirement preset to use (default: web)",
        )
        parser_cli.add_argument(
            "--output", "-o", default=None,
            help="Optional: path to export results as JSON (e.g. results.json)",
        )
        args = parser_cli.parse_args()
        args.custom = args.role == "custom"
    else:
        args = interactive_mode()

    print(BANNER)

    # ── Collect PDF paths ─────────────────────────────────────────────────────
    pdf_paths = collect_pdfs(args.resumes)
    if not pdf_paths:
        print("❌  No PDFs to process. Exiting.")
        sys.exit(1)

    print(f"📄  Found {len(pdf_paths)} PDF(s):\n" + "\n".join(f"     {p}" for p in pdf_paths))

    # ── Determine HR requirements ─────────────────────────────────────────────
    if getattr(args, "custom", False):
        hr_req = build_custom_requirements()
    else:
        hr_req = ROLE_MAP.get(args.role, WEB_DEV_REQUIREMENTS)
    print(f"\n🎯  Job role: {hr_req['job_role']}")

    # ── Initialise pipeline ───────────────────────────────────────────────────
    print("\n⏳  Initialising pipeline (this may take a moment to load the model)...")
    resume_parser = ResumeParser()
    resume_matcher   = ResumeMatcher(resume_parser.model)
    resume_clusterer = ResumeClusterer(resume_parser.model)

    # ── Parse resumes ─────────────────────────────────────────────────────────
    print(f"\n⚙️   Parsing {len(pdf_paths)} resume(s)...\n")
    resume_db = resume_parser.process_many(pdf_paths)
    print(f"\n✅  Parsed {len(resume_db)} resume(s).")

    # ── ATS matching ──────────────────────────────────────────────────────────
    resume_matcher.set_requirements(hr_req)
    results = resume_matcher.match_all(resume_db)
    print_ats_results(results)

    # ── Clustering ────────────────────────────────────────────────────────────
    clusters = resume_clusterer.cluster(resume_db)
    resume_clusterer.print_summary(clusters)

    # ── Export ────────────────────────────────────────────────────────────────
    if args.output:
        # Convert clusters to JSON-serialisable format (remove numpy arrays etc.)
        clusters_serialisable = {
            domain: [
                {k: v for k, v in m.items()}
                for m in members
            ]
            for domain, members in clusters.items()
        }
        export_json(resume_db, results, clusters_serialisable, hr_req, args.output)

    print("\n✅  Done!\n")


if __name__ == "__main__":
    main()
