"""
ats_system
----------
ATS Resume Screening System

Modules:
    parser      — PDF extraction, spaCy NER, embedding generation
    matcher     — Section-by-section ATS scoring with rapidfuzz
    clusterer   — Zero-shot domain clustering
    requirements — Example HR requirement configs

Quick start:
    from ats_system.parser    import ResumeParser
    from ats_system.matcher   import ResumeMatcher
    from ats_system.clusterer import ResumeClusterer

    parser    = ResumeParser()
    matcher   = ResumeMatcher(parser.model)
    clusterer = ResumeClusterer(parser.model)

    resume_db = parser.process_many(["cv1.pdf", "cv2.pdf"])
    matcher.set_requirements(HR_REQUIREMENTS)
    results  = matcher.match_all(resume_db)
    clusters = clusterer.cluster(resume_db)
"""

from .parser    import ResumeParser
from .matcher   import ResumeMatcher
from .clusterer import ResumeClusterer

__all__ = ["ResumeParser", "ResumeMatcher", "ResumeClusterer"]
__version__ = "3.0.0"
