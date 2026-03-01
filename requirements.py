"""
requirements.py
---------------
Example HR requirements configurations.
Copy and modify these for your own use case.

Usage:
    from requirements import WEB_DEV_REQUIREMENTS, DL_REQUIREMENTS
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE: Junior Web / Full Stack Developer
# ─────────────────────────────────────────────────────────────────────────────

WEB_DEV_REQUIREMENTS = {
    "job_role": "Junior Web / Full Stack Developer",
    "experience": "Fresher / Student (Project-based experience)",

    "role_description": """
        We are looking for a Junior Web Developer with hands-on experience in
        building frontend and full stack web applications. The candidate should
        be skilled in HTML, CSS, JavaScript, and backend technologies like Java
        and Spring Boot, and should be able to collaborate in a team to deliver
        responsive and user-friendly web solutions.
    """,

    "required_skills": [
        "HTML", "CSS", "JavaScript", "Java", "Spring Boot",
        "MySQL", "Git", "GitHub", "Postman", "Figma",
    ],

    "preferred_skills": [
        "React.js", "REST API", "responsive design",
        "UI/UX design basics", "VS Code", "IntelliJ",
    ],

    "projects_required": {
        "min_count": 2,
        "description": "Course Registration System and Responsive Music Web Application",
    },

    "education_required": {
        "degree": "Bachelor of Computer Science and Engineering",
        "field": "Computer Science or related field",
    },

    "experience_required": {
        "freshers_allowed": True,
        "description": "Project-based experience acceptable for freshers.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE: Deep Learning / AI Engineer
# ─────────────────────────────────────────────────────────────────────────────

DL_REQUIREMENTS = {
    "job_role": "Deep Learning / AI Engineer",
    "experience": "2+ years or strong project portfolio",

    "role_description": """
        We are looking for a Deep Learning Engineer with strong hands-on experience
        in building and deploying neural network models for computer vision or NLP tasks.
        The candidate should be comfortable with PyTorch/TensorFlow, model training pipelines,
        and deploying models to production environments.
    """,

    "required_skills": [
        "Python", "PyTorch", "TensorFlow", "deep learning",
        "neural networks", "model training", "GPU", "CUDA",
    ],

    "preferred_skills": [
        "Hugging Face", "BERT", "GPT", "YOLO", "OpenCV",
        "MLflow", "Docker", "AWS", "ONNX",
    ],

    "projects_required": {
        "min_count": 2,
        "description": "End-to-end deep learning projects with training and deployment.",
    },

    "education_required": {
        "degree": "Bachelor or Master in Computer Science, AI, or related",
        "field": "Computer Science, AI, Machine Learning, or Data Science",
    },

    "experience_required": {
        "freshers_allowed": False,
        "description": "Minimum 2 years of industry or research experience required.",
    },
}
