import spacy
from pathlib import Path

# -----------------------------------
# LOAD SPACY MODEL
# -----------------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------------
# SKILL KEYWORDS
# -----------------------------------
SKILL_KEYWORDS = [

    # Programming
    "python",
    "java",
    "c++",

    # Core CS
    "operating system",
    "dbms",
    "networking",
    "data structures",
    "algorithms",

    # AI/ML
    "machine learning",
    "deep learning",

    # Web/Backend
    "sql",
    "rest",
    "api",
    "fastapi",
    "django"
]

# -----------------------------------
# PROJECT ROOT
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------------------
# EXTRACT SKILLS
# -----------------------------------
def extract_skills(jd_text: str, resume_text: str):

    jd_text = jd_text.lower()

    resume_text = resume_text.lower()

    jd_skills = set()

    resume_skills = set()

    # JD skills
    for skill in SKILL_KEYWORDS:

        if skill in jd_text:
            jd_skills.add(skill)

    # Resume skills
    for skill in SKILL_KEYWORDS:

        if skill in resume_text:
            resume_skills.add(skill)

    # MATCHED SKILLS ONLY
    matched_skills = list(
        jd_skills & resume_skills
    )

    return matched_skills


# -----------------------------------
# READ TEXT FILE
# -----------------------------------
def read_text_file(relative_path):

    file_path = BASE_DIR / relative_path

    return file_path.read_text(
        encoding="utf-8"
    )


# -----------------------------------
# TEST RUN
# -----------------------------------
if __name__ == "__main__":

    jd_text = read_text_file(
        "data/jd/sample_jd.txt"
    )

    resume_text = read_text_file(
        "data/resumes/student1.txt"
    )

    matched_skills = extract_skills(
        jd_text,
        resume_text
    )

    print("\nMatched Skills:\n")

    print(matched_skills)