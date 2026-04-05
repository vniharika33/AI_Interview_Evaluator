import spacy
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Skill keywords (expandable later)
SKILL_KEYWORDS = [
    "python", "java", "c++", "machine learning",
    "deep learning", "data structures", "algorithms",
    "sql", "rest", "api", "fastapi", "django"
]

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

def extract_skills(text: str):
    text = text.lower()
    found_skills = set()

    # Phrase-level matching (for multi-word skills)
    for skill in SKILL_KEYWORDS:
        if skill in text:
            found_skills.add(skill)

    return list(found_skills)


def read_text_file(relative_path):
    file_path = BASE_DIR / relative_path
    return file_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    jd_text = read_text_file("data/jd/sample_jd.txt")
    resume_text = read_text_file("data/resumes/student1.txt")

    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    print("JD Skills:", jd_skills)
    print("Resume Skills:", resume_skills)
    print("Matched Skills:", list(set(jd_skills) & set(resume_skills)))
