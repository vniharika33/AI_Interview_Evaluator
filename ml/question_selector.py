import json
from pathlib import Path
from ml.resume_parser import extract_skills



# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

def load_questions():
    path = BASE_DIR / "data/questions/questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def select_questions(jd_text, resume_text, max_questions=5):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched_skills = list(set(jd_skills) & set(resume_skills))
    question_bank = load_questions()

    selected = []

    for skill in matched_skills:
        if skill in question_bank:
            selected.extend(question_bank[skill])

    return selected[:max_questions]

if __name__ == "__main__":
    jd_path = BASE_DIR / "data/jd/sample_jd.txt"
    resume_path = BASE_DIR / "data/resumes/student1.txt"

    jd_text = jd_path.read_text(encoding="utf-8")
    resume_text = resume_path.read_text(encoding="utf-8")

    questions = select_questions(jd_text, resume_text)

    print("Selected Interview Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
