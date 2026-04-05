from pathlib import Path

from ml.resume_parser import extract_skills
from ml.question_selector import select_questions
from ml.answer_evaluator import evaluate_answer
from ml.ranking import rank_candidates
from speech.speech_to_text import transcribe_audio

BASE_DIR = Path(__file__).resolve().parent


def process_jd_and_resume(jd_text, resume_text):
    """
    Extract skills and select interview questions
    """
    questions = select_questions(jd_text, resume_text)
    return questions


def evaluate_audio_answer(question, audio_filename):
    """
    Full pipeline: audio -> text -> evaluation
    """
    answer_text = transcribe_audio(audio_filename)
    evaluation = evaluate_answer(question, answer_text)
    return answer_text, evaluation


def rank_multiple_candidates(question, candidate_audio_files):
    """
    candidate_audio_files = {
        "Student A": "a.wav",
        "Student B": "b.wav"
    }
    """
    candidate_answers = {}

    for candidate, audio_file in candidate_audio_files.items():
        text = transcribe_audio(audio_file)
        candidate_answers[candidate] = text

    ranked = rank_candidates(question, candidate_answers)
    return ranked
