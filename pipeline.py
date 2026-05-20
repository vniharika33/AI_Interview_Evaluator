from pathlib import Path

from ml.resume_parser import extract_skills
from ml.question_selector import generate_interview_questions
from ml.answer_evaluator import evaluate_answer

from speech.speech_to_text import (
    record_audio,
    transcribe_audio
)

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------
# PROCESS JD + RESUME
# ---------------------------------------------------
def process_jd_and_resume(jd_text, resume_text):

    """
    Extract skills/topics from JD + Resume
    """

    topics = extract_skills(
        jd_text,
        resume_text
    )

    return topics


# ---------------------------------------------------
# GENERATE QUESTIONS USING LLM
# ---------------------------------------------------
def generate_questions_from_topics(topics):

    """
    Generate interview questions using custom LLM
    """

    questions = generate_interview_questions(topics)

    return questions


# ---------------------------------------------------
# EVALUATE SINGLE AUDIO ANSWER
# ---------------------------------------------------
def evaluate_audio_answer(question, audio_filename):

    """
    Full pipeline:
    audio -> text -> evaluation
    """

    answer_text = transcribe_audio(audio_filename)

    evaluation = evaluate_answer(
        question,
        answer_text
    )

    return answer_text, evaluation


# ---------------------------------------------------
# COMPLETE MOCK INTERVIEW PIPELINE
# ---------------------------------------------------
def run_mock_interview(jd_text, resume_text):

    # STEP 1: Extract topics
    topics = process_jd_and_resume(
        jd_text,
        resume_text
    )

    print("\nDetected Topics:")
    print(topics)

    # STEP 2: Generate questions
    generated_questions = generate_questions_from_topics(
        topics
    )

    all_results = []

    # STEP 3: Ask questions
    for item in generated_questions:

        topic = item["topic"]

        print("\n===================================")
        print(f"TOPIC: {topic}")
        print("===================================")

        question_block = item["questions"]

        print(question_block)

        # OPTIONAL:
        # split generated text into lines/questions
        question_lines = [
            q.strip()
            for q in question_block.split("\n")
            if q.strip()
        ]

        for idx, question in enumerate(question_lines):

            print(f"\nQuestion {idx+1}:")
            print(question)

            # RECORD AUDIO
            audio_file = (
                BASE_DIR /
                f"audio/answers/answer_{idx}.wav"
            )

            record_audio(str(audio_file))

            # TRANSCRIBE
            answer_text = transcribe_audio(
                f"answer_{idx}.wav"
            )

            print("\nCandidate Answer:")
            print(answer_text)

            # EVALUATE
            result = evaluate_answer(
                question,
                answer_text
            )

            print("\nEvaluation Result:")
            print(result)

            all_results.append(result)

    # STEP 4: FINAL SCORE
    if all_results:

        avg_score = sum(
            r["final_score"]
            for r in all_results
        ) / len(all_results)

    else:
        avg_score = 0

    print("\n===================================")
    print("FINAL INTERVIEW SCORE")
    print("===================================")

    print(f"Final Score: {avg_score:.2f}/10")

    return {
        "topics": topics,
        "results": all_results,
        "final_score": round(avg_score, 2)
    }


# ---------------------------------------------------
# TEST RUN
# ---------------------------------------------------
if __name__ == "__main__":

    sample_jd = """
    Looking for candidates skilled in
    Operating System, DBMS, Networking,
    and Data Structures.
    """

    sample_resume = """
    Student has worked on DBMS project,
    networking lab assignments,
    and operating system concepts.
    """

    run_mock_interview(
        sample_jd,
        sample_resume
    )