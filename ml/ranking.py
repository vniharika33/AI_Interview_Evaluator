from ml.answer_evaluator import evaluate_answer


def rank_candidates(question, candidate_answers):
    """
    candidate_answers = {
        "student_name": "answer text",
        ...
    }
    """
    results = []

    for candidate, answer in candidate_answers.items():
        evaluation = evaluate_answer(question, answer)
        results.append({
            "candidate": candidate,
            "score": evaluation["final_score"],
            "details": evaluation
        })

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    question = "Explain stack vs queue."

    # Simulated answers (later these will come from voice → text)
    candidate_answers = {
        "Student_A": "A stack follows LIFO while queue follows FIFO.",
        "Student_B": "Stack and queue are data structures used in programs.",
        "Student_C": (
            "Stack works on LIFO principle using push and pop. "
            "Queue works on FIFO using enqueue and dequeue."
        )
    }

    ranked = rank_candidates(question, candidate_answers)

    print("\nFinal Candidate Ranking:\n")
    for idx, r in enumerate(ranked, 1):
        print(f"{idx}. {r['candidate']} - Score: {r['score']}")
