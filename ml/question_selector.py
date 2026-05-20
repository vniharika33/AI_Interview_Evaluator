from ml.llm.generate import generate_text


# -----------------------------
# GENERATE QUESTIONS FROM TOPICS
# -----------------------------
def generate_interview_questions(topics, questions_per_topic=2):

    all_questions = []

    for topic in topics:

        prompt = f"""
Generate {questions_per_topic} interview questions on {topic}.

Questions:
"""

        output = generate_text(prompt, max_tokens=60)

        all_questions.append({
            "topic": topic,
            "questions": output
        })

    return all_questions


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":

    matched_topics = [
        "Operating System",
        "DBMS",
        "Networking"
    ]

    questions = generate_interview_questions(matched_topics)

    for item in questions:
        print("\n====================")
        print("TOPIC:", item["topic"])
        print("====================")
        print(item["questions"])