import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# -----------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# PROJECT ROOT
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------
# LOAD EXPECTED CONCEPTS
# -----------------------------------
with open(
    BASE_DIR / "data/questions/expected_concepts.json",
    "r",
    encoding="utf-8"
) as f:
    EXPECTED_CONCEPTS = json.load(f)

# -----------------------------------
# FILLER WORDS
# -----------------------------------
FILLER_WORDS = [
    "uh",
    "um",
    "hmm",
    "like",
    "you know"
]


# -----------------------------------
# CONCEPT COVERAGE SCORE
# -----------------------------------
def concept_coverage_score(answer, question):

    expected = EXPECTED_CONCEPTS.get(question, [])

    answer_lower = answer.lower()

    words = set(re.findall(r"\b\w+\b", answer_lower))

    matched = []

    for concept in expected:

        concept_words = concept.lower().split()

        if all(word in words for word in concept_words):
            matched.append(concept)

    coverage = len(matched) / len(expected) if expected else 0

    return coverage, matched


# -----------------------------------
# SEMANTIC RELEVANCE SCORE
# -----------------------------------
def relevance_score(answer, question):

    expected = EXPECTED_CONCEPTS.get(question, [])

    expected_text = " ".join(expected)

    emb_answer = model.encode(
        answer,
        convert_to_tensor=True
    )

    emb_expected = model.encode(
        expected_text,
        convert_to_tensor=True
    )

    score = util.cos_sim(
        emb_answer,
        emb_expected
    ).item()

    return max(score, 0)


# -----------------------------------
# FLUENCY SCORE
# -----------------------------------
def fluency_score(answer):

    words = answer.split()

    word_count = len(words)

    # LENGTH SCORE
    if word_count < 15:
        length_score = 0.4

    elif word_count <= 120:
        length_score = 1.0

    else:
        length_score = 0.7

    # FILLER PENALTY
    filler_count = sum(
        answer.lower().count(f)
        for f in FILLER_WORDS
    )

    filler_penalty = min(
        filler_count * 0.05,
        0.3
    )

    # SENTENCE STRUCTURE
    sentences = re.split(r"[.!?]", answer)

    sentences = [
        s for s in sentences
        if s.strip()
    ]

    sentence_score = 1.0 if len(sentences) >= 2 else 0.6

    fluency = max(
        length_score * sentence_score - filler_penalty,
        0
    )

    return fluency


# -----------------------------------
# FINAL ANSWER EVALUATION
# -----------------------------------
def evaluate_answer(question, answer):

    coverage, matched_concepts = concept_coverage_score(
        answer,
        question
    )

    relevance = relevance_score(
        answer,
        question
    )

    fluency = fluency_score(answer)

    # FINAL WEIGHTED SCORE
    final_score = (
        0.4 * coverage +
        0.3 * relevance +
        0.3 * fluency
    ) * 10

    # MISSING CONCEPTS
    missing_concepts = list(
        set(EXPECTED_CONCEPTS.get(question, []))
        - set(matched_concepts)
    )

    # FEEDBACK
    feedback = {
        "matched_concepts": matched_concepts,
        "missing_concepts": missing_concepts,
        "coverage_score": round(coverage * 10, 2),
        "relevance_score": round(relevance * 10, 2),
        "fluency_score": round(fluency * 10, 2),
        "final_score": round(final_score, 2)
    }

    return feedback


# -----------------------------------
# TEST RUN
# -----------------------------------
if __name__ == "__main__":

    sample_question = "Explain stack vs queue."

    sample_answer = """
    Stack follows LIFO whereas queue follows FIFO.
    Stack insertion and deletion happen at one end.
    Queue operations happen at both ends.
    """

    result = evaluate_answer(
        sample_question,
        sample_answer
    )

    print("\nEvaluation Result:\n")

    for key, value in result.items():
        print(f"{key}: {value}")