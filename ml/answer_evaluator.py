import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load expected concepts
with open(BASE_DIR / "data/questions/expected_concepts.json", "r", encoding="utf-8") as f:
    EXPECTED_CONCEPTS = json.load(f)

FILLER_WORDS = ["uh", "um", "hmm", "like", "you know"]


def concept_coverage_score(answer, question):
    expected = EXPECTED_CONCEPTS.get(question, [])
    answer_lower = answer.lower()

    # tokenize answer into words
    words = set(re.findall(r"\b\w+\b", answer_lower))

    matched = []
    for concept in expected:
        concept_words = concept.split()
        if all(word in words for word in concept_words):
            matched.append(concept)

    coverage = len(matched) / len(expected) if expected else 0
    return coverage, matched


def relevance_score(answer, question):
    emb_answer = model.encode(answer, convert_to_tensor=True)
    emb_question = model.encode(question, convert_to_tensor=True)
    score = util.cos_sim(emb_answer, emb_question).item()
    return max(score, 0)


def fluency_score(answer):
    words = answer.split()
    word_count = len(words)

    # Length-based fluency
    if word_count < 20:
        length_score = 0.4
    elif word_count <= 120:
        length_score = 1.0
    else:
        length_score = 0.7

    # Filler words
    filler_count = sum(answer.lower().count(f) for f in FILLER_WORDS)
    filler_penalty = min(filler_count * 0.05, 0.3)

    # Sentence structure
    sentences = re.split(r"[.!?]", answer)
    sentences = [s for s in sentences if s.strip()]
    sentence_score = 1.0 if len(sentences) >= 2 else 0.6

    fluency = max(length_score * sentence_score - filler_penalty, 0)
    return fluency


def evaluate_answer(question, answer):
    coverage, matched_concepts = concept_coverage_score(answer, question)
    relevance = relevance_score(answer, question)
    fluency = fluency_score(answer)

    final_score = (
        0.4 * coverage +
        0.25 * relevance +
        0.35 * fluency
    ) * 10

    feedback = {
        "matched_concepts": matched_concepts,
        "missing_concepts": list(
            set(EXPECTED_CONCEPTS.get(question, [])) - set(matched_concepts)
        ),
        "fluency_score": round(fluency * 10, 2),
        "relevance_score": round(relevance * 10, 2),
        "final_score": round(final_score, 2)
    }

    return feedback


if __name__ == "__main__":
    sample_question = "Explain stack vs queue."
    sample_answer = (
        "stack follows LIFO queue follows FIFO there are various other data structures like graphs, trees, binary"
    )

    result = evaluate_answer(sample_question, sample_answer)
    print(result)
