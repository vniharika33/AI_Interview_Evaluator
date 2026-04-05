import streamlit as st
from PyPDF2 import PdfReader
from pathlib import Path

from pipeline import process_jd_and_resume, evaluate_audio_answer

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Interview Evaluator",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio" / "answers"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- HELPERS ----------------
def read_uploaded_file(uploaded_file):
    """
    Reads text from TXT or PDF file
    """
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return uploaded_file.read().decode("utf-8")


# ---------------- UI ----------------
st.title("🎯 AI Interview Evaluation System")
st.markdown(
    """
    This system helps recruiters evaluate candidates using:
    - Resume + JD based question generation
    - Voice-based answers
    - AI scoring (technical + communication)
    """
)

st.divider()

# ---------------- FILE UPLOAD ----------------
st.subheader("📄 Upload Job Description & Resume")

jd_file = st.file_uploader(
    "Upload Job Description (PDF or TXT)",
    type=["pdf", "txt"]
)

resume_file = st.file_uploader(
    "Upload Resume (PDF or TXT)",
    type=["pdf", "txt"]
)

if jd_file and resume_file:
    jd_text = read_uploaded_file(jd_file)
    resume_text = read_uploaded_file(resume_file)

    st.success("Files uploaded successfully")

    # ---------------- QUESTION SELECTION ----------------
    questions = process_jd_and_resume(jd_text, resume_text)

    if not questions:
        st.warning("No matching questions found for this JD and Resume.")
    else:
        st.subheader("🧠 Generated Interview Questions")

        selected_question = st.selectbox(
            "Select a question to answer",
            questions
        )

        st.divider()

        # ---------------- AUDIO UPLOAD ----------------
        st.subheader("🎤 Upload Audio Answer")

        audio_file = st.file_uploader(
            "Upload your answer (.wav format)",
            type=["wav"]
        )

        if audio_file and selected_question:
            temp_audio_path = AUDIO_DIR / "temp.wav"

            with open(temp_audio_path, "wb") as f:
                f.write(audio_file.read())

            with st.spinner("Evaluating answer..."):
                answer_text, evaluation = evaluate_audio_answer(
                    selected_question,
                    "temp.wav"
                )

            # ---------------- RESULTS ----------------
            st.subheader("📝 Transcribed Answer")
            st.write(answer_text)

            st.subheader("📊 Evaluation Result")
            st.json(evaluation)

            st.success("Evaluation completed successfully")
