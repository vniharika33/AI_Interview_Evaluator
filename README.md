# AI-Interview-Evaluator

An end-to-end **GenAI-powered AI Interview Evaluation System** that dynamically generates interview questions based on a candidate’s Resume and Job Description (JD), records voice responses, converts speech to text using Whisper, evaluates answers using semantic NLP techniques, and generates an AI-based interview score with feedback.

---

#  Features

##  Resume & JD Skill Matching
- Extracts overlapping technical skills from:
  - Resume
  - Job Description
- Uses NLP-based keyword matching

---

##  Custom Transformer-based Question Generation
- Built a custom GPT-style Transformer model from scratch
- Generates interview questions dynamically based on detected topics
- Includes:
  - Tokenization
  - Embeddings
  - Multi-Head Self Attention
  - Transformer Blocks
  - Text Generation

---

##  Voice-based Interview Answers
- Candidate uploads or records audio answers
- Uses OpenAI Whisper for:
  - Speech-to-text conversion
  - Real-time answer transcription

---

##  Semantic AI Evaluation
- Evaluates candidate answers using:
  - Concept coverage
  - Semantic similarity
  - Fluency analysis
- Uses SentenceTransformers embeddings

---

##  Final AI Interview Score
- Generates:
  - Technical score
  - Relevance score
  - Fluency score
  - Final interview score
  - Missing concepts feedback

---

##  Streamlit Frontend
- Interactive web interface
- Upload Resume/JD
- Upload voice answers
- View AI evaluation results instantly

---

#  System Architecture

```
![Uploading ChatGPT Image May 21, 2026, 12_01_04 AM.png…]()


---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Frontend | Streamlit |
| NLP | SpaCy |
| Speech Recognition | OpenAI Whisper |
| Deep Learning | PyTorch |
| Embeddings | SentenceTransformers |
| Transformer Model | Custom GPT-style LLM |
| PDF Parsing | PyPDF2 |

---

# 📂 Project Structure

```text
GenAI-Interview-Evaluator/
│
├── ml/
│   ├── llm/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── generate.py
│   │   └── data.py
│   │
│   ├── answer_evaluator.py
│   ├── question_selector.py
│   ├── resume_parser.py
│   └── ranking.py
│
├── speech/
│   └── speech_to_text.py
│
├── data/
│   ├── jd/
│   ├── resumes/
│   └── questions/
│
├── audio/
│   └── answers/
│
├── app.py
├── pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/vniharika33/GenAI-Interview-Evaluator.git
```

---

## 2️⃣ Move into Project Folder

```bash
cd GenAI-Interview-Evaluator
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Install SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

#  Evaluation Metrics

The system evaluates answers using:

##  Concept Coverage
Checks whether important technical concepts are present in the answer.

---

##  Semantic Relevance
Uses sentence embeddings to compare answer meaning with expected concepts.

---

##  Fluency Analysis
Evaluates:
- Answer length
- Sentence structure
- Filler word usage

---

#  Example Workflow

```text
Resume + JD
      ↓
Skill Matching
      ↓
AI Question Generation
      ↓
Candidate Voice Response
      ↓
Whisper Speech-to-Text
      ↓
Semantic AI Evaluation
      ↓
Final Interview Score + Feedback
```

---

#Dashboard

<img width="383" height="491" alt="image" src="https://github.com/user-attachments/assets/5c6c4d26-d629-4f72-956a-9075257e7c52" />


---

#  Future Improvements

- Fine-tuned LLM for better question quality
- Real-time voice interview mode
- Facial emotion analysis
- AI-based behavioral interview evaluation
- Cloud deployment
- Multi-language support

---


