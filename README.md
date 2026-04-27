# 🎯 AI Interview Integrity Agent (Fraud Detection System)

An AI-powered interview monitoring and fraud detection system built using **Streamlit**, **Whisper**, and **NLP models** to evaluate candidate responses in real time.

---

## 🚀 Overview

This project was developed during an AI Hackathon to tackle a critical problem:

> ❗ How do we ensure authenticity and integrity in remote interviews?

The system acts as an **AI interviewer + integrity checker**, analyzing:

* Response relevance
* Linguistic complexity
* Behavioral consistency
* Suspicious latency patterns

---

## 🧠 Key Features

### 🎙️ AI Interview System

* Dynamic or pre-configured question bank
* Behavioral & technical interview tracks
* Real-time voice recording and transcription

### 🧾 Fraud Detection Engine

A custom-built **Integrity Sub-Agent** evaluates:

* Semantic similarity (is the answer relevant?)
* Vocabulary complexity spikes
* Readability shifts (Flesch-Kincaid score)
* Response latency anomalies

### 🤖 AI Models Used

* **Whisper** → Speech-to-text
* **Sentence Transformers** → Semantic embeddings
* **LiteLLM (Groq)** → Dynamic question generation

### 📊 Recruiter Dashboard

* View candidate responses
* Fraud flags with confidence scores
* Latency tracking
* Explainable AI reasoning

### 🎥 Live Monitoring

* Webcam integration
* Audio input capture
* Real-time interaction

---

## ⚙️ How It Works

1. Candidate selects:

   * Level (Fresher / Intermediate / Senior)
   * Track (Behavioral / Technical)

2. System:

   * Asks questions (static or AI-generated)
   * Records voice responses
   * Converts speech → text

3. Fraud Detection:

   * Compares response with previous answers
   * Checks semantic relevance
   * Detects unnatural jumps in complexity or delay

4. Output:

   * ✅ Clear
   * ⚠️ Flagged (with reason & confidence)

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **ML/NLP:** Whisper, Sentence Transformers
* **LLM Integration:** LiteLLM (Groq API)
* **Data Processing:** NumPy, SciPy
* **Visualization:** Matplotlib
* **Text Analysis:** TextStat
* **Speech:** gTTS

---

## 📂 Project Structure

ai-interview-integrity-agent/
│
├── app.py                # Main Streamlit application
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
├── assets/               # (Optional) Images / demo GIFs
└── .gitignore            # Ignore unnecessary files

---

## ▶️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/Ishaaaann/ai-interview-integrity-agent.git
cd ai-interview-integrity-agent

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the application

streamlit run app.py

---

## 🔑 API Setup (Optional)

To enable dynamic AI-generated questions:

1. Get a Groq API key
2. Enter it in the sidebar inside the app

The app will still work without it using the pre-configured question bank.

---

## 🧪 Example Fraud Signals

| Scenario                     | Detection     |
| ---------------------------- | ------------- |
| Off-topic answer             | ❌ Flagged     |
| Sudden high complexity       | ⚠️ Suspicious |
| Long unnatural delay         | ⚠️ Flagged    |
| Consistent natural responses | ✅ Clear       |

---

## 🏆 Hackathon Project

Built during an AI Hackathon focusing on:

* Real-world interview integrity
* Explainable AI
* Human-AI interaction

---

## 👨‍💻 Author

**Ishaan Sharma**
Computer Engineering Student | AI + Motorsport Enthusiast

---

## 📜 License

MIT License
