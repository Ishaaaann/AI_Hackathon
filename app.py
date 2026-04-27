import streamlit as st
import streamlit.components.v1 as components

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Interview Integrity Agent",
    layout="wide"
)

import whisper
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import textstat
import matplotlib.pyplot as plt
import tempfile
import os
import re
import time
import base64
import wave
from gtts import gTTS
from litellm import completion

# =========================
# 1. DATA & MODELS
# =========================
DEFAULT_QUESTION_BANK = {
    "fresher": {
        "Behavioral": [
            "Tell me about yourself and your background.",
            "Describe a time you had to learn a new framework or technology extremely quickly.",
            "What is your biggest weakness, and how do you manage it?",
            "Tell me about a time you made a mistake on a project. How did you fix it?",
            "Describe a situation where you disagreed with a teammate. How did you resolve it?",
            "Tell me about your proudest technical achievement so far.",
            "How do you prioritize tasks when you have multiple tight deadlines?",
            "Why do you want to transition into this specific role?"
        ],
        "Technical": [
            "Explain what an API is in simple terms.",
            "What are the core differences between C++ and a memory-safe language like Rust?",
            "Explain the basics of Object-Oriented Programming.",
            "How does Git version control work under the hood?",
            "Walk me through what happens when you type a URL into a browser.",
            "Describe how you would approach reverse-engineering a simple Android APK.",
            "What is the difference between a microcontroller (like an ESP32) and a microprocessor?",
            "How do you typically debug a program when it crashes unexpectedly?"
        ]
    },
    "intermediate": {
        "Behavioral": [
            "Tell me about a time you took the lead on a project without a formal title.",
            "Describe a time you had to build a platform or app for a large-scale event. What went wrong?",
            "How do you balance writing perfect code versus meeting project deadlines?",
            "Tell me about a time you received difficult constructive feedback.",
            "Describe a situation where you had to explain a complex technical concept to a non-technical person.",
            "How do you handle scope creep on a project?",
            "Tell me about a time you optimized a highly inefficient process.",
            "Describe your ideal team culture."
        ],
        "Technical": [
            "Explain how the Rust borrow checker prevents data races.",
            "Walk me through your process for customizing and managing a Linux environment.",
            "How does PID control logic work in the context of robotics or hardware integration?",
            "Explain the difference between REST and gRPC.",
            "Describe how you use probability modeling and statistical tools in data analysis.",
            "What are the trade-offs of using a tiling window manager over a standard desktop environment?",
            "Explain how indexing works in a relational database.",
            "Walk me through the lifecycle of a React application."
        ]
    },
    "senior": {
        "Behavioral": [
            "Tell me about a time you led a team through a difficult technical pivot.",
            "How do you approach mentoring junior developers?",
            "Describe a time you made a wrong architectural decision. How did you rectify it?",
            "How do you balance technical debt with the need to deliver new features?",
            "Tell me about a time you had to push back on unrealistic stakeholder demands.",
            "Describe your process for conducting code reviews.",
            "How do you ensure your engineering team stays motivated during a long sprint?",
            "Tell me about a major conflict you resolved between the engineering and product teams."
        ],
        "Technical": [
            "Design a scalable microservices architecture. What are the bottlenecks?",
            "Explain the complexities of cross-compiling code for niche architectures like IBM z/OS.",
            "How do you handle race conditions in high-concurrency environments?",
            "Explain the CAP theorem and how it influences your database choices.",
            "Walk me through how you would write a custom Linux compositor (e.g., using Wayland).",
            "Discuss advanced Smali patching techniques for modifying compiled applications.",
            "How do you optimize a data science pipeline that is taking too long to train?",
            "Explain kernel-level threads versus user-level threads."
        ]
    }
}


DOMAIN_KEYWORDS = {
    "software_engineering": { "api", "architecture", "c++", "rust", "assembly", "linux", 
        "kernel", "threads", "memory", "database", "scalable", "qemu",
        "compositor", "apk", "smali", "microcontroller", "pid", "pointers"}
}

@st.cache_resource
def load_models():
    return whisper.load_model("base"), SentenceTransformer('all-MiniLM-L6-v2')

stt_model, embed_model = load_models()

# =========================
# 2. SESSION STATE
# =========================
state_defaults = {
    "history": [],
    "interview_active": True,
    "has_started": False,
    "track_selected": False,
    "indices": {"Behavioral": 0, "Technical": 0},
    "answered_current": False,
    "q_start_time": 0,
    "custom_questions": {"Behavioral": [], "Technical": []},
    "current_dynamic_q": "",
    "view_role": "Candidate",
    "active_track": "Behavioral",
    "cam_on": True,
    "mic_on": True
}

for key, val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================
# 3. AGENT LOGIC (WITH RELEVANCY)
# =========================
class IntegritySubAgent:
    def __init__(self):
        self.thresholds = {"fresher": 0.5, "intermediate": 0.35, "senior": 0.25}
        self.weights = {
            "technical": {"vocab": 0.3, "readability": 0.3, "semantic": 0.4}, 
            "behavioral": {"vocab": 0.4,"readability": 0.4, "structure": 0.2}
        }

    def clean(self, text): return re.sub(r'[^\w\s]', '', text.lower()).split()

    def extract(self, text, domain):
        words = self.clean(text)
        used = [w for w in words if w in DOMAIN_KEYWORDS.get(domain, set())]
        return {
            "vocab": len(used)/max(len(words), 1),
            "word_len": np.mean([len(w) for w in words]) if words else 0,
            "fk": textstat.flesch_kincaid_grade(text),
            "embedding": embed_model.encode(text)
        }

    def evaluate(self, text, question_text, q_type, level, domain, history, latency):
        f = self.extract(text, domain)
        rel = [h for h in history if h["q_type"] == q_type]
        word_count = len(self.clean(text))

        # --- SEMANTIC RELEVANCY (The "Pandi Lun" Fix) ---
        q_emb = embed_model.encode(question_text)
        relevancy_dist = cosine(f["embedding"], q_emb)

        if relevancy_dist > 0.82 or word_count < 4:
            return {"flag": True, "confidence": 0.98, "reason": "Review Flag: Answer is off-topic or nonsensical.", "features": f, "latency": round(latency, 2)}

        if len(rel) < 1:
            is_suspicious = (level == "fresher" and (f["fk"] > 12 or f["vocab"] > 0.15)) or (level == "intermediate" and (f["fk"] > 15 or f["vocab"] > 0.20))
            return {"flag": is_suspicious, "confidence": 0.75 if is_suspicious else 0.0, "reason": "Building baseline", "features": f, "latency": round(latency, 2)}

        base_vocab = np.mean([h["features"]["vocab"] for h in rel])
        base_fk = np.mean([h["features"]["fk"] for h in rel])
        dv = abs(f["vocab"] - base_vocab) / max(base_vocab, 0.01)
        df = abs(f["fk"] - base_fk) / max(base_fk, 0.01)

        threshold = self.thresholds.get(level, 0.4)
        score = self.weights["technical"]["vocab"]*min(dv,2) + self.weights["technical"]["readability"]*min(df,2)
        lat_flag = (latency > 12.0 and f["fk"] > 12) or (latency < 1.5 and q_type == "Technical")
        
        return {
            "flag": (score > threshold) or lat_flag, 
            "confidence": round(min(0.99, score/threshold), 2),
            "reason": f"Suspicious Latency" if lat_flag else "Consistent with baseline",
            "features": f, "latency": round(latency, 2)
        }

agent = IntegritySubAgent()

# =========================
# 4. LLM & UI HELPERS
# =========================
def generate_dynamic_question(domain, level, q_type, history, api_key, model_choice):
    if not api_key: return "Please provide an API key for dynamic mode."
    os.environ["GROQ_API_KEY"] = api_key
    history_context = ""
    relevant_history = [h for h in history if h["q_type"] == q_type]
    if relevant_history:
        for h in relevant_history: history_context += f"Q: {h['question']} A: {h['text']}\n"
    
    system_prompt = f"Expert recruiter. Ask ONE {q_type} question for {level} candidate in {domain}. Under 25 words. Return ONLY question."
    try:
        response = completion(model=model_choice, messages=[{"role": "user", "content": system_prompt}], temperature=0.7)
        return response.choices[0].message.content.strip()
    except Exception as e: return f"API ERROR: {str(e)}"

def autoplay_audio(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        with open(tmp.name, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

def render_live_webcam():
    components.html("""<video id="webcam" autoplay muted playsinline style="width: 100%; max-width: 500px; border-radius: 12px; transform: scaleX(-1); background-color: #1e1e1e;"></video>
    <script>navigator.mediaDevices.getUserMedia({ video: true }).then(s => { document.getElementById('webcam').srcObject = s; });</script>""", height=320)

def render_question_box(text):
    st.markdown(f"""<div style="display: flex; justify-content: center; margin-bottom: 25px;"><div style="background-color: white; color: #1E88E5; padding: 25px 40px; border-radius: 24px; font-weight: 800; font-size: 1.5rem; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.15); display: inline-block; max-width: 800px;">{text}</div></div>""", unsafe_allow_html=True)

# =========================
# 5. MAIN ROUTING
# =========================
with st.sidebar:
    st.session_state.view_role = st.radio("Access Level", ["Candidate", "Developer"])
    st.divider()
    level = st.selectbox("Candidate Level", ["fresher","intermediate","senior"])
    domain = st.selectbox("Domain", ["software_engineering"])
    mode = st.radio("AI Mode", ["Pre-configured Bank", "Dynamic LLM"])
    if mode == "Dynamic LLM":
        llm_model = st.selectbox("Model", ["groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile"])
        api_key = st.text_input("API Key", type="password")
        max_q = st.number_input("Max Qs per Track", min_value=1, value=3)
    
    if st.button("Reset Everything", type="primary"):
        st.session_state.clear()
        st.rerun()

# --- CANDIDATE VIEW ---
if st.session_state.view_role == "Candidate":
    st.title("🎙️ Interview Portal")
    
    if not st.session_state.has_started:
        st.markdown("### Step 1: Initial Setup")
        if mode == "Pre-configured Bank":
            col1, col2 = st.columns(2)
            with col1: b_in = st.text_area("Behavioral Qs", "\n".join(DEFAULT_QUESTION_BANK[level]["Behavioral"]))
            with col2: t_in = st.text_area("Technical Qs", "\n".join(DEFAULT_QUESTION_BANK[level]["Technical"]))
        
        if st.button("Proceed to Track Selection ➔", use_container_width=True, type="primary"):
            if mode == "Pre-configured Bank":
                st.session_state.custom_questions["Behavioral"] = [q.strip() for q in b_in.split('\n') if q.strip()]
                st.session_state.custom_questions["Technical"] = [q.strip() for q in t_in.split('\n') if q.strip()]
            st.session_state.has_started = True
            st.rerun()

    elif not st.session_state.track_selected:
        st.markdown("### Step 2: Select Track")
        track = st.selectbox("Focus Track", ["Behavioral", "Technical"])
        if st.button("Start Interview 🚀", use_container_width=True, type="primary"):
            st.session_state.active_track = track
            if mode == "Dynamic LLM":
                with st.spinner("Generating opening question..."):
                    st.session_state.current_dynamic_q = generate_dynamic_question(domain, level, track, [], api_key, llm_model)
            st.session_state.track_selected = True
            st.rerun()

    else:
        q_type = st.session_state.active_track
        idx = st.session_state.indices[q_type]
        total = len(st.session_state.custom_questions[q_type]) if mode == "Pre-configured Bank" else max_q
        
        if idx < total:
            q_text = st.session_state.custom_questions[q_type][idx] if mode == "Pre-configured Bank" else st.session_state.current_dynamic_q
            
            render_question_box(q_text)
            if st.session_state.cam_on: render_live_webcam()
            
            c1, c2, _ = st.columns([1,1,2])
            st.session_state.cam_on = c1.toggle("📷 Camera", value=st.session_state.cam_on)
            st.session_state.mic_on = c2.toggle("🎙️ Mic", value=st.session_state.mic_on)

            if not st.session_state.answered_current:
                if st.session_state.q_start_time == 0:
                    autoplay_audio(q_text)
                    st.session_state.q_start_time = time.time()
                
                audio = st.audio_input("Record Answer", key=f"mic_{idx}")
                if audio:
                    with st.spinner("Analyzing..."):
                        elapsed = time.time() - st.session_state.q_start_time
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio.getvalue()); path = tmp.name
                        with wave.open(path, 'rb') as wf: dur = wf.getnframes() / float(wf.getframerate())
                        text = stt_model.transcribe(path)["text"]
                        os.remove(path)
                        
                        # True Latency calculation
                        true_lat = max(0.0, elapsed - (len(q_text.split())/2.3) - dur)
                        res = agent.evaluate(text, q_text, q_type, level, domain, st.session_state.history, true_lat)
                        
                        st.session_state.history.append({"question": q_text, "text": text, "q_type": q_type, **res})
                        st.session_state.answered_current = True
                        st.session_state.q_start_time = 0; st.rerun()
            else:
                st.success("Captured.")
                if st.button("Next Question ➔"):
                    if mode == "Dynamic LLM":
                        with st.spinner("Generating follow-up..."):
                            st.session_state.current_dynamic_q = generate_dynamic_question(domain, level, q_type, st.session_state.history, api_key, llm_model)
                    st.session_state.indices[q_type] += 1
                    st.session_state.answered_current = False; st.rerun()
        else:
            st.success(f"{q_type} Track Complete!")
            if st.button("Finish Interview"): st.session_state.interview_active = False; st.rerun()

# --- DEVELOPER VIEW ---
else:
    st.title("👤 Recruiter Dashboard")
    if not st.session_state.history:
        st.info("No interview data recorded yet.")
    else:
        for i, h in enumerate(st.session_state.history):
            with st.expander(f"Q{i+1} ({h['q_type']}): {h['question']}"):
                col_a, col_b = st.columns([2, 1])
                col_a.write(f"**Transcript:** {h['text']}")
                col_b.write(f"⏱️ Latency: {h['latency']}s")
                col_b.write(f"🚩 Status: {'⚠️ FLAGGED' if h['flag'] else '✅ CLEAR'}")
                if h['flag']: st.error(f"Reason: {h['reason']}")
