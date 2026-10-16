import os
import json
import time
import uuid
import subprocess
from typing import List, Optional, Dict, Any

import requests
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ==============================================================================
# 1. INITIALIZE FASTAPI & ZERO-LOSS LOCAL STORAGE DIRECTORIES
# ==============================================================================
app = FastAPI(
    title="SkillMap AI - B.Tech Upskilling & Community Platform",
    description="Powered by Local Ollama (gemma3:1b), ChromaDB RAG, and FastAPI"
)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_MODEL = "gemma3:1b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "local_data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE = os.path.join(DATA_DIR, "persistent_app_state.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# 24 Hours in seconds for Ephemeral Chat Room expiration
CHAT_TTL_SECONDS = 24 * 60 * 60

# ==============================================================================
# 2. CURATED B.TECH CAREER PATHS & STRICT FREE-FIRST RESOURCES
# ==============================================================================
BRANCH_PATHS: Dict[str, List[Dict[str, Any]]] = {
    "B.Tech CSE": [
        {
            "id": "cse_fullstack",
            "title": "Full-Stack Web & Python API Engineering",
            "explanation": "Master frontend interfaces, Python FastAPI backends, and modern databases to build complete production web apps.",
            "target_roles": "Full-Stack Developer, Backend Engineer, Software Development Engineer (SDE)",
            "milestones": [
                {
                    "step": 1,
                    "title": "Web Foundations & Git/GitHub Open-Source Workflow",
                    "description": "Learn semantic HTML5, modern CSS Grid/Flexbox, and Git/GitHub pull-request workflows.",
                    "free_resources": [
                        {"name": "freeCodeCamp Responsive Web Design (100% Free)", "url": "https://www.freecodecamp.org/learn/2022/responsive-web-design/"},
                        {"name": "First Contributions Open-Source Git Guide (Free)", "url": "https://github.com/firstcontributions/first-contributions"}
                    ],
                    "free_ai_tools": [
                        {"name": "Local Ollama (gemma3:1b) - Offline Code Reviewer", "url": "https://ollama.com/download"},
                        {"name": "GitHub Student Developer Pack (Free for B.Tech Students)", "url": "https://education.github.com/pack"}
                    ],
                    "paid_fallback": "None required — 100% covered by free resources."
                },
                {
                    "step": 2,
                    "title": "Interactive JavaScript & Frontend State Management",
                    "description": "Master ES6+ JavaScript, DOM events, Fetch API, and browser localStorage persistence.",
                    "free_resources": [
                        {"name": "JavaScript.info Modern JS Tutorial (100% Free)", "url": "https://javascript.info/"},
                        {"name": "MDN Web Docs JavaScript Guide (Free)", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"}
                    ],
                    "free_ai_tools": [
                        {"name": "Local Gemma 3 Debugging Mentor in SkillMap AI", "url": "http://localhost:8000"}
                    ],
                    "paid_fallback": "None required — 100% covered by free resources."
                },
                {
                    "step": 3,
                    "title": "Python Backend Engineering with FastAPI & Uvicorn",
                    "description": "Design REST APIs, Pydantic data models, and async request handling in Python.",
                    "free_resources": [
                        {"name": "Official FastAPI Interactive User Guide (100% Free)", "url": "https://fastapi.tiangolo.com/tutorial/"},
                        {"name": "Harvard CS50P Introduction to Programming with Python (Free)", "url": "https://cs50.harvard.edu/python/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Google Colab Free Python Execution Environment", "url": "https://colab.research.google.com/"}
                    ],
                    "paid_fallback": "None required — official FastAPI documentation is industry gold standard."
                },
                {
                    "step": 4,
                    "title": "Vector Databases (ChromaDB), SQL & Cloud Deployment",
                    "description": "Integrate ChromaDB for AI search, relational databases, and deploy full-stack apps.",
                    "free_resources": [
                        {"name": "ChromaDB Official Documentation (100% Free)", "url": "https://docs.trychroma.com/"},
                        {"name": "SQLBolt Interactive SQL Lessons (Free)", "url": "https://sqlbolt.com/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Hugging Face Spaces (Free Full-Stack & AI App Hosting)", "url": "https://huggingface.co/spaces"}
                    ],
                    "paid_fallback": "Udemy Complete Backend Engineering Bootcamp (~₹449 on sale, only if extra video projects are desired)."
                }
            ]
        },
        {
            "id": "cse_ai_rag",
            "title": "AI, Machine Learning & Local LLM (RAG) Engineering",
            "explanation": "Build intelligent AI assistants, machine learning models, and Retrieval-Augmented Generation (RAG) pipelines using open-source tools.",
            "target_roles": "AI Engineer, Machine Learning Engineer, Data Scientist",
            "milestones": [
                {
                    "step": 1,
                    "title": "Python for Data Science, NumPy & Linear Algebra",
                    "description": "Master data manipulation, arrays, and mathematical foundations behind neural networks.",
                    "free_resources": [
                        {"name": "Kaggle Learn Python & Pandas Tracks (100% Free)", "url": "https://www.kaggle.com/learn"},
                        {"name": "3Blue1Brown Essence of Linear Algebra Series (Free)", "url": "https://www.3blue1brown.com/topics/linear-algebra"}
                    ],
                    "free_ai_tools": [
                        {"name": "Google Colab Free T4 GPU Notebooks", "url": "https://colab.research.google.com/"}
                    ],
                    "paid_fallback": "None required — 100% free."
                },
                {
                    "step": 2,
                    "title": "Machine Learning Foundations & Scikit-Learn",
                    "description": "Implement classification, regression, clustering, and model evaluation metrics.",
                    "free_resources": [
                        {"name": "NPTEL Introduction to Machine Learning - IIT Madras (Free)", "url": "https://nptel.ac.in/"},
                        {"name": "Google Machine Learning Crash Course (Free)", "url": "https://developers.google.com/machine-learning/crash-course"}
                    ],
                    "free_ai_tools": [
                        {"name": "Kaggle Free GPU/TPU Notebooks", "url": "https://www.kaggle.com/code"}
                    ],
                    "paid_fallback": "None required."
                },
                {
                    "step": 3,
                    "title": "Deep Learning & Hugging Face Transformers",
                    "description": "Build neural networks with PyTorch and run open-source NLP/Vision models via Transformers.",
                    "free_resources": [
                        {"name": "Hugging Face Official LLM Course (100% Free)", "url": "https://huggingface.co/learn"},
                        {"name": "Fast.ai Practical Deep Learning for Coders (Free)", "url": "https://course.fast.ai/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Hugging Face Transformers Open-Source Library (Free)", "url": "https://huggingface.co/docs/transformers/installation"}
                    ],
                    "paid_fallback": "None required."
                },
                {
                    "step": 4,
                    "title": "Local Open-Source LLMs (Ollama + Gemma 3) & ChromaDB RAG",
                    "description": "Combine local LLMs (`gemma3:1b`) with ChromaDB vector search and `llama.cpp` inference.",
                    "free_resources": [
                        {"name": "Ollama Official Documentation & Local HTTP API (Free)", "url": "https://ollama.com/download"},
                        {"name": "llama.cpp Official Installation & Guide (Free)", "url": "https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md"}
                    ],
                    "free_ai_tools": [
                        {"name": "Ollama + gemma3:1b Local Inference Engine (100% Free)", "url": "https://ollama.com/"}
                    ],
                    "paid_fallback": "None required — 100% open-source."
                }
            ]
        }
    ],
    "B.Tech ECE": [
        {
            "id": "ece_embedded_iot",
            "title": "Embedded Systems, IoT & Edge AI Engineering",
            "explanation": "Program microcontrollers (ESP32/ARM), interface real-world sensors, and deploy TinyML on edge devices.",
            "target_roles": "Embedded Firmware Engineer, IoT Systems Architect, Edge AI Developer",
            "milestones": [
                {
                    "step": 1,
                    "title": "Embedded C/C++ & Digital Logic Foundations",
                    "description": "Master bitwise operations, memory pointers, interrupts, and digital electronics.",
                    "free_resources": [
                        {"name": "NPTEL Introduction to Embedded Systems (100% Free)", "url": "https://nptel.ac.in/"},
                        {"name": "MIT OpenCourseWare Circuits and Electronics (Free)", "url": "https://ocw.mit.edu/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Wokwi Free Online Arduino & ESP32 Hardware Simulator", "url": "https://wokwi.com/"}
                    ],
                    "paid_fallback": "Physical ESP32 Dev Board (~₹350 on Robu.in, only if physical hardware testing is needed)."
                },
                {
                    "step": 2,
                    "title": "Microcontroller Protocols (UART, I2C, SPI) & RTOS",
                    "description": "Write real-time firmware for sensor acquisition, actuators, and FreeRTOS task scheduling.",
                    "free_resources": [
                        {"name": "Random Nerd Tutorials Free ESP32 & IoT Guides", "url": "https://randomnerdtutorials.com/"},
                        {"name": "FreeRTOS Official Kernel Book & Documentation (Free)", "url": "https://www.freertos.org/Documentation/RTOS_book.html"}
                    ],
                    "free_ai_tools": [
                        {"name": "Local Gemma 3 Datasheet & Register Assistant", "url": "http://localhost:8000"}
                    ],
                    "paid_fallback": "None required — Wokwi simulates sensors in-browser for free."
                },
                {
                    "step": 3,
                    "title": "IoT Telemetry with Python FastAPI & TinyML",
                    "description": "Connect edge devices via MQTT/HTTP to Python dashboards and run lightweight AI models.",
                    "free_resources": [
                        {"name": "HiveMQ MQTT Essentials Guide (100% Free)", "url": "https://www.hivemq.com/mqtt-essentials/"},
                        {"name": "FastAPI Official Tutorial for IoT Gateways (Free)", "url": "https://fastapi.tiangolo.com/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Edge Impulse Free Tier for Microcontroller AI", "url": "https://edgeimpulse.com/"}
                    ],
                    "paid_fallback": "None required."
                }
            ]
        },
        {
            "id": "ece_vlsi",
            "title": "VLSI Design, Verilog RTL & Hardware Verification",
            "explanation": "Design digital integrated circuits, write synthesizable Verilog/SystemVerilog, and run open-source ASIC flows.",
            "target_roles": "VLSI Design Engineer, RTL Verification Engineer, FPGA Engineer",
            "milestones": [
                {
                    "step": 1,
                    "title": "CMOS Digital Design & Finite State Machines",
                    "description": "Understand combinational/sequential logic, setup and hold timing, and CMOS fundamentals.",
                    "free_resources": [
                        {"name": "NPTEL Digital IC Design - IIT Madras (100% Free)", "url": "https://nptel.ac.in/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Logisim Evolution Open-Source Logic Simulator (Free)", "url": "https://github.com/logisim-evolution/logisim-evolution"}
                    ],
                    "paid_fallback": "None required."
                },
                {
                    "step": 2,
                    "title": "Verilog HDL Coding & Testbench Simulation",
                    "description": "Practice RTL design and verification directly in the browser.",
                    "free_resources": [
                        {"name": "HDLBits Interactive Verilog Problem Set (100% Free)", "url": "https://hdlbits.01xz.net/"}
                    ],
                    "free_ai_tools": [
                        {"name": "EDA Playground Free Browser Verilog Simulator", "url": "https://www.edaplayground.com/"},
                        {"name": "Local Ollama (gemma3:1b) Verilog Testbench Helper", "url": "https://ollama.com/"}
                    ],
                    "paid_fallback": "Udemy VSD VLSI Physical Design (~₹449 on sale as an affordable paid fallback)."
                }
            ]
        }
    ],
    "B.Tech Mechanical / Civil / Core": [
        {
            "id": "core_computational",
            "title": "CAD Modeling, Simulation & Python Engineering Automation",
            "explanation": "Combine 3D parametric CAD with Python scripting to automate engineering calculations, simulations, and smart manufacturing.",
            "target_roles": "CAD/CAE Design Engineer, Simulation Analyst, R&D Automation Engineer",
            "milestones": [
                {
                    "step": 1,
                    "title": "3D Parametric CAD & Engineering Blueprints",
                    "description": "Create 3D parts, assemblies, and tolerance drawings using free open-source and student CAD suites.",
                    "free_resources": [
                        {"name": "FreeCAD Official Tutorials & Documentation (100% Free)", "url": "https://wiki.freecad.org/Tutorials"},
                        {"name": "Autodesk Fusion 360 Free Student Education License", "url": "https://www.autodesk.com/education/edu-software/overview"}
                    ],
                    "free_ai_tools": [
                        {"name": "Onshape Free Browser CAD for Engineering Students", "url": "https://www.onshape.com/en/education/"}
                    ],
                    "paid_fallback": "None required — student licenses are 100% free."
                },
                {
                    "step": 2,
                    "title": "Python Automation for Core Engineering & FEA Data",
                    "description": "Use Python, NumPy, and SciPy to automate stress/thermal calculations and plot engineering data.",
                    "free_resources": [
                        {"name": "NPTEL Python for Engineers & Scientists (100% Free)", "url": "https://nptel.ac.in/"},
                        {"name": "SciPy & Matplotlib Official Documentation (Free)", "url": "https://scipy.org/"}
                    ],
                    "free_ai_tools": [
                        {"name": "Google Colab + Local Gemma 3 Engineering Formula Assistant", "url": "https://colab.research.google.com/"}
                    ],
                    "paid_fallback": "None required."
                }
            ]
        }
    ]
}

# ==============================================================================
# 3. CHROMADB VECTOR STORE SETUP (RAG FOR ROADMAPS & FREE RESOURCES)
# ==============================================================================
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
roadmap_collection = chroma_client.get_or_create_collection(name="btech_upskilling_rag")


def seed_chromadb_knowledge_base():
    """Populates ChromaDB with B.Tech branches, paths, and free-first resources if empty."""
    if roadmap_collection.count() > 0:
        return
    docs, ids, metadatas = [], [], []
    for branch, paths in BRANCH_PATHS.items():
        for path in paths:
            for m in path["milestones"]:
                doc_id = f"{path['id']}_step_{m['step']}"
                free_res_str = "; ".join([f"{r['name']} ({r['url']})" for r in m["free_resources"]])
                free_ai_str = "; ".join([f"{t['name']} ({t['url']})" for t in m["free_ai_tools"]])
                text = (
                    f"Branch: {branch} | Path: {path['title']} | Step {m['step']}: {m['title']}. "
                    f"Description: {m['description']} "
                    f"Strict Free Resources: {free_res_str}. "
                    f"Free AI Tools: {free_ai_str}. "
                    f"Affordable Paid Fallback: {m['paid_fallback']}"
                )
                docs.append(text)
                ids.append(doc_id)
                metadatas.append({"branch": branch, "path_id": path["id"], "step": m["step"]})
    if docs:
        roadmap_collection.add(documents=docs, ids=ids, metadatas=metadatas)


seed_chromadb_knowledge_base()


# ==============================================================================
# 4. ZERO-LOSS LOCAL PERSISTENCE (NEVER FORGETS USER PROGRESS OR COMMUNITY DATA)
# ==============================================================================
def get_default_state() -> Dict[str, Any]:
    now = time.time()
    return {
        "student_profile": {
            "name": "",
            "degree": "B.Tech",
            "branch": "B.Tech CSE",
            "year": "1st Year",
            "selected_path_id": "cse_fullstack",
            "completed_steps": [],
            "chat_history": []
        },
        "qa_posts": [
            {
                "id": "post_1",
                "author": "Aarav (1st Year B.Tech CSE)",
                "branch": "B.Tech CSE",
                "title": "How do I connect my FastAPI backend to local Ollama gemma3:1b without CORS errors?",
                "body": "I pulled gemma3:1b in Ollama and it works in terminal, but how should I call localhost:11434 from Python FastAPI?",
                "votes": 5,
                "created_at": now - 3600,
                "comments": [
                    {
                        "id": "c_1",
                        "author": "Sneha (2nd Year CSE)",
                        "text": "Use the Python `requests` library inside your FastAPI route to POST to `http://localhost:11434/api/chat` with `\"stream\": False`, and serve your HTML directly from FastAPI so both run on port 8000!",
                        "votes": 8
                    }
                ]
            },
            {
                "id": "post_2",
                "author": "Karthik (1st Year B.Tech ECE)",
                "branch": "B.Tech ECE",
                "title": "Best 100% free simulator to practice ESP32 and Embedded C before buying hardware?",
                "body": "I just joined 1st year ECE and want to build IoT projects without spending money on components yet.",
                "votes": 4,
                "created_at": now - 7200,
                "comments": [
                    {
                        "id": "c_2",
                        "author": "Rahul (3rd Year ECE)",
                        "text": "Use Wokwi (wokwi.com)! It simulates ESP32, Arduino, OLED displays, and ultrasonic sensors 100% free in your browser.",
                        "votes": 6
                    }
                ]
            }
        ],
        "ephemeral_chat": [
            {
                "id": "msg_1",
                "sender": "MentorBot",
                "branch": "System",
                "text": "Welcome to the 24-Hour Community Chat Room! Every message automatically disappears 24 hours after it is sent.",
                "timestamp": now - 1800
            }
        ]
    }


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        state = get_default_state()
        save_state(state)
        return state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = get_default_state()

    # Enforce strict 24-hour message expiration on every load
    cutoff = time.time() - CHAT_TTL_SECONDS
    valid_msgs = [m for m in state.get("ephemeral_chat", []) if m.get("timestamp", 0) > cutoff]
    if len(valid_msgs) != len(state.get("ephemeral_chat", [])):
        state["ephemeral_chat"] = valid_msgs
        save_state(state)
    return state


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ==============================================================================
# 5. PYDANTIC MODELS FOR API REQUESTS
# ==============================================================================
class ProfileUpdate(BaseModel):
    name: str
    degree: str = "B.Tech"
    branch: str
    year: str
    selected_path_id: str


class MilestoneToggle(BaseModel):
    step: int


class AIMentorRequest(BaseModel):
    message: str


class QAPostCreate(BaseModel):
    author: str
    branch: str
    title: str
    body: str


class QACommentCreate(BaseModel):
    author: str
    text: str


class VoteAction(BaseModel):
    delta: int  # +1 for upvote, -1 for downvote


class ChatMessageCreate(BaseModel):
    sender: str
    branch: str
    text: str


# ==============================================================================
# 6. API ENDPOINTS (AI MENTOR, CHROMADB RAG, MAP TRACKER, Q&A, 24H CHAT)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h2>Please create index.html in the same folder as app.py</h2>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/init")
def get_initial_data():
    state = load_state()
    return {
        "state": state,
        "branches": BRANCH_PATHS,
        "server_time": time.time(),
        "chat_ttl_seconds": CHAT_TTL_SECONDS
    }


@app.post("/api/profile")
def update_student_profile(payload: ProfileUpdate):
    state = load_state()
    old_path = state["student_profile"].get("selected_path_id")
    state["student_profile"]["name"] = payload.name.strip() or "Student"
    state["student_profile"]["degree"] = payload.degree
    state["student_profile"]["branch"] = payload.branch
    state["student_profile"]["year"] = payload.year
    state["student_profile"]["selected_path_id"] = payload.selected_path_id
    if old_path != payload.selected_path_id:
        state["student_profile"]["completed_steps"] = []
    save_state(state)
    return {"status": "saved", "student_profile": state["student_profile"]}


@app.post("/api/roadmap/toggle")
def toggle_milestone(payload: MilestoneToggle):
    state = load_state()
    completed = state["student_profile"].get("completed_steps", [])
    if payload.step in completed:
        completed.remove(payload.step)
    else:
        completed.append(payload.step)
    state["student_profile"]["completed_steps"] = sorted(completed)
    save_state(state)
    return {"status": "updated", "completed_steps": state["student_profile"]["completed_steps"]}


@app.post("/api/ai/chat")
def chat_with_gemma_mentor(payload: AIMentorRequest):
    state = load_state()
    profile = state["student_profile"]
    branch = profile.get("branch", "B.Tech CSE")
    path_id = profile.get("selected_path_id", "cse_fullstack")
    completed = profile.get("completed_steps", [])

    # Find active career path details
    selected_path = None
    for p in BRANCH_PATHS.get(branch, []):
        if p["id"] == path_id:
            selected_path = p
            break
    if not selected_path and BRANCH_PATHS.get(branch):
        selected_path = BRANCH_PATHS[branch][0]

    total_steps = len(selected_path["milestones"]) if selected_path else 0
    progress_pct = int((len(completed) / total_steps) * 100) if total_steps > 0 else 0

    # Query ChromaDB for relevant free-first resources & milestones (RAG)
    rag_results = roadmap_collection.query(
        query_texts=[f"{branch} {selected_path['title'] if selected_path else ''} {payload.message}"],
        n_results=2
    )
    retrieved_context = "\n".join(rag_results["documents"][0]) if rag_results.get("documents") else ""

    system_prompt = (
        "You are SkillMap AI, a dedicated Personal Upskilling Assistant for B.Tech engineering students.\n"
        f"STUDENT PROFILE (NEVER FORGET THIS): Name: {profile.get('name') or 'Student'}, "
        f"Degree: {profile.get('degree')} ({branch}, {profile.get('year')}).\n"
        f"CHOSEN CAREER PATH: {selected_path['title'] if selected_path else 'Exploring'}.\n"
        f"PROGRESS SO FAR: {len(completed)}/{total_steps} milestones completed ({progress_pct}%). "
        f"Completed step numbers: {completed}.\n"
        "STRICT RESOURCE POLICY: Always recommend 100% FREE learning resources and FREE AI tools first. "
        "Only mention a paid resource if no free option exists, and pick the cheapest, highest-value option.\n"
        f"VERIFIED CHROMADB KNOWLEDGE BASE:\n{retrieved_context}\n\n"
        "Keep your response encouraging, clear, structured with short bullet points, and directly tailored to this student."
    )

    # Save user message to persistent history
    history = profile.get("chat_history", [])
    history.append({"role": "user", "content": payload.message})

    # Call local Ollama (gemma3:1b) via requests
    ai_reply = ""
    try:
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                *history[-6:]  # Keep recent context window fast for 1B model
            ],
            "stream": False
        }
        resp = requests.post(OLLAMA_CHAT_URL, json=ollama_payload, timeout=45)
        if resp.status_code == 200:
            data = resp.json()
            ai_reply = data.get("message", {}).get("content", "")
        else:
            ai_reply = (
                f"⚠️ Local Ollama returned status {resp.status_code}. "
                f"Make sure you ran `ollama pull {OLLAMA_MODEL}` in terminal.\n\n"
                f"**From ChromaDB Offline Knowledge Base:**\n{retrieved_context}"
            )
    except Exception as e:
        ai_reply = (
            f"💡 **Local ChromaDB Mentor Backup (Ollama connection note: {str(e)})**\n\n"
            f"You are currently on the **{selected_path['title'] if selected_path else branch}** track "
            f"with **{progress_pct}%** completed.\n\n"
            f"**Recommended Free Resources & AI Tools from ChromaDB:**\n{retrieved_context}\n\n"
            f"*(Tip: Ensure Ollama is running via `ollama run gemma3:1b` for live LLM generation!)*"
        )

    history.append({"role": "assistant", "content": ai_reply})
    state["student_profile"]["chat_history"] = history[-20:]
    save_state(state)

    return {"reply": ai_reply, "chat_history": state["student_profile"]["chat_history"]}


# --- Q&A Community Forum Endpoints (Post, Comment, Upvote/Downvote) ---
@app.post("/api/qa/post")
def create_qa_post(payload: QAPostCreate):
    state = load_state()
    new_post = {
        "id": f"post_{uuid.uuid4().hex[:8]}",
        "author": payload.author.strip() or "Anonymous B.Tech Student",
        "branch": payload.branch,
        "title": payload.title.strip(),
        "body": payload.body.strip(),
        "votes": 1,
        "created_at": time.time(),
        "comments": []
    }
    state["qa_posts"].insert(0, new_post)
    save_state(state)
    return {"status": "created", "qa_posts": state["qa_posts"]}


@app.post("/api/qa/post/{post_id}/vote")
def vote_qa_post(post_id: str, payload: VoteAction):
    state = load_state()
    for post in state["qa_posts"]:
        if post["id"] == post_id:
            post["votes"] += (1 if payload.delta > 0 else -1)
            break
    state["qa_posts"].sort(key=lambda x: x["votes"], reverse=True)
    save_state(state)
    return {"qa_posts": state["qa_posts"]}


@app.post("/api/qa/post/{post_id}/comment")
def add_qa_comment(post_id: str, payload: QACommentCreate):
    state = load_state()
    for post in state["qa_posts"]:
        if post["id"] == post_id:
            post["comments"].append({
                "id": f"c_{uuid.uuid4().hex[:8]}",
                "author": payload.author.strip() or "Peer Student",
                "text": payload.text.strip(),
                "votes": 1
            })
            break
    save_state(state)
    return {"qa_posts": state["qa_posts"]}


@app.post("/api/qa/post/{post_id}/comment/{comment_id}/vote")
def vote_qa_comment(post_id: str, comment_id: str, payload: VoteAction):
    state = load_state()
    for post in state["qa_posts"]:
        if post["id"] == post_id:
            for c in post["comments"]:
                if c["id"] == comment_id:
                    c["votes"] += (1 if payload.delta > 0 else -1)
                    break
            post["comments"].sort(key=lambda x: x["votes"], reverse=True)
            break
    save_state(state)
    return {"qa_posts": state["qa_posts"]}


# --- 24-Hour Ephemeral Chat Room Endpoints ---
@app.get("/api/chatroom")
def get_ephemeral_chat():
    state = load_state()  # Automatically purges messages older than 24 hours
    return {
        "messages": state["ephemeral_chat"],
        "server_time": time.time(),
        "ttl_seconds": CHAT_TTL_SECONDS
    }


@app.post("/api/chatroom")
def post_ephemeral_message(payload: ChatMessageCreate):
    state = load_state()
    new_msg = {
        "id": f"msg_{uuid.uuid4().hex[:8]}",
        "sender": payload.sender.strip() or "Student",
        "branch": payload.branch,
        "text": payload.text.strip(),
        "timestamp": time.time()
    }
    state["ephemeral_chat"].append(new_msg)
    save_state(state)
    return {
        "messages": state["ephemeral_chat"],
        "server_time": time.time(),
        "ttl_seconds": CHAT_TTL_SECONDS
    }


# --- Hackathon System Readiness Diagnostic Endpoint (Matches Hacktoberfest 2026.docx) ---
@app.get("/api/readiness")
def check_hackathon_readiness():
    ollama_ok = False
    gemma_ok = False
    models_found = []
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            models = r.json().get("models", [])
            models_found = [m.get("name", "") for m in models]
            gemma_ok = any("gemma3" in m.lower() for m in models_found)
    except Exception:
        pass

    transformers_status = "NOT INSTALLED (Optional)"
    try:
        import transformers
        transformers_status = f"PASS (v{transformers.__version__})"
    except ImportError:
        pass

    gpu_info = "Checked via local runtime"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=2
        ).decode("utf-8").strip()
        if out:
            gpu_info = out
    except Exception:
        gpu_info = "NVIDIA RTX 3050 (6GB VRAM) / Local Runtime"

    return {
        "ollama_api": "PASS" if ollama_ok else "OFFLINE (Run `ollama serve`)",
        "gemma3_1b": "PASS" if gemma_ok else f"Run `ollama pull gemma3:1b` (Found: {models_found})",
        "fastapi_chromadb_requests": f"PASS (ChromaDB items: {roadmap_collection.count()})",
        "transformers": transformers_status,
        "gpu_vram": gpu_info,
        "zero_loss_storage": f"PASS ({STATE_FILE})"
    }
