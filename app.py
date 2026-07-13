# ============================================================
# Job Application Assistant — Streamlit Web App
# ============================================================

# ── SECTION 1: IMPORTS ──────────────────────────────────────
import os
import re
import io
import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import PyPDF2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from pypdf import PdfReader 

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI


# ── SECTION 2: LOAD GEMINI API KEY ──────────────────────────
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]


# ── SECTION 3: LOAD AND TRAIN THE ML MODEL ──────────────────
@st.cache_resource
def load_model():
    path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
    file_path = os.path.join(path, "postings.csv")
    df = pd.read_csv(file_path)

    df = df.dropna(subset=["formatted_experience_level"])

    entry_labels = ["Internship", "Entry level", "Associate"]
    senior_labels = ["Mid-Senior level", "Director", "Executive"]
    df = df[df["formatted_experience_level"].isin(entry_labels + senior_labels)].copy()

    df["target"] = df["formatted_experience_level"].apply(
        lambda x: 0 if x in entry_labels else 1
    )

    df["word_count"] = df["description"].apply(get_word_count)
    df["years_experience"] = df["description"].apply(extract_years_experience)
    df["management_keywords_count"] = df["description"].apply(
        lambda x: count_keywords(x, management_words)
    )
    df["technical_skills_count"] = df["description"].apply(
        lambda x: count_keywords(x, technical_words)
    )

    features = ["word_count", "years_experience", "management_keywords_count", "technical_skills_count"]
    X = df[features]
    y = df["target"]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


# ── SECTION 4: KEYWORD LISTS ────────────────────────────────
management_words = [
    "lead", "manage", "mentor", "oversee", "strategy",
    "stakeholder", "director", "ownership", "coordinate", "supervise"
]

technical_words = [
    "python", "sql", "excel", "aws", "tableau",
    "java", "machine learning", "power bi", "spark", "git",
    "cloud", "statistics", "pandas", "numpy", "data analysis"
]


# ── SECTION 5: FEATURE EXTRACTION FUNCTIONS ─────────────────
def get_word_count(text):
    """Count words in a job description. Returns NaN if text is missing."""
    if pd.isna(text):
        return np.nan
    return len(str(text).split())


def extract_years_experience(text):
    """Extract years of experience mentioned in a job description using regex."""
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    patterns = [
        r'(\d+)\+?\s+years',
        r'(\d+)\+?\s+yrs',
        r'(\d+)-\d+\s+years',
        r'(\d+)-\d+\s+yrs'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return 0


def count_keywords(text, keyword_list):
    """Count how many times keywords from a list appear in the text."""
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    return sum(text.count(word) for word in keyword_list)


# ── SECTION 6: RESUME PDF EXTRACTOR ─────────────────────────
def extract_text_from_pdf(uploaded_file):
    """
    Extract plain text from an uploaded PDF resume.
    PyPDF2 reads each page of the PDF and joins all the text together.
    Returns the full resume text as a single string.
    """
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


# ── SECTION 7: ML PREDICTION FUNCTION ───────────────────────
chosen_threshold = 0.5

def predict_job_level(word_count, years_experience, management_keywords_count, technical_skills_count):
    """
    Predict whether a job is Entry-Level or Senior-Level.
    Returns a dict with the prediction label and confidence score.
    """
    model = load_model()
    input_df = pd.DataFrame([{
        "word_count": word_count,
        "years_experience": years_experience,
        "management_keywords_count": management_keywords_count,
        "technical_skills_count": technical_skills_count
    }])
    probability = model.predict_proba(input_df)[0, 1]
    prediction = "Senior-Level" if probability >= chosen_threshold else "Entry-Level"
    return {"prediction": prediction, "confidence": float(round(probability, 3))}


# ── SECTION 8: AGENT TOOLS ──────────────────────────────────
@tool
def classify_job_posting(job_description: str) -> str:
    """
    Classify a job posting as Entry-Level or Senior-Level from its raw text.
    Use this when the user shares a job description and asks about seniority level.
    """
    wc = get_word_count(job_description)
    yrs = extract_years_experience(job_description)
    mgmt = count_keywords(job_description, management_words)
    tech = count_keywords(job_description, technical_words)
    result = predict_job_level(
        word_count=wc,
        years_experience=yrs,
        management_keywords_count=mgmt,
        technical_skills_count=tech
    )
    return (
        f"Prediction: {result['prediction']}\n"
        f"Confidence: {result['confidence']}\n"
        f"Extracted Features:\n"
        f"  - Word count: {wc}\n"
        f"  - Years of experience mentioned: {yrs}\n"
        f"  - Management keywords: {mgmt}\n"
        f"  - Technical skill keywords: {tech}"
    )


@tool
def match_skills(job_description: str, user_skills: str) -> str:
    """
    Compare user's skills to a job description and identify matches and gaps.
    Use this when the user shares their skills (typed or from resume) and wants
    to know how well they match a job posting.
    """
    job_text = job_description.lower()
    user_skill_list = [s.strip().lower() for s in user_skills.split(",")]
    matched = [skill for skill in user_skill_list if skill in job_text]
    missing = [skill for skill in user_skill_list if skill not in job_text]
    job_keywords_found = [word for word in technical_words if word in job_text]
    skill_gaps = [kw for kw in job_keywords_found if kw not in user_skill_list]
    match_rate = len(matched) / len(user_skill_list) if user_skill_list else 0
    return (
        f"Match rate: {round(match_rate * 100)}%\n"
        f"Your skills that match the job: {', '.join(matched) if matched else 'None found'}\n"
        f"Your skills not mentioned in job: {', '.join(missing) if missing else 'None'}\n"
        f"Skills the job wants that you didn't list: {', '.join(skill_gaps[:5]) if skill_gaps else 'None detected'}\n"
        f"Tip: {'Strong match! Highlight these skills prominently.' if match_rate >= 0.6 else 'Consider adding more relevant skills to your resume.'}"
    )


@tool
def generate_resume_bullets(job_description: str, user_background: str) -> str:
    """
    Generate 3 tailored resume bullet points for a specific job.
    Use this when the user wants help writing resume content for a job.
    """
    job_text = job_description.lower()
    is_senior = any(word in job_text for word in ["lead", "manage", "director", "strategy", "oversee"])
    relevant_tech = [word for word in technical_words if word in job_text][:3]
    tech_str = ", ".join(relevant_tech) if relevant_tech else "relevant tools"
    if is_senior:
        bullets = [
            f"• Led cross-functional initiatives leveraging {tech_str} to drive measurable business outcomes, improving team efficiency by X%.",
            f"• Managed end-to-end project lifecycle for data-driven solutions, collaborating with stakeholders to align technical work with strategic goals.",
            f"• Mentored junior team members on best practices in {tech_str}, contributing to a culture of continuous learning and technical excellence.",
        ]
    else:
        bullets = [
            f"• Analyzed and visualized data using {tech_str} to support business reporting and inform team decisions.",
            f"• Collaborated with cross-functional teams to gather requirements and deliver data solutions aligned with project goals.",
            f"• Built and maintained dashboards and automated reports using {tech_str}, reducing manual effort by X hours per week.",
        ]
    return (
        f"Here are 3 tailored resume bullets for this role:\n\n"
        + "\n".join(bullets)
        + "\n\nTip: Replace 'X%' and 'X hours' with real numbers from your experience — "
          "quantified achievements stand out to recruiters."
    )


# ── SECTION 9: BUILD THE LANGGRAPH AGENT ────────────────────
@st.cache_resource
def build_graph():
    tools = [classify_job_posting, match_skills, generate_resume_bullets]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        system_prompt = SystemMessage(
            content=(
                "You are a helpful job application assistant. "
                "Your goal is to help users understand job postings and strengthen their applications.\n\n"
                "You have access to three tools — use them when appropriate:\n"
                "1. classify_job_posting: Use when the user shares a job description and wants to know "
                "if it's entry-level or senior-level.\n"
                "2. match_skills: Use when the user shares their skills AND a job description and wants "
                "to know how well they match. The skills may come from a resume they uploaded, "
                "or they may have typed them manually.\n"
                "3. generate_resume_bullets: Use when the user wants help writing resume bullet points "
                "for a specific job.\n\n"
                "Always explain results in plain, encouraging language. "
                "For general questions (interview tips, cover letters, salary negotiation), "
                "answer directly without using a tool.\n\n"
                "Remember previous messages — refer back to job descriptions or skills the user "
                "already shared so they don't have to repeat themselves."
            )
        )
        response = llm_with_tools.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def route_tools(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    tool_node = ToolNode(tools)
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "assistant")
    graph_builder.add_conditional_edges("assistant", route_tools)
    graph_builder.add_edge("tools", "assistant")

    return graph_builder.compile()


# ── SECTION 10: GEMINI RESPONSE HELPER ──────────────────────
def get_content(message):
    """Extract clean text from Gemini's response (handles list or string format)."""
    if isinstance(message.content, list):
        return " ".join([block.get("text", "") for block in message.content if isinstance(block, dict)])
    return message.content


# ── SECTION 11: STREAMLIT UI ────────────────────────────────
st.title("💼 Job Application Assistant")
st.write(
    "I can help you **classify job postings**, **match your skills**, "
    "and **generate tailored resume bullet points** — all through chat!"
)

# ── RESUME UPLOAD SECTION ────────────────────────────────────
# This appears at the top of the page as a sidebar-style panel.
# The user uploads their resume once, and the extracted text is
# stored in session_state so every chat message can reference it.

with st.expander("📄 Upload Your Resume (optional)", expanded=False):
    uploaded_resume = st.file_uploader(
        "Upload your resume as a PDF to automatically match your skills to job postings",
        type=["pdf"]
    )

    if uploaded_resume is not None:
        # Extract text from the uploaded PDF
        resume_text = extract_text_from_pdf(uploaded_resume)

        # Save it to session state so the agent can use it throughout the conversation
        st.session_state.resume_text = resume_text

        # Show a success message and a preview of the extracted text
        st.success("✅ Resume uploaded successfully!")
        if st.checkbox("Preview extracted text"):
            st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

# Show a reminder if a resume is already uploaded
if "resume_text" in st.session_state:
    st.info("📄 Resume loaded — I'll use it automatically when matching your skills to jobs.")

# ── CONVERSATION MEMORY ──────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Load the compiled graph (cached — only built once)
graph = build_graph()

# Display all previous messages
for message in st.session_state.conversation_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# ── CHAT INPUT ───────────────────────────────────────────────
user_input = st.chat_input("Paste a job description or ask me anything...")

if user_input:
    # If the user has uploaded a resume, automatically append it to their message
    # so the agent can use it for skill matching without the user having to retype it.
    if "resume_text" in st.session_state:
        full_message = (
            f"{user_input}\n\n"
            f"[Resume context — use this for skill matching if relevant]:\n"
            f"{st.session_state.resume_text[:3000]}"  # limit to first 3000 chars to stay within token limits
        )
    else:
        full_message = user_input

    # Show only the user's original message on screen (not the full resume text)
    with st.chat_message("user"):
        st.write(user_input)

    # Add the full message (with resume if uploaded) to memory
    st.session_state.conversation_history.append(
        HumanMessage(content=full_message)
    )

    # Send to agent
    with st.spinner("Thinking..."):
        response = graph.invoke({
            "messages": st.session_state.conversation_history
        })
        agent_reply = response["messages"][-1]
        reply_text = get_content(agent_reply)

    # Show the assistant's reply
    with st.chat_message("assistant"):
        st.write(reply_text)

    # Save clean reply to memory
    st.session_state.conversation_history.append(
        AIMessage(content=reply_text)
    )