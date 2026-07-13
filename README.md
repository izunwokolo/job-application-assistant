# 💼 Job Application Assistant

An AI-powered Streamlit app that helps job seekers **classify job postings**, **match their resume skills against job requirements**, and **generate tailored resume bullet points** — all through a simple chat interface.

Built using **LangGraph** and **Google Gemini** for conversational reasoning, with a Streamlit front end for an interactive, no-code experience.

## ✨ Features

- **Resume Upload & Parsing** — Upload a PDF resume and the app extracts and stores the text for use throughout the session.
- **Job Posting Classification** — Paste a job description and the app classifies it (e.g. entry-level vs. senior) using a trained model.
- **Skill Matching** — Compares your resume's skills and experience against a pasted job description, highlighting strong matches and gaps.
- **Tailored Bullet Point Generation** — Generates resume bullet points tailored to a specific job posting based on your background.
- **Conversational Interface** — All interactions happen through natural chat, powered by an LLM agent.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM / Agent Framework:** LangGraph, LangChain
- **Model:** Google Gemini (via `langchain-google-genai`)
- **ML:** scikit-learn (job classification model)
- **PDF Parsing:** pypdf / PyPDF2
- **Data:** Kaggle job postings dataset (via `kagglehub`)

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/izunwokolo/job-application-assistant.git
cd job-application-assistant
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy kagglehub PyPDF2 pypdf scikit-learn langchain-core langgraph langchain-google-genai
```

### 3. Add your Gemini API key

Create a folder called `.streamlit` in the project root (if it doesn't already exist), and inside it create a file called `secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

> You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

**Note:** `.streamlit/secrets.toml` is excluded via `.gitignore` and will not be tracked by git — this keeps your API key private.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📁 Project Structure

```
job-application-assistant/
├── app.py              # Main Streamlit app
├── .gitignore
└── .streamlit/
    └── secrets.toml    # Your API key (not tracked by git)
```

## 📌 Background

This project was originally built as the final project for **INFO 6105 (Data Science)** at Northeastern University, and was presented at the Northeastern Student Research & Capstone Poster Showcase.

## 📄 License

This project is for educational and portfolio purposes.
