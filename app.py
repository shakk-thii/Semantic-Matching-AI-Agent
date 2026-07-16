"""
AI Semantic Matching Engine — Web App
========================================
Streamlit app that ties together:
    - Student & business profile loading
    - Semantic embedding (Sentence Transformers, with automatic TF-IDF fallback
      if the embedding model can't be downloaded — e.g. no internet at runtime)
    - Cosine similarity top-K matching
    - Rule-based explainable AI reasoning per match
    - A feedback loop: users rate matches, feedback is appended to feedback_logs.csv
      so the matching engine has real signal to improve on over time.

Run locally:
    streamlit run app.py

Deploy for free:
    Streamlit Community Cloud (share.streamlit.io) — push this repo to GitHub,
    connect it, done. See DEPLOYMENT.md for the full walkthrough.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from llm_explainer import LLMExplainer, build_student_facts, build_business_facts

st.set_page_config(page_title="AI Semantic Matching Engine", page_icon="🎯", layout="wide")

STUDENTS_CSV = "students.csv"
BUSINESSES_CSV = "businesses.csv"
FEEDBACK_CSV = "feedback_logs.csv"

# ---------------------------------------------------------------------------
# Data + model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    students = pd.read_csv(STUDENTS_CSV)
    businesses = pd.read_csv(BUSINESSES_CSV)
    return students, businesses


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """
    Tries to load the real Sentence Transformer model.
    Falls back to TF-IDF if the model can't be downloaded (e.g. restricted network).
    Returns a tuple: (mode, encoder_object)
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return "sentence-transformers", model
    except Exception as e:
        st.warning(
            "Falling back to TF-IDF embeddings — Sentence Transformer model "
            f"could not be loaded ({type(e).__name__}). Matching will still work, "
            "but semantic quality is lower than the real model."
        )
        from sklearn.feature_extraction.text import TfidfVectorizer
        return "tfidf", TfidfVectorizer(stop_words="english")


@st.cache_resource(show_spinner="Building profile embeddings...")
def build_embeddings(_mode, _encoder, students_texts, businesses_texts):
    if _mode == "sentence-transformers":
        student_vecs = _encoder.encode(students_texts, normalize_embeddings=True)
        business_vecs = _encoder.encode(businesses_texts, normalize_embeddings=True)
    else:
        all_text = list(students_texts) + list(businesses_texts)
        _encoder.fit(all_text)
        student_vecs = _encoder.transform(students_texts)
        business_vecs = _encoder.transform(businesses_texts)
    return student_vecs, business_vecs


def explain_match(student_row, business_row, score):
    student_skills = set(s.strip().lower() for s in str(student_row["skills"]).split(";"))
    business_skills = set(s.strip().lower() for s in str(business_row["expertise_areas"]).split(";"))
    shared_skills = student_skills & business_skills

    student_mentorship = set(s.strip().lower() for s in str(student_row["preferred_mentorship_type"]).split(";"))
    business_offerings = set(s.strip().lower() for s in str(business_row["offerings"]).split(";"))
    shared_focus = student_mentorship & business_offerings

    same_domain = student_row["domain_category"] == business_row["domain_category"]

    reasons = []
    if same_domain:
        reasons.append(f"both are in the **{student_row['domain_category']}** space")
    if shared_skills:
        reasons.append(f"overlapping skills in **{', '.join(sorted(shared_skills))}**")
    if shared_focus:
        reasons.append(f"the mentor offers **{', '.join(sorted(shared_focus))}**, matching what the student wants")
    if not reasons:
        reasons.append("general semantic similarity between profile descriptions")

    return "This match was surfaced because " + ", and ".join(reasons) + "."


def append_feedback(student_id, business_id, rating, comment):
    new_row = pd.DataFrame([{
        "student_id": student_id,
        "business_id": business_id,
        "rating_1_to_5": rating,
        "accepted_match": rating >= 3,
        "feedback_comment": comment,
    }])
    if os.path.exists(FEEDBACK_CSV):
        new_row.to_csv(FEEDBACK_CSV, mode="a", header=False, index=False)
    else:
        new_row.to_csv(FEEDBACK_CSV, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("🎯 AI Semantic Matching Engine")
st.caption("Student ↔ Business/Mentor matching powered by semantic embeddings + cosine similarity")

students, businesses = load_data()
mode, encoder = load_embedder()
student_vecs, business_vecs = build_embeddings(
    mode, encoder,
    students["semantic_profile_text"].tolist(),
    businesses["semantic_profile_text"].tolist(),
)

st.sidebar.header("Find Matches")
student_display = students["student_id"] + " — " + students["name"] + " (" + students["major"] + ")"
selected = st.sidebar.selectbox("Select a student", student_display)
selected_id = selected.split(" — ")[0]
top_k = st.sidebar.slider("Number of matches to show", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.caption(f"Embedding mode: `{mode}`")
st.sidebar.caption(f"{len(students)} students · {len(businesses)} businesses loaded")

st.sidebar.markdown("---")
st.sidebar.subheader("Explainability")
use_llm_explanations = st.sidebar.toggle("Use AI-generated explanations", value=False)
api_key_input = None
llm_explainer = None
if use_llm_explanations:
    api_key_input = (
        st.secrets.get("ANTHROPIC_API_KEY", None)
        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets
        else st.sidebar.text_input("Anthropic API Key", type="password")
    )
    if api_key_input:
        try:
            llm_explainer = LLMExplainer(api_key=api_key_input)
            st.sidebar.success("AI explanations enabled.")
        except Exception as e:
            st.sidebar.error(f"Could not initialize Anthropic client: {e}")
            use_llm_explanations = False
    else:
        st.sidebar.info("Enter your Anthropic API key to enable AI-generated explanations.")
        use_llm_explanations = False

student_idx = students.index[students["student_id"] == selected_id][0]
student_row = students.iloc[student_idx]

st.subheader(f"Profile: {student_row['name']}")
st.write(student_row["semantic_profile_text"])

sims = cosine_similarity(student_vecs[student_idx].reshape(1, -1) if mode == "sentence-transformers" else student_vecs[student_idx], business_vecs)[0]
top_idx = np.argsort(sims)[::-1][:top_k]

st.subheader(f"Top {top_k} Matches")

for rank, b_idx in enumerate(top_idx, start=1):
    b = businesses.iloc[b_idx]
    score = float(sims[b_idx])
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{rank}. {b['mentor_name']}** — {b['title']} at {b['company_name']}")
            st.caption(b["semantic_profile_text"])
            if use_llm_explanations and llm_explainer is not None:
                try:
                    with st.spinner("Generating AI explanation..."):
                        explanation_text = llm_explainer.explain(
                            build_student_facts(student_row), build_business_facts(b), score
                        )
                    st.markdown(f"🤖 {explanation_text}")
                except Exception as e:
                    st.warning(f"AI explanation failed ({type(e).__name__}), showing rule-based reasoning instead.")
                    st.markdown(explain_match(student_row, b, score))
            else:
                st.markdown(explain_match(student_row, b, score))
        with col2:
            st.metric("Similarity", f"{score:.2f}")

        # --- Feedback loop UI ---
        fb_col1, fb_col2 = st.columns([2, 3])
        with fb_col1:
            rating = st.slider(
                "Rate this match", 1, 5, 3,
                key=f"rating_{student_row['student_id']}_{b['business_id']}"
            )
        with fb_col2:
            comment = st.text_input(
                "Optional comment", key=f"comment_{student_row['student_id']}_{b['business_id']}"
            )
        if st.button("Submit feedback", key=f"submit_{student_row['student_id']}_{b['business_id']}"):
            append_feedback(student_row["student_id"], b["business_id"], rating, comment)
            st.success("Feedback recorded — thank you!")

st.markdown("---")
st.caption(
    "Feedback submitted here is appended to feedback_logs.csv. "
    "This is the foundation for the planned feedback-loop retraining step."
)
