"""
SANDBOX-ONLY DEMO
==================
This script proves the matching_engine.py pipeline logic end-to-end using
TF-IDF vectors instead of Sentence Transformer embeddings, because this
sandbox cannot reach huggingface.co to download the real model.

On your own machine / deployment host, matching_engine.py will use REAL
Sentence Transformer embeddings (semantic, not just keyword overlap) —
this demo is just to visually confirm the matching + explainability logic
works correctly before you run the real thing.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

students = pd.read_csv("students.csv")
businesses = pd.read_csv("businesses.csv")

vectorizer = TfidfVectorizer(stop_words="english")
all_text = pd.concat([students["semantic_profile_text"], businesses["semantic_profile_text"]])
vectorizer.fit(all_text)

student_vecs = vectorizer.transform(students["semantic_profile_text"])
business_vecs = vectorizer.transform(businesses["semantic_profile_text"])


def explain_match(student_row, business_row, score):
    student_skills = set(s.strip().lower() for s in str(student_row["skills"]).split(";"))
    business_skills = set(s.strip().lower() for s in str(business_row["expertise_areas"]).split(";"))
    shared_skills = student_skills & business_skills
    same_domain = student_row["domain_category"] == business_row["domain_category"]

    reasons = []
    if same_domain:
        reasons.append(f"both are in the {student_row['domain_category']} space")
    if shared_skills:
        reasons.append(f"overlapping skills in {', '.join(sorted(shared_skills))}")
    if not reasons:
        reasons.append("general semantic similarity between profile descriptions")

    confidence = "high" if score >= 0.3 else "moderate" if score >= 0.15 else "low"
    return f"Match confidence: {confidence} (similarity {score:.3f}). Surfaced because {', and '.join(reasons)}."


def top_matches(student_id, top_k=3):
    idx = students.index[students["student_id"] == student_id][0]
    sims = cosine_similarity(student_vecs[idx], business_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    print(f"\n=== Top {top_k} matches for {students.iloc[idx]['name']} ({student_id}) ===")
    print(f"Domain: {students.iloc[idx]['domain_category']} | Major: {students.iloc[idx]['major']}\n")

    for rank, b_idx in enumerate(top_idx, start=1):
        b = businesses.iloc[b_idx]
        score = sims[b_idx]
        explanation = explain_match(students.iloc[idx], b, score)
        print(f"{rank}. {b['mentor_name']} — {b['title']} at {b['company_name']}")
        print(f"   {explanation}\n")


# Demo for 3 sample students across different domains
for sid in ["STU0001", "STU0002", "STU0010"]:
    top_matches(sid, top_k=3)
