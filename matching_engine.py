"""
AI Semantic Matching Engine — Core Matching Logic
====================================================
Pipeline:
    1. Load student & business profiles (semantic_profile_text field)
    2. Generate embeddings via Sentence Transformers
    3. Compute cosine similarity (student embeddings vs business embeddings)
    4. Return top-K matches per student
    5. Generate explainable AI reasoning for each match (rule-based, no API cost —
       upgradeable later to an LLM-generated explanation using the Anthropic API)

Run:
    python matching_engine.py --student_id STU0001 --top_k 5

Requires: pip install sentence-transformers pandas numpy scikit-learn
"""

import argparse
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, lightweight, free — good default for production


class MatchingEngine:
    def __init__(self, students_csv, businesses_csv, model_name=MODEL_NAME):
        self.students = pd.read_csv(students_csv)
        self.businesses = pd.read_csv(businesses_csv)
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        self.student_embeddings = None
        self.business_embeddings = None

    # ------------------------------------------------------------------
    def build_embeddings(self):
        """Encode semantic_profile_text for every profile. Cache in memory."""
        print(f"Embedding {len(self.students)} student profiles...")
        self.student_embeddings = self.model.encode(
            self.students["semantic_profile_text"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True,  # pre-normalize -> cosine sim = dot product
        )
        print(f"Embedding {len(self.businesses)} business profiles...")
        self.business_embeddings = self.model.encode(
            self.businesses["semantic_profile_text"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    # ------------------------------------------------------------------
    def save_embeddings(self, path="embeddings.npz"):
        np.savez(
            path,
            student_embeddings=self.student_embeddings,
            business_embeddings=self.business_embeddings,
        )
        print(f"Saved embeddings to {path}")

    def load_embeddings(self, path="embeddings.npz"):
        data = np.load(path)
        self.student_embeddings = data["student_embeddings"]
        self.business_embeddings = data["business_embeddings"]

    # ------------------------------------------------------------------
    def top_matches_for_student(self, student_id, top_k=5):
        idx = self.students.index[self.students["student_id"] == student_id]
        if len(idx) == 0:
            raise ValueError(f"student_id {student_id} not found")
        idx = idx[0]

        student_vec = self.student_embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(student_vec, self.business_embeddings)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, b_idx in enumerate(top_idx, start=1):
            business_row = self.businesses.iloc[b_idx]
            score = float(sims[b_idx])
            reasoning = self.explain_match(self.students.iloc[idx], business_row, score)
            results.append({
                "rank": rank,
                "business_id": business_row["business_id"],
                "company_name": business_row["company_name"],
                "mentor_name": business_row["mentor_name"],
                "similarity_score": round(score, 4),
                "explanation": reasoning,
            })
        return results

    # ------------------------------------------------------------------
    def explain_match(self, student_row, business_row, score):
        """
        Rule-based explainable AI layer.
        Compares structured attributes (not just the black-box embedding score)
        so the user gets a human-readable 'why' for every match.

        NOTE: This can later be swapped for an LLM-generated explanation
        (e.g. via the Anthropic API) by feeding these same overlap signals
        as context to a prompt — see explain_match_llm() stub below.
        """
        student_skills = set(s.strip().lower() for s in str(student_row["skills"]).split(";"))
        business_skills = set(s.strip().lower() for s in str(business_row["expertise_areas"]).split(";"))
        shared_skills = student_skills & business_skills

        same_domain = student_row["domain_category"] == business_row["domain_category"]

        student_mentorship = set(s.strip().lower() for s in str(student_row["preferred_mentorship_type"]).split(";"))
        business_offerings = set(s.strip().lower() for s in str(business_row["offerings"]).split(";"))
        shared_mentorship_focus = student_mentorship & business_offerings

        reasons = []
        if same_domain:
            reasons.append(f"both are in the {student_row['domain_category']} space")
        if shared_skills:
            reasons.append(f"overlapping skills in {', '.join(sorted(shared_skills))}")
        if shared_mentorship_focus:
            reasons.append(f"the mentor offers {', '.join(sorted(shared_mentorship_focus))}, which matches what the student is seeking")

        if not reasons:
            reasons.append("a general semantic similarity between profile descriptions")

        confidence = "high" if score >= 0.6 else "moderate" if score >= 0.4 else "low"

        return (
            f"Match confidence: {confidence} (cosine similarity {score:.2f}). "
            f"This pairing was surfaced because {', and '.join(reasons)}."
        )

    # ------------------------------------------------------------------
    def explain_match_llm(self, student_row, business_row, score, llm_explainer=None):
        """
        LLM-generated explanation, using llm_explainer.LLMExplainer.
        Pass an already-constructed LLMExplainer instance (so the API client
        and on-disk cache are reused across calls instead of rebuilt each time).

        Falls back to the rule-based explain_match() if no explainer is passed
        or if the API call fails for any reason (e.g. missing/invalid key,
        rate limit, network issue) — this keeps the pipeline usable even
        without API access.
        """
        if llm_explainer is None:
            return self.explain_match(student_row, business_row, score)

        from llm_explainer import build_student_facts, build_business_facts
        try:
            return llm_explainer.explain(
                build_student_facts(student_row), build_business_facts(business_row), score
            )
        except Exception:
            return self.explain_match(student_row, business_row, score)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--students_csv", default="students.csv")
    parser.add_argument("--businesses_csv", default="businesses.csv")
    parser.add_argument("--student_id", default="STU0001")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    engine = MatchingEngine(args.students_csv, args.businesses_csv)
    engine.build_embeddings()
    engine.save_embeddings()

    matches = engine.top_matches_for_student(args.student_id, top_k=args.top_k)
    print(json.dumps(matches, indent=2))
