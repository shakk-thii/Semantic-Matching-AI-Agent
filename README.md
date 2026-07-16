# AI Semantic Matching Engine

A semantic ai agent that matches business owners and other relations with students based on cosine similarity , traits , requirements and other attributes

## How it works
1. **Parse profiles** — student and business profiles are combined into a single
   natural-language `semantic_profile_text` field per profile (see `students.csv` /
   `businesses.csv`).
2. **Embed** — each profile's text is encoded into a vector using Sentence
   Transformers (`all-MiniLM-L6-v2`).
3. **Match** — cosine similarity is computed between a student's embedding and every
   business embedding; the top-K highest scoring businesses are returned.
4. **Explain** — each match includes a plain-language reason it was surfaced, either:
   - rule-based (free, always on — compares shared skills/domain/mentorship focus), or
   - AI-generated via the Anthropic API (optional, richer reasoning, off by default)
5. **Feedback loop** — users rate each match (1-5) with an optional comment; this is
   logged to `feedback_logs.csv` as the foundation for future retraining/reweighting.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app — the main deliverable |
| `matching_engine.py` | Standalone CLI version of the matching logic |
| `llm_explainer.py` | Optional LLM-based explanation generator (Anthropic API) |
| `students.csv` / `businesses.csv` | Synthetic profile datasets |
| `ground_truth_matches.csv` | Known-good domain-matched pairs, for evaluation |
| `feedback_logs.csv` | Simulated + live user feedback on matches |
| `generate_dataset.py` | Script that generated the synthetic datasets |
| `demo_tfidf_pipeline.py` | Offline-friendly demo using TF-IDF instead of Sentence Transformers |
| `requirements.txt` | Python dependencies |
| `DEPLOYMENT.md` | Full deployment walkthrough (Streamlit Community Cloud, free) |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free
See `DEPLOYMENT.md`.
