# Deployment Guide — AI Semantic Matching Engine

This app is built so it can be deployed **entirely for free** on Streamlit Community Cloud.

## Files you need in your repo
```
app.py
matching_engine.py
students.csv
businesses.csv
ground_truth_matches.csv
feedback_logs.csv
requirements.txt
```

## Step 1 — Push to GitHub
1. Create a new GitHub repo (public or private).
2. Add all the files above to the repo root.
3. Commit and push.

## Step 2 — Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"**.
3. Select your repo, branch (`main`), and set the main file path to `app.py`.
4. (Optional, for AI-generated explanations) In your app's **Settings → Secrets**,
   add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
   Get a key at https://console.anthropic.com. Without this, the app still works
   fully — users just won't see the "Use AI-generated explanations" toggle populate
   automatically (they can still paste in their own key at runtime if you leave that
   input visible).
5. Click **Deploy**.
6. First deploy will take a few minutes — it needs to download the Sentence Transformer
   model (~90MB) and install dependencies. Subsequent restarts are much faster.

That's it — you'll get a free public URL like:
`https://your-app-name.streamlit.app`

## Alternative free hosting options (if you outgrow Streamlit Cloud's limits)
- **Hugging Face Spaces** (free tier, supports Streamlit and Gradio natively, and since
  it's HF's own infrastructure, model downloads are instant — no cold-start delay for
  Sentence Transformers)
- **Render** (free web service tier — note: free tier spins down after inactivity,
  causing a ~30s cold start on the next visit)

## Notes on AI-generated explanations
- This feature calls the Anthropic API (`claude-haiku-4-5-20251001` by default —
  chosen for low cost/latency on this simple task) and is **off by default** since
  it costs API credits per new match explained.
- Explanations are cached to `llm_explanation_cache.json` by student+business pair,
  so re-viewing the same match never re-triggers a paid API call.
- If the API call fails for any reason (bad key, rate limit, no network), the app
  automatically falls back to the free rule-based explanation — the feature never
  breaks the app, it just silently degrades.

## Notes on the feedback loop
- `feedback_logs.csv` will accumulate new rows as users submit feedback through the app.
- On Streamlit Community Cloud, the filesystem is **ephemeral** — feedback written to
  the CSV will NOT persist across app restarts/redeploys. For a real production feedback
  loop, swap the `append_feedback()` function in `app.py` to write to a persistent store:
  - Free options: a free-tier Supabase or Neon Postgres database, or a Google Sheet via
    the Google Sheets API.
  - This is a small, contained change — only `append_feedback()` and the data-loading
    function for feedback need to be updated; the matching logic is unaffected.

## Scaling notes
- Current setup re-embeds all profiles on every cold start. Fine for hundreds of
  profiles (current dataset: 500 students / 200 businesses). If you scale into the
  thousands+, precompute embeddings offline (`matching_engine.py` already supports
  `save_embeddings()` / `load_embeddings()`) and load the cached `.npz` file instead
  of recomputing at runtime.
- If profile count grows large (10,000+), consider a proper vector database
  (Qdrant, Weaviate, or pgvector) instead of in-memory cosine similarity — all of these
  have free tiers suitable for a project at this stage.
