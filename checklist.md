# 📋 Progression Checklist

Track your progress. Each task teaches a specific AI engineering skill.

## Week 1: Get It Working
- [ ] Set up Python venv + install dependencies
- [ ] Get Anthropic API key from console.anthropic.com
- [ ] Run `rag.py` with sample docs → get first answer
- [ ] Turn on debug mode (`debug`) → see which chunks are retrieved
- [ ] Swap in 3-5 real documents from your work (STTM exports, metadata docs, etc.)
- [ ] Ask 10 questions you actually need answered → note which ones work and which fail

## Week 1 Skill Unlocked: You understand the RAG loop (embed → store → retrieve → generate)


## Week 2: Improve Quality
- [ ] Experiment with CHUNK_SIZE: try 200, 500, 1000 → which gives better answers?
- [ ] Experiment with TOP_K: try 1, 3, 5 → too few = missing info, too many = noise
- [ ] Improve the system prompt: add specifics about your domain (Sigma, Snowflake, SAP terminology)
- [ ] Add metadata to chunks: tag each chunk with table_type, workstream, data_layer
- [ ] Use metadata filtering in retrieval: `collection.query(where={"data_layer": "platinum"})`
- [ ] Create a test set: write 20 question-answer pairs, track accuracy manually
- [ ] Try loading Excel files: `pip install openpyxl`, read tabs as documents

## Week 2 Skill Unlocked: Prompt engineering + chunking strategy + metadata filtering


## Week 3: Add a UI + Share It
- [ ] Install Streamlit: `pip install streamlit`
- [ ] Build a basic chat UI (Streamlit has a chat template — google "streamlit chat example")
- [ ] Add file upload: let users drag-and-drop docs through the UI
- [ ] Add source citations: show which file each answer came from
- [ ] Deploy locally for your team to try (streamlit run app.py)
- [ ] Push to GitHub with a proper README
- [ ] Write a 1-paragraph summary of what you learned → post on LinkedIn

## Week 3 Skill Unlocked: Full-stack AI app + deployment + portfolio piece


## Bonus Challenges (when you're ready)
- [ ] Replace ChromaDB default embeddings with Voyage AI or OpenAI embeddings
- [ ] Add conversation memory (multi-turn chat)
- [ ] Handle PDF ingestion: `pip install pymupdf`
- [ ] Add a reranking step (retrieve 10, rerank to top 3)
- [ ] Log every query + response + latency to a JSON file (baby eval pipeline)
- [ ] Compare Claude Haiku vs Sonnet: which is better for your use case? Measure cost vs quality.


## Key Metrics to Track
As you iterate, keep a simple spreadsheet with:
| Question | Expected Answer | Actual Answer | Correct? | Chunks Retrieved | Notes |
|----------|----------------|---------------|----------|-----------------|-------|

This is the seed of your eval pipeline (Phase 3 of the roadmap).