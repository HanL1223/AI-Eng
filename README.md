# STTM Assistant 📊

A Retrieval-Augmented Generation (RAG) chatbot for querying Sigma Healthcare's data warehouse documentation. Ask natural language questions about tables, columns, mappings, and data pipelines — get accurate, source-cited answers.

## What This Does

The STTM (Source-to-Target Mapping) Assistant reads your data warehouse documentation (Excel workbooks, text files, markdown) and answers questions like:

- "What is the grain of DIM_PRODUCT?"
- "Where does StoreKey come from?"
- "Which tables use PDB08 as a source system?"
- "What transformations happen between Bronze and Gold for DIM_STORE?"

It retrieves relevant documentation chunks using vector similarity search, then sends them to Claude as context for generating accurate, grounded answers.

## Architecture

```
                    ┌──────────────┐
                    │  STTM Excel  │
                    │  .txt / .md  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ sttm_loader  │  Smart Excel parser
                    │  + chunker   │  (understands STTM structure)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   ChromaDB   │  Vector store
                    │  (embedded)  │  (cosine similarity search)
                    └──────┬───────┘
                           │
               ┌───────────┼───────────┐
               │           │           │
        ┌──────▼──┐  ┌─────▼────┐  ┌──▼──────────┐
        │  app.py │  │ eval.py  │  │ experiment  │
        │Streamlit│  │  20-Q    │  │  runner.py  │
        │  Chat   │  │  bench   │  │ auto-tuning │
        └─────────┘  └──────────┘  └─────────────┘
```

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key ([get one here](https://console.anthropic.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/sttm-assistant.git
cd sttm-assistant

# Set your API key
# Windows PowerShell:
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# macOS/Linux:
export ANTHROPIC_API_KEY="sk-ant-..."

# Or create a .env file:
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### Add Your Documents

Place your STTM Excel workbooks, `.txt`, or `.md` files in the `docs/` folder:

```
docs/
├── STTM.xlsx            # Your Source-to-Target Mapping workbook
├── architecture.md      # Optional: additional documentation
└── metadata_standards.txt
```

### Run the Chat UI

```bash
uv run streamlit run app.py
```

Opens a web browser at `http://localhost:8501` with the chat interface.

### Run from Terminal (No UI)

```bash
uv run python rag.py
```

### Run the Evaluation Suite

```bash
# Keyword scoring only (fast, free)
uv run python eval.py --tag baseline

# With LLM-as-judge (smarter, costs API $)
uv run python eval.py --tag baseline --llm-judge

# Compare two runs
uv run python eval.py --compare eval_results/run_a.csv eval_results/run_b.csv
```

### Run Automated Experiments

```bash
# Test different chunk sizes
uv run python experiment_runner.py --experiment chunk_size

# Test different TOP_K values
uv run python experiment_runner.py --experiment top_k

# Run all experiments
uv run python experiment_runner.py --experiment all
```

## Project Structure

```
sttm-assistant/
├── rag.py                  # Core RAG pipeline (load → chunk → embed → retrieve → generate)
├── sttm_loader.py          # Smart STTM Excel workbook parser
├── app.py                  # Streamlit chat interface
├── eval.py                 # 20-question evaluation framework
├── experiment_runner.py    # Automated parameter tuning
├── eval_questions.csv      # Benchmark questions with expected answers
├── pyproject.toml          # Python dependencies
├── .env                    # API key (not committed)
├── .gitignore              # Excludes secrets, caches, results
├── docs/                   # Your STTM documentation files
├── chroma_db/              # Vector store (auto-generated)
└── eval_results/           # Experiment outputs (auto-generated)
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| ChromaDB (embedded) | Zero infrastructure — runs in-process, no server needed |
| 800-char chunks with 100-char overlap | Optimized via experiment_runner.py across 5 chunk sizes |
| STTM-aware chunking | Each Excel entity → 2 docs (summary + column mapping) for better retrieval |
| Metadata filtering | `table_name` and `doc_type` filters reduce search space per query |
| Cross-entity detection | Questions mentioning multiple tables get wider retrieval (TOP_K doubled) |

## Tech Stack

- **LLM**: Anthropic Claude (claude-sonnet-4-5-20250929)
- **Vector Store**: ChromaDB (embedded, default embeddings)
- **UI**: Streamlit
- **Language**: Python 3.12
- **Package Manager**: uv

## Domain Context

This chatbot is built for Sigma Healthcare's Snowflake data warehouse:

- **Source Systems**: SAP (CDS Views), MyPOS, PDB08, PDB15, TDB08AX2012
- **Extraction**: Azure Data Factory → ADLS Gen2
- **Layers**: Bronze → Platinum (dbt) → Gold (Star Schema)
- **Reporting**: Power BI reads from Gold layer
- **Tables**: FACT_*, DIM_*, BRIDGE_* naming conventions
- **Keys**: SK_*_KEY (surrogate), FK_* (foreign), BK_* (business)

## Evaluation Results

The system is measured against a 20-question benchmark spanning three categories:

| Category | Description | Example |
|----------|-------------|---------|
| simple_lookup | Single-table factual questions | "What is the grain of DIM_STORE?" |
| cross_entity | Multi-table relationship questions | "Which dimensions does FACT_INVENTORY reference?" |
| edge_case | Questions the system should decline | "What is the SLA for FACT_SALES_ORDER refresh?" |

## What I Learned

Built over 3 weeks as a hands-on AI engineering learning project:

1. **Week 1**: Core RAG loop — embedding, retrieval, generation. Learned that chunking strategy matters more than model choice for retrieval quality.
2. **Week 2**: Quality tuning — automated experiments proved 800-char chunks with TOP_K=3 optimal for this domain. Built eval framework with keyword scoring + LLM-as-judge.
3. **Week 3**: Production UI — Streamlit's re-run model requires careful state management. `@st.cache_resource` is essential for expensive initialization. Citations build user trust.

## Roadmap

- [x] Phase 1: RAG Foundations (Weeks 1-6)
- [ ] Phase 2: Agents & Tool Use (Weeks 7-9)
- [ ] Phase 3: Multi-Agent & Eval at Scale (Weeks 10-12)
- [ ] Phase 4: Fine-Tuning with LoRA/SFT/DPO (Weeks 13-16)
- [ ] Phase 5: Production Deployment (Week 17+)

## License

This project is for educational purposes. The STTM data is proprietary to Sigma Healthcare and not included in the repository.