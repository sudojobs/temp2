# Xgent Agentic AI – Demo User Guide

This project is a local, offline-friendly assistant for Xcelium / UVM debug with:

- Jira + BGE similarity
- PDF QA with BGE
- UVM testcase training & retrieval
- Log triage
- CCF redundancy analysis
- Memory profiler assistant
- Guided debug script generator (YAML recipes)
- Learning mode (store resolved solutions)

## 1. Setup

Create venv and install:

```bash
pip install sentence-transformers torch PyPDF2 numpy pyyaml
# optional for real LLM:
pip install llama-cpp-python
```

Download `BAAI/bge-base-en-v1.5` to `models/bge-base-en-v1.5/` or let sentence-transformers download it.

Replace `models/llama-demo.gguf` with a real GGUF model and update `config.json` if needed.

## 2. Run

```bash
python server.py
```

Open:

```text
http://localhost:8099/
```

Login: `xgent / xgent123`.

Tabs are self-describing and mirror your requested recipes.
