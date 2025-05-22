# RAG Project – Summer Semester 2025
A simple and extensible local Retrieval-Augmented Generation (RAG) system using FAISS and SentenceTransformers and T5. It supports `.txt`, `.md`, and `.pdf` files.

## 🚀 Features

- Chunking and vector indexing of documents
- Natural language query over local files
- Save/load FAISS index
- Supports `.txt`, `.md`, `.pdf`

## Overview
This repository hosts the code for a project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results, contains some test files.
- `utils/`: Helper functions shared across code.
- `data/`: Contains the data, mostly document files
- `homeworks/`: Contains all the homework independently. Files from this are also being incorporated in the baseline structure.

## Getting Started
1. Clone the repo
2. `cd baseline/`
3. Install dependencies: `pip install -r ../requirements.txt`
---

## 🛠 Installation
**Step 1**: download and access the project in your device.
```bash
git clone repo_link
cd ./NLProc-Proj-M-SS25
```

(Optional) **Step 2**: Create a virtual environment
```bash
python -m venv nlp_proj_ss_25
```

Activate the virtual environment
```bash
# On macOS/Linux:
source nlp_proj_hw_w3/bin/activate
# On Windows:
nlp_proj_ss_25\Scripts\activate
```

**Step 3**: Install dependencies
```bash
pip install -r requirements.txt
```

## Teams & Tracks
**Group**: `@Team Oneironauts`