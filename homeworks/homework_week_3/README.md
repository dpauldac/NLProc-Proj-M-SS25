# Document Retriever based on FAISS and SentenceTransformers
A simple and extensible local document retriever using FAISS and SentenceTransformers. It supports `.txt`, `.md`, and `.pdf` files.

---

## 🚀 Features

- Chunking and vector indexing of documents
- Natural language query over local files
- Save/load FAISS index
- Supports `.txt`, `.md`, `.pdf`

---

## 🛠 Installation
**Step 1**: download and access the project in your device.
```bash
git clone repo_link
cd homeworks/homework_week_3
```

(Optional) **Step 2**: Create a virtual environment
```bash
python -m venv nlp_proj_hw_w3
```

Activate the virtual environment
```bash
# On macOS/Linux:
source nlp_proj_hw_w3/bin/activate
# On Windows:
nlp_proj_hw_w3\Scripts\activate
```

**Step 3**: Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to use the **Retriever class**?
### Example
```python
# file: main.py
from pathlib import Path
from retriever import Retriever

if __name__ == '__main__':
    #create the object
    retriever = Retriever()
    
    # add documents
    documents_base_path = Path("documents")
    retriever.add_documents([
        documents_base_path / "demo.txt",
        documents_base_path / "demo.md",
        documents_base_path / "demo.pdf"
        # add more or less
    ])

    # persist/save the indexing in the memory for future use
    retriever.save("vector_index")

    # run a query
    print(retriever.query("When was the QuantumLink v2.0 launched?"))

    # create another object
    new_retriever = Retriever()

    # load the previously persisted indexing for search
    new_retriever.load("vector_index")

    # run more queries
    print(new_retriever.query("Disk space required to install visual studio", k=2))
    print(new_retriever.query("How many mb of disk space required to install visual studio", k=3))
```
run the `main.py` using the following command 
```bash
python -m main 
```

---

## Run unit test file
```bash
python -m pytest test_retriever.py
```
**Note**: Some test cases might fail, as the current version of the retriever approach has certain limitations or shortcomings.

---

## Frequent Error and Solution:
1. pytest : The term 'pytest' is not recognized as the name of a cmdlet, function, script file, or operable program.
   * **solution**: use terminal or powershell for windows and run `python -m pip install pytest`. Then it should be possible to run the unit test file.
---
**Group**: `@Team Oneironauts`