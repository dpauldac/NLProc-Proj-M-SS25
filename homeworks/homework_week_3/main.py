from pathlib import Path
from retriever import Retriever

if __name__ == '__main__':
    #Tests
    retriever = Retriever()
    documents_base_path = Path("documents")
    retriever.add_documents([
        documents_base_path / "demo.txt",
        documents_base_path / "demo.md",
        documents_base_path / "demo.pdf"
    ])
    print(retriever.query("When was the QuantumLink v2.0 launched?"))
    retriever.save("sentence_embeddings_index")

    new_retriever = Retriever()
    new_retriever.load("sentence_embeddings_index")
    print(new_retriever.query("Disk space required to install visual studio", k=2))
    print(new_retriever.query("How many mb of disk space required to install visual studio", k=2))