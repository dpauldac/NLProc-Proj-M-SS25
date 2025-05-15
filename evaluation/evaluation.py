#implement your evaluation code here
from pathlib import Path
import random
from baseline.retriever import Retriever

def retriever_test():
    #Tests
    retriever = Retriever()
    documents_base_path = Path("../baseline/data")
    retriever.add_documents([
        documents_base_path / "demo.txt",
        documents_base_path / "demo.md",
        documents_base_path / "demo.pdf"
    ])
    retriever.save("sentence_embeddings_index")

    new_retriever = Retriever()
    new_retriever.load("sentence_embeddings_index")

    # %%
    # Query variations
    queries = [
        "When was the QuantumLink v2.0 launched?"
        "Disk space required to install visual studio",
        "How many mb of disk space required to install visual studio",
    ]
    # print(new_retriever.query("", k=2))
   # print(new_retriever.query("", k=2))

    for query in queries:
        k = random.randint(2, 4)
        matches = new_retriever.query( query, k)
        print(f"\nQuery: '{query}'\nTop matches:")
        for count, match in enumerate(matches):
            print(f"__{count}__\n {match}")
        print("___________________________")

retriever_test()