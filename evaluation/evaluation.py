#implement your evaluation code here
from pathlib import Path
import random
import json
from transformers import GenerationConfig
from baseline.retriever import Retriever
from baseline.generator import Generator
from baseline.pipeline import Pipeline

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

def rag_test():
    # Initialize components
    retriever = Retriever()
    generator = Generator(model_name="google/flan-t5-base")  # CPU-friendly model

    # Load test documents
    documents_base_path = Path("../baseline/data")
    test_docs = [
        documents_base_path / "demo.txt",
        documents_base_path / "demo.md",
        documents_base_path / "demo.pdf"
    ]

    # Build index
    retriever.add_documents(test_docs)
    retriever.save("sentence_embeddings_index")

    # Test queries with variations
    test_queries = [
        "When was the QuantumLink v2.0 launched?",
        "Disk space required to install visual studio",
        "How many mb of disk space required to install visual studio",
        "What are the system requirements for development tools?"  # New test case
    ]

    # Evaluation loop
    for query in test_queries:
        print(f"\n{'=' * 50}\nQuery: {query}\n{'-' * 50}")

        # Retrieve context
        k = random.randint(2, 4)
        contexts = retriever.query(query, k)

        # Generate answer
        prompt,answer = generator.generate_answer(query, contexts)

        # Display results
        print(f"\nTop {k} Context Chunks:")
        for i, ctx in enumerate(contexts, 1):
            print(f"[Chunk {i}]\n {ctx[:100]}...")  # Show first 100 chars

        print(f"{'-' * 50}\n=> Generated Answer: {answer}\n")
        print("=" * 50 + "\n")

def rag_pipeline_test():

    gen_config = GenerationConfig(
    #    max_length=256,
   #     temperature=0.3,  # a bit of creativity
    #    num_beams=2,  # Enables beam search. More beams = more exploration for best output, but slower.
    #    early_stopping=True,  # Prevents unnecessarily long outputs with beam search.
    #    do_sample=True  # Randomly selects tokens based on probabilities.
    )

    # Load test documents
    documents_base_path = Path("../baseline/data")
    doc_paths = [
        documents_base_path / "demo.txt",
        documents_base_path / "demo.md",
        documents_base_path / "demo.pdf"
    ]

    rag_pipeline = Pipeline(
        document_paths = doc_paths,
        index_save_path="./sentence_embeddings_index",
        generation_config=gen_config
    )

    answer = rag_pipeline.query("When was the QuantumLink v2.0 launched?")
    print("Answer:", answer)

    # Test queries with variations
    test_queries = [
        "When was the QuantumLink v2.0 launched?",
        "Disk space required to install visual studio",
        "How many mb of disk space required to install visual studio",
        "What are the system requirements for development tools?"  # New test case
    ]

    # Evaluation loop
    for query in test_queries:
        print(f"\n{'=' * 50}\nQuery: {query}\n{'-' * 50}")

        answer = rag_pipeline.query(query)

        # Display results
        print(f"=> Generated Answer: {answer}\n")
        print("=" * 50 + "\n")


#def test_script():
#    path = Path("testing/test_inputs.json")
   # test_cases = with open(path, "r") as f:
   #     return json.load(f)

"""
    results = {}

    for idx, test in enumerate(self.test_cases):
        answer = self.pipeline.query(test["question"])
        contexts = self.pipeline.retriever.query(test["question"])

        results[f"test_{idx}"] = {
            "question": test["question"],
            "answer_received": answer,
            "answer_valid": bool(answer.strip()),
            "grounding_check": self._check_grounding(answer, contexts),
            "expected_terms_present": all(
                term in " ".join(contexts).lower()
                for term in test.get("expected_context_terms", [])
            )
        }
"""

if __name__ == "__main__":
    rag_pipeline_test()