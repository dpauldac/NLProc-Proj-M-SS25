#implement your evaluation code here
from pathlib import Path
import random
import json
from transformers import GenerationConfig
#from baseline.retriever import Retriever

from specialization.retriever_speci import RetrieverSpeci
from specialization import PipelineSpeci

#from baseline.generator import Generator
from baseline.pipeline import Pipeline

def getPaths():
    #Tests
    documents_base_path = Path("../baseline/data/findoc_mini_samples_2")

    all_document_paths = []
    for file_path in documents_base_path.rglob('*'):
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            print(file_path)
            all_document_paths.append(file_path)
    return all_document_paths

def retriever_test():
    #Tests
    retriever = RetrieverSpeci()
    documents_base_path = Path("../baseline/data/findoc_mini_samples_2")

    all_document_paths = []
    for file_path in documents_base_path.rglob('*'):
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            print(file_path)
            all_document_paths.append(file_path)

    print(all_document_paths)
    retriever.add_documents(all_document_paths)
    retriever.save("sentence_embeddings_index_speci")

    new_retriever = RetrieverSpeci()
    new_retriever.load("sentence_embeddings_index_speci")

    # %%
    # Query variations
    queries = [
        "Apple iphone Q2 sales?",
        "How much revenue did apple make from MAC in 2024?"
        "How much was the Loss From Operations on Global Services for Boeing in 2023?" # 3,329
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


def rag_pipeline_test():

    gen_config = GenerationConfig(
    #    max_length=256,
   #     temperature=0.3,  # a bit of creativity
    #    num_beams=2,  # Enables beam search. More beams = more exploration for best output, but slower.
    #    early_stopping=True,  # Prevents unnecessarily long outputs with beam search.
    #    do_sample=True  # Randomly selects tokens based on probabilities.
    )

    #Tests
    retriever = RetrieverSpeci()
    documents_base_path = Path("../baseline/data/findoc_mini_samples")

    all_document_paths = []
    for file_path in documents_base_path.rglob('*'):
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            print(file_path)
            all_document_paths.append(file_path)

    print(all_document_paths)

    rag_pipeline = Pipeline(
        document_paths = all_document_paths,
        index_save_path="./sentence_embeddings_index",
        generation_config=gen_config
    )

    answer = rag_pipeline.query("When was the QuantumLink v2.0 launched?")
    print("Answer:", answer)

    # Test queries with variations
    queries = [
        "Apple iphone Q2 sales?",
        "How much revenue did apple make from MAC in 2024?"
        "How much was the Loss From Operations on Global Services for Boeing in 2023?" # 3,329
    ]

    # Evaluation loop
    for query in queries:
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
"""
def testSampleCode():
    documents_base_path = Path("../baseline/data/findocs")

    all_document_paths = []
    for file_path in documents_base_path.rglob('*'):
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            # You might want to add a filter here for specific file extensions
            # For example, to only include .txt, .md, .pdf:
            # if file_path.suffix in ['.txt', '.md', '.pdf']:
            print(file_path)
            all_document_paths.append(file_path)
 """

def single_query_test():
    # "What percentage of Apple’s net sales in 2024 came from direct distribution?"
    user_query = input("Enter the query: ")
    pipeline = PipelineSpeci(
        rebuild_index=False,
    )
    pipeline.query(user_query)
    #check the answer in the logs

if __name__ == "__main__":
    single_query_test()