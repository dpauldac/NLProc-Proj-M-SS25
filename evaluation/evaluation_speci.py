#implement your evaluation code here
from pathlib import Path
import random
import json
from transformers import GenerationConfig

from pipeline_tester_speci import PipelineTester
#from baseline.retriever import Retriever

from specialization.retriever_speci import RetrieverSpeci
from specialization import PipelineSpeci

#from baseline.generator import Generator
from baseline.pipeline import Pipeline

def getPaths(documents_base_path):
    #Tests
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
    index_path = "./vector_index_speci"
    all_document_paths = getPaths(documents_base_path)

    print(all_document_paths)

    if index_path.exists():
        print(f"Index exists at {index_path}, loading index...")
        retriever.load(str(index_path))
    else:
        print("Index does not exist. Creating new index...")
        retriever.add_documents(all_document_paths)
        retriever.save(str(index_path))
        retriever.load(str(index_path))

    # %%
    # Query variations
    queries = [
        "Apple iphone Q2 sales?",
        "How much revenue did apple make from MAC in 2024?"
        "How much was the Loss From Operations on Global Services for Boeing in 2023?" # 3,329
    ]
    # print(retriever.query("", k=2))
   # print(retriever.query("", k=2))

    for query in queries:
        k = random.randint(2, 4)
        matches = retriever.query( query, k)
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
    all_document_paths = getPaths(documents_base_path)

    print(all_document_paths)

    rag_pipeline = Pipeline(
        document_paths = all_document_paths,
        index_save_path="./vector_index_speci",
        generation_config=gen_config
    )

    answer = rag_pipeline.query("How much was the Loss From Operations on Global Services for Boeing in 2023?")
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

def single_query_test():
    # "What percentage of Apple’s net sales in 2024 came from direct distribution?"
    user_query = input("Enter the query: ")
    pipeline = PipelineSpeci(
        rebuild_index=False,
    )
    print("Generating answers...:")
    print(pipeline.query(user_query))
    #check the answer in the logs

def run_pipeline_tester_speci():
    documents_base_path = Path("../baseline/data/findoc_mini_samples_2")

    all_document_paths = []
    for file_path in documents_base_path.rglob('*'):
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            print(file_path)
            all_document_paths.append(file_path)

    pipeline = PipelineSpeci(
        document_paths=all_document_paths,
        index_save_path="./vector_index_speci",
        rebuild_index=True,
    )
    tester = PipelineTester(pipeline, "test/test_inputs_speci.json")
    test_results = tester.run_tests()

if __name__ == "__main__":
    single_query_test() # run to test a single query-answer generation
    #run_pipeline_tester_speci() #uncomment and run to test the pipeline