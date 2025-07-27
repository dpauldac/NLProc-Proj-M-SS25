#implement your evaluation code here
from pathlib import Path
import random
import json

from huggingface_hub import login
from transformers import GenerationConfig

from pipeline_tester_speci import PipelineTesterSpeci
#from baseline.retriever import Retriever

from baseline.generator import Generator
from specialization.retriever_speci import RetrieverSpeci
from specialization.generator_speci import GeneratorSpeci
from specialization.generator_groq import GeneratorGroq
from specialization import PipelineSpeci

#from baseline.generator import Generator
from baseline.pipeline import Pipeline
import os
from dotenv import load_dotenv

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

    if Path(index_path).exists():
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
    documents_base_path = Path("../baseline/data/findoc_xsm_samples")
    all_document_paths = getPaths(documents_base_path)

    print(all_document_paths)

    rag_pipeline = Pipeline(
        document_paths = all_document_paths,
        index_save_path="./vector_index_speci_xsm",
        generation_config=gen_config
    )

    answer = rag_pipeline.query("What is the total current assets of Apple as of June 29, 2024?")
    print("Answer:", answer)

    # Test queries with variations
    queries = [
        "Apple iphone Q2 sales?",
        "How much revenue did apple make from MAC in 2024?"
        "What is the total current assets of Apple as of June 29, 2026?" # halucination
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
    print(pipeline.query(user_query, 8))
    #check the answer in the logs

def rag_pipeline_speci_test():
    gen_config = GenerationConfig(
    #    max_length=256,
   #     temperature=0.3,  # a bit of creativity
    #    num_beams=2,  # Enables beam search. More beams = more exploration for best output, but slower.
    #    early_stopping=True,  # Prevents unnecessarily long outputs with beam search.
    #    do_sample=True  # Randomly selects tokens based on probabilities.
    )

    #Tests
    retriever = RetrieverSpeci()
    documents_base_path = Path("../baseline/data/findoc_xsm_samples")
    all_document_paths = getPaths(documents_base_path)

    print(all_document_paths)

    rag_pipeline = PipelineSpeci(
        document_paths = all_document_paths,
        index_save_path="./vector_index_speci_xsm",
        generation_config=gen_config,
        rebuild_index = False,
    )

    query = "What is the total current assets of Apple as of June 29, 2024?"  # hallucination, in current retreived files it is not included
    query = "What is the total current assets of Apple as of  March 29,2025?"
    answer = rag_pipeline.query(query, k=5)
    print(f"\n{'=' * 50}\nQuery: {query}\n{'-' * 50}")
    print("Answer:", answer)

    # Test queries with variations
    queries = [
        "Apple iphone Q2 sales?",
        "Which of the following are not products from Apple? a) iPhone b) Mac c) Vision Pro d) AWS",
        "Which of the following are products from Apple? a) iPhone b) Mac c) Vision Pro d) Both a and b e) Both a, b and c",
        "How much revenue did apple make from MAC in 2024?",
        "What is the total current assets of Apple as of June 29, 2026?" # hallucination
    ]

    # Evaluation loop
    for query in queries:
        print(f"\n{'=' * 50}\nQuery: {query}\n{'-' * 50}")

        answer = rag_pipeline.query(query)

        # Display results
        print(f"=> Generated Answer: {answer}\n")
        print("=" * 50 + "\n")

def run_pipeline_tester_speci():
    documents_base_path = Path("../baseline/data/findoc_xsm_samples")

    all_document_paths = []
    for file_path in documents_base_path.rglob('*'):
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            print(file_path)
            all_document_paths.append(file_path)

    pipeline = PipelineSpeci(
        document_paths=all_document_paths,
        index_save_path="./vector_index_speci_xsm",
        rebuild_index=False,
    )
    tester = PipelineTesterSpeci(pipeline, "test/test_input_speci_new.json")
    test_results = tester.run_tests()

def retriever_test2():
    #Tests
    retriever = RetrieverSpeci()
    documents_base_path = Path("../baseline/data/findoc_xmini_samples")
    index_path = "./vector_index_speci_xmini"
    all_document_paths = getPaths(documents_base_path)

    print(all_document_paths)

    if Path(index_path).exists():
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


def retriever_test3():
    #Tests
    retriever = RetrieverSpeci()
    documents_base_path = Path("../baseline/data/findoc_xsm_samples")
    index_path = "./vector_index_speci_xsm"
    all_document_paths = getPaths(documents_base_path)

    print(all_document_paths)

    if Path(index_path).exists():
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
        "How much revenue did apple make from MAC in 2024?",
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


def retriever_retreved_chunk_tester():
    #Tests
    retriever = RetrieverSpeci()
    index_path = "./vector_index_speci_xmini"

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

def generatorTest():
    # Tests
    retriever = RetrieverSpeci()
    index_path = "./vector_index_speci_xmini"
    # Set a new environment variable

    if Path(index_path).exists():
        print(f"Index exists at {index_path}, loading index...")
        retriever.load(str(index_path))
    else:
        print("Index does not exist.")

    question = "What is the total current assets of Apple as of June 29, 2024?" # hallucination, in current retreived files it is not included
    question = "What is the total current assets of Apple as of  March 29,2025?"
    results = retriever.query(question, 5)
    #print(results)
    #contexts = results.chunk_text
    print(results.chunk_text)
    print(results.score)
    print(results.source)
    print(results.pages)


    #generator = Generator(model_name="google/flan-t5-large")
    #generator = GeneratorSpeci(model_name="google/flan-t5-xxl")
    # Load model directly
    # generator = Generator(model_name="t5-small")
    generator = GeneratorGroq()

    contexts = results.chunk_text
    answer, prompt = generator.generate_answer(question, contexts)
    print(f"{'-' * 50}\nPrompt:\n{prompt}\n\n=> Generated Answer: {answer}\n")


if __name__ == "__main__":
    #documents_base_path = "../baseline/data/findoc_xmini_samples"
    #single_query_test() # run to test a single query-answer generation
    run_pipeline_tester_speci() #uncomment and run to test the pipeline
    #generatorTest()
    #retriever_test2()
    #retriever_test3()
    #rag_pipeline_speci_test()