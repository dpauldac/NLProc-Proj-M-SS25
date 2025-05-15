import os
import shutil
import pytest
from retriever import Retriever

TEST_DIR = "test_index"
TEST_FILE = "test_doc.txt"
TEST_CONTENT = (
    "Python is a high-level programming language.\n"
    "It supports multiple programming paradigms.\n"
    "It is widely used in data science, web development, and automation.\n"
    "Its syntax is clear and readable."
)

@pytest.fixture(scope="module")
def retriever_instance():
    with open(TEST_FILE, "w") as f:
        f.write(TEST_CONTENT)

    retriever = Retriever(chunk_size=50)
    retriever.add_documents([TEST_FILE])
    yield retriever

    # Cleanup
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_chunking_and_indexing(retriever_instance):
    assert retriever_instance.index.ntotal > 0
    assert len(retriever_instance.id_to_chunk) > 0

# Test a sample query
def test_query_response(retriever_instance):
    result = retriever_instance.query("What is Python?")
    assert any("Python" in chunk for chunk in result)

# Test for querying an exact match
def test_query_exact_match(retriever_instance):
    result = retriever_instance.query("Python is a high-level programming language.")
    assert any("Python is a high-level programming language." in chunk for chunk in result), \
        "Query should return the exact matching chunk"

# Test for querying a term that should return multiple relevant chunks
def test_query_multiple_results(retriever_instance):
    result = retriever_instance.query("programming")
    assert len(result) > 1, "Query should return multiple chunks containing 'programming'"
    assert all("programming" in chunk for chunk in result), "All returned chunks should contain 'programming'"

# Test for querying a term that is in the document but not in the chunk exactly
def test_query_partial_match(retriever_instance):
    result = retriever_instance.query("Python programming")
    # Since it's a chunk-based system, it might break across chunks
    assert any("Python" in chunk and "programming" in chunk for chunk in result), \
        "Query should return chunks containing both 'Python' and 'programming'"

# Test for querying a term that isn't present in the document
def test_query_no_match(retriever_instance):
    result = retriever_instance.query("JavaScript")
    assert len(result) == 0, "Query for a term not in the document should return an empty result"

# Test for querying a term with multiple word match (should handle both words in one chunk)
def test_query_multiple_words_in_one_chunk(retriever_instance):
    result = retriever_instance.query("multiple programming")
    assert len(result) == 1, "Query should return only one chunk containing 'multiple programming'"
    assert "multiple programming" in result[0], "Returned chunk should contain 'multiple programming'"

# Test for querying an empty string (edge case)
def test_query_empty_string(retriever_instance):
    result = retriever_instance.query("")
    assert len(result) == 0, "Querying an empty string should return no results"

# Test for querying a string with special characters or punctuation
def test_query_special_characters(retriever_instance):
    result = retriever_instance.query("Python, programming?")
    assert any("Python" in chunk and "programming" in chunk for chunk in result), \
        "Query should return chunks containing 'Python' and 'programming' even with special characters"

# Test for querying a substring that spans across chunks
def test_query_spanning_chunks(retriever_instance):
    result = retriever_instance.query("clear and readable")
    assert len(result) > 0, "Querying a substring that spans chunks should return results"
    assert all("clear" in chunk and "readable" in chunk for chunk in result), \
        "Returned chunks should contain the entire substring 'clear and readable'"

# Test for querying case-insensitive match
def test_query_case_insensitive(retriever_instance):
    result = retriever_instance.query("PYTHON")
    assert any("Python" in chunk for chunk in result), "Query should match 'Python' regardless of case"


def test_save_and_load(retriever_instance):
    retriever_instance.save(TEST_DIR)

    new_instance = Retriever()
    new_instance.load(TEST_DIR)

    result = new_instance.query("What is Python?")
    assert any("Python" in chunk for chunk in result)
