#implement your retriever here
from typing import List, Union
from pathlib import Path
import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle

from baseline.retriever.retrieval_results import RetrievalResults
from utils.utils import chunk_text_recursive_character, chunk_text_fixed_size, read_file
from .retrieval_results import RetrievalResults

class Retriever:
    """
    A semantic document retriever using FAISS for similarity search and
    SentenceTransformers for generating embeddings.

    This class supports loading documents (PDF, TXT, or Markdown),
    chunking them into smaller segments, encoding them into vector embeddings,
    and indexing them using FAISS for fast similarity-based retrieval.

    Attributes:
        chunk_size (int): Number of characters per text chunk. (default=100)
        model (SentenceTransformer): Sentence embedding model from SentenceTransformers.
        embedding (List[np.ndarray]): List of embedding vectors corresponding to each chunk.
        index (faiss.Index): FAISS vector index for similarity search.
        id_to_chunk (dict): Dictionary mapping numerical IDs to text chunks.

    Example:
        ::
        from retriever import Retriever
        # Initialize
        retriever = Retriever(chunk_size=200)

        # Add documents
        retriever.add_documents(["doc1.pdf", "notes.txt"])

        # Search
        results = retriever.query("What is the main theme?", k=2)

        # Save/Load
        retriever.save("my_index")
        loaded_retriever = Retriever()
        loaded_retriever.load("my_index")

    """
    def __init__(self, chunk_size: int = 200, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", indexing_type: str = ""):
       self.chunk_size = chunk_size
       self.model = SentenceTransformer(model_name)
       self.embeddings = []
       self.index = None

       # Stores metadata per chunk: {id: {"text": str, "source": str}}
       self.id_to_chunk = {}

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks (without overlap).

        Args:
            text (str): The input text to be chunked.

        Returns:
            List[str]: List of text chunks with maximum length chunk_size
        """
        chunks = []
       # start = 0
      #  while start < len(text):
      #      end = min(len(text), start + self.chunk_size)
      #      chunks.append(text[start:end])
      #      start += self.chunk_size
        chunks = chunk_text_fixed_size(text, chunk_size = self.chunk_size)
        return chunks

    def add_documents(self, file_paths: List[Union[str, Path]]):
        """
         Process(Read, chunk and encode) and index documents from the given file paths for retrieval.

            1. Reads files and splits into chunks
            2. Generates embeddings using SentenceTransformer
            3. Creates/updates FAISS index
            4. Stores chunks in ID mapping dictionary

         Args:
             file_paths (List[Union[str, Path]]): List of document paths to add to the index.
         """
        all_chunks = []
        sources = []  # Track source path for each chunk
        for path in file_paths:
            raw_text = read_file(path)
            doc_chunks = self._chunk_text(raw_text)
            all_chunks.extend(doc_chunks)
            # Assign source path to each chunk from this document
            sources.extend([str(path)] * len(doc_chunks))

        #Convert each chunk of text into a vector using the SentenceTransformer model.
        embeddings = self.model.encode(all_chunks, convert_to_tensor=True)

        # Get the dimension of each embedding vector (e.g., 384 for all-MiniLM-L6-v2).
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.embeddings.extend(embeddings)

        #Inside a dictionary saving each chunk by assigning each to the next available integer key, will be useful for future retrival when there is a query performed
        next_id = len(self.id_to_chunk)  # Start ID for new chunks
        for idx, (chunk,source) in enumerate(zip(all_chunks, sources)):
            self.id_to_chunk[next_id + idx] = { # using len(self.id_to_chunk) to get the next integer, if the length is 3, then starting from 0, 1, 2 will be already used, then next integer 3 will be assigned.
                "chunk_text": chunk,
                "source": source,
            }

    def query(self, query: str, k: int = 3) -> RetrievalResults:
        """
        Retrieve the top-k most semantically similar chunks to the query.

        Args:
            query (str): The input search query text.
            k (int): Number of top results to return. (default=3)
        Returns:
            List[str]: List of top-k relevant text chunks, ordered by similarity
        """
        emb = self.model.encode([query], convert_to_numpy=True)

        # Search: D = cosine similarities, I = chunk IDs
        D, I = self.index.search(emb,k)
        #use the top k indices returned by FAISS search to fetch the actual chunks from the saved dictionary containing all the chunks

        # Prepare results with metadata and scores
        results = []
        for idx, score in zip(I[0], D[0]):
            metadata = self.id_to_chunk[idx]
            results.append({
                "chunk_text": metadata["chunk_text"],
                "source": metadata["source"],
                "similarity": float(score)
            })
        return RetrievalResults(results)

    def save(self, dir_path:  Union[str, Path] = "vector_index"):
        """
        Save the FAISS index and chunk metadata to disk.

        Saves:
            - FAISS index (faiss.index)
            - ID-to-chunk mapping (metadata.pkl)

        Args:
            dir_path (Union[str, Path]): Directory path to save the index and metadata.
        """
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "faiss.index"))
        with open(os.path.join(dir_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.id_to_chunk, f)

    def load(self, dir_path:  Union[str, Path]):
        """
        Load the FAISS index and chunk metadata from disk.

        Args:
            dir_path (Union[str, Path]): Directory path to load the index and metadata from.
        """
        self.index = faiss.read_index(os.path.join(dir_path, "faiss.index"))
        with open(os.path.join(dir_path, "metadata.pkl"), "rb") as f:
            self.id_to_chunk = pickle.load(f)