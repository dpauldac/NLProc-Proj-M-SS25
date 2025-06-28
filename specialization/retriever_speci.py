#implement your retriever here
from typing import List, Union
from pathlib import Path
import os
import faiss
import fitz  # For PDF handling (pip install pymupdf)
from sentence_transformers import SentenceTransformer
import pickle
from utils.utils import chunk_text_recursive_character, chunk_text_fixed_size
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from utils.ner_utils import extract_org_ner
from utils.utils import read_file

class RetrieverSpeci:
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
        from retriever_speci import RetrieverSpeci
        # Initialize
        retriever = RetrieverSpeci(chunk_size=200)

        # Add documents
        retriever.add_documents(["doc1.pdf", "notes.txt"])

        # Search
        results = retriever.query("What is the main theme?", k=2)

        # Save/Load
        retriever.save("my_index")
        loaded_retriever = Retriever()
        loaded_retriever.load("my_index")

    """
    def __init__(self, chunk_size: int = 200, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 320):
       self.chunk_size = chunk_size
       self.model_name = model_name
       self.max_tokens = max_tokens
       self.model = SentenceTransformer(self.model_name)
       self.tokenizer =  HuggingFaceTokenizer(
           tokenizer=AutoTokenizer.from_pretrained(self.model_name),
           max_tokens=self.max_tokens ,  # optional, by default derived from `tokenizer` for HF case
        )
       self.chunker = HybridChunker(
           tokenizer= self.tokenizer,
           merge_peers = True,  # optional, defaults to True
           #max_tokens = self.max_tokens  # optional, default resolved from tokenizer
       )
       self.converter = DocumentConverter()
       self.embeddings = []
       self.index = None

       # Stores metadata per chunk: {id: {"text": str, "source": str}}
       self.id_to_chunk = {}

    def chunk_doc(self, doc_path: Union[str, Path]):
        """
        Converts all supported files in a directory to Docling documents
        and then chunks them.

        Args:
            directory_path (str): The path to the directory containing the documents.
            max_tokens (int): The maximum number of tokens per chunk.
        """
        result = self.converter.convert(doc_path)
        doc = result.document
        chunk_iter = self.chunker.chunk(dl_doc=doc)
        cur_doc_chunks = []
        for chunk in chunk_iter:
            enriched_text = self.chunker.contextualize(chunk=chunk)
            cur_doc_chunks.append(enriched_text)
        #print(cur_doc_chunks)
        print(f"#########Processed and chunked: {doc_path}######")
        return cur_doc_chunks

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
        for path in file_paths:
            cur_doc_chunks = self.chunk_doc(path)
            raw_text = read_file(path, max_pages=2)
            ner_data = extract_org_ner(raw_text)
            cur_org_name = ner_data[0][0] if ner_data and ner_data[0] else ""  # first detected org entity

            # Attach org name to each chunk
            if cur_org_name:
                cur_doc_chunks = [f"[ORG: {cur_org_name}] {chunk}" for chunk in cur_doc_chunks]

            all_chunks.extend(cur_doc_chunks)

        # Convert chunks to embeddings
        if not all_chunks:
            print("No valid chunks to process")
            return

        #Convert each chunk of text into a vector using the SentenceTransformer model.
        embeddings = self.model.encode(all_chunks, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy().astype('float32')

        #Get the dimension of each embedding vector (384 for all-MiniLM-L6-v2).
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.embeddings.extend(embeddings)

        #Inside a dictionary saving each chunk by assigning each to the next available integer key, will be useful for future retrival when there is a query performed
        for idx, chunk in enumerate(all_chunks):
            self.id_to_chunk[len(self.id_to_chunk)] = chunk # using len(self.id_to_chunk) to get the next integer, if the length is 3, then starting from 0, 1, 2 will be already used, then next integer 3 will be assigned.

    def query(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the top-k most semantically similar chunks to the query.

        Args:
            query (str): The input search query text.
            k (int): Number of top results to return. (default=3)
        Returns:
            List[str]: List of top-k relevant text chunks, ordered by similarity
        """
        if self.index is None:
            return []

        emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        D, I = self.index.search(emb,k)
        #use the top k indices returned by FAISS search to fetch the actual chunks from the saved dictionary containing all the chunks
        return [self.id_to_chunk[i] for i in I[0]]

    def save(self, dir_path:  Union[str, Path] = "./vector_index_speci"):
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