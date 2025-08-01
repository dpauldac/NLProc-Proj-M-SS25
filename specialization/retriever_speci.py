#implement your retriever here
from typing import List, Union, Any
from pathlib import Path
import os
import faiss
import nltk
import numpy as np
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi  # Added for BM25
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
from utils.utils import chunk_text_recursive_character, chunk_text_fixed_size
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from utils.ner_utils import extract_org_ner
from utils.utils import read_file
from baseline.retriever.retrieval_results import RetrievalResults
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

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
    def __init__(self, chunk_size: int = 200, model_name: str = "sentence-transformers/all-mpnet-base-v2", max_tokens: int = 320):
       self.chunk_size = chunk_size
       self.model_name = model_name
       self.max_tokens = max_tokens
       self.model = SentenceTransformer(self.model_name)
       self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
       self.tokenizer =  HuggingFaceTokenizer(
           tokenizer=AutoTokenizer.from_pretrained(self.model_name),
           #max_tokens=self.max_tokens ,  # optional, by default derived from `tokenizer` for HF case
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

       # BM25 components
       self.bm25_index = None
       self.bm25_tokenizer = lambda text: [
           token for token in word_tokenize(text.lower())
           if token.isalnum() and token not in stopwords.words('english')
       ]  # Simple tokenizer
       self.bm25_doc_tokens = []  # Stores tokenized chunks for BM25

    def chunk_doc(self, doc_path: Union[str, Path]):
        """
        Converts all supported files in a directory to Docling documents
        and then chunks them.

        Args:
            directory_path (str, Path): The path to the directory containing the documents.
            max_tokens (int): The maximum number of tokens per chunk.
        """
        result = self.converter.convert(doc_path)
        doc = result.document
        chunk_iter = self.chunker.chunk(dl_doc=doc)
        cur_doc_chunks = []
        sources = []
        pages = []
        for chunk in chunk_iter:
            enriched_text = self.chunker.contextualize(chunk=chunk)
            cur_doc_chunks.append(enriched_text)
            source = chunk.meta.origin.filename
            sources.append(source)
            page_numbers = sorted(
                set(
                    prov.page_no
                    for item in chunk.meta.doc_items
                    for prov in item.prov
                    if hasattr(prov, "page_no")
                )
            )
            pages.append(page_numbers)
        #print(cur_doc_chunks)
        print(f"#########Processed and chunked: {doc_path}######")
        return cur_doc_chunks, sources, pages

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
        all_sources = []
        all_pages = []
        for path in file_paths:
            cur_doc_chunks, sources, pages  = self.chunk_doc(path)
            raw_text = read_file(path, max_pages=2)
            ner_data = extract_org_ner(raw_text)
            cur_org_name = ner_data[0][0] if ner_data and ner_data[0] else ""  # first detected org entity

            # Attach org name to each chunk
            if cur_org_name:
                cur_doc_chunks = [f"[ORG: {cur_org_name}] {chunk}" for chunk in cur_doc_chunks]

            all_chunks.extend(cur_doc_chunks)
            all_sources.extend(sources)
            all_pages.extend(pages)

        # Convert chunks to embeddings
        if not all_chunks:
            print("No valid chunks to process")
            return

        #For FAISS
        #Convert each chunk of text into a vector using the SentenceTransformer model.
        embeddings = self.model.encode(all_chunks, convert_to_numpy=True)
        #embeddings = embeddings.cpu().numpy().astype('float32')

        #Get the dimension of each embedding vector (384 for all-MiniLM-L6-v2).
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        dim = embeddings.shape[1]

        # for cosine similarity indexing, not used now
        faiss.normalize_L2(embeddings)
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        # for L2 similarity
        #if self.index is None:
        #    self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.embeddings.extend(embeddings)

        # Tokenize for BM25
        tokenized_chunks = [self.bm25_tokenizer(chunk) for chunk in all_chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        self.bm25_doc_tokens.extend(tokenized_chunks)

        # Inside a dictionary saving each chunk by assigning each to the next available integer key, will be useful for future retrival when there is a query performed
        next_id = len(self.id_to_chunk)  # Start ID for new chunks
        for idx, (chunk, src, pgs) in enumerate(zip(all_chunks, all_sources, all_pages)):
            self.id_to_chunk[next_id + idx] = {
                # using len(self.id_to_chunk) to get the next integer, if the length is 3, then starting from 0, 1, 2 will be already used, then next integer 3 will be assigned.
                "chunk_text": chunk,
                "source": src,
                "pages": pgs,
            }

    def _retrieve_bm25(self, query: str, candidate_size: int):
        """
           Retrieve candidate text chunks using BM25 sparse retrieval.

           Args:
               query (str): The search query string.
               candidate_size (int): Number of top candidates to return.

           Returns:
               Tuple[List[dict], List[float], List[int]]:
                   - List of result dictionaries containing chunk text, source, pages, and BM25 score.
                   - List of BM25 scores corresponding to all chunks.
                   - List of indices of top-ranked chunks based on BM25 scores.
        """
        if not self.bm25_index:
            return [], []
        query_tokens = self.bm25_tokenizer(query)
        scores = self.bm25_index.get_scores(query_tokens)
        indices = np.argsort(scores)[::-1][:candidate_size]
        results = []
        for idx in indices:
            entry = self.id_to_chunk[idx]
            results.append({
                "chunk_text": entry["chunk_text"],
                "source": entry.get("source", ""),
                "pages": entry["pages"],
                "score": float(scores[idx])
            })
        return results, scores, indices

    def _retrieve_faiss(self, query: str, candidate_size: int):
        """
        Retrieve candidate text chunks using FAISS dense vector similarity search.

        Args:
            query (str): The search query string.
            candidate_size (int): Number of top candidates to return.

        Returns:
            Tuple[List[dict], List[float], List[int]]:
                - List of result dictionaries containing chunk text, source, pages, and FAISS similarity score.
                - List of FAISS similarity scores for the top candidates.
                - List of indices of the top retrieved chunks from the FAISS index.
        """
        if not self.index:
            return [], []
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, candidate_size)

        results = []
        for i, idx in enumerate(indices[0]):
            entry = self.id_to_chunk[idx]
            results.append({
                "chunk_text": entry["chunk_text"],
                "source": entry.get("filename", ""),
                "pages": entry["pages"],
                "score": float(scores[0][i])
            })
        return results, scores[0], indices[0]

    def query(self, query: str, k: int = 5, candidate_size: int = 5) -> list[Any] | RetrievalResults:
        """
        Perform hybrid retrieval to find top-k relevant chunks using FAISS, BM25, and cross-encoder reranking.

        Args:
            query (str): The user’s search query.
            k (int): Number of top final results to return after reranking. Default is 5.
            candidate_size (int): Number of top candidates to retrieve from each method (BM25/FAISS) before reranking.

        Returns:
            RetrievalResults: Top-k reranked results with metadata (chunk text, source, pages, score).
        """
        if not self.index:
            return []

        if candidate_size is None or candidate_size <= k/2:
            candidate_size = k

        # Stage 1: Independent Retrieval
        # Get BM25 candidates
        results_bm25, bm25_scores, bm25_candidates = self._retrieve_bm25(query, candidate_size)

        # Get FAISS candidates
        results_faiss, faiss_score, faiss_candidates = self._retrieve_faiss(query, candidate_size)
        #print(faiss_candidates)

        # Combine candidates (remove duplicates?)
        combined_results = results_bm25 + results_faiss

        print(combined_results)
        print(f"combined results: {combined_results}")
        # Combine candidates and deduplicate by chunk_text (or use source + pages if needed)
        combined_dict = {}
        for r in combined_results:
            key = r["chunk_text"]
            if key not in combined_dict:
                combined_dict[key] = r

        candidates_unique = list(combined_dict.values())
        print(f"Candidate uniqe: {candidates_unique}")
        # ______Optional step: Reranking___
        if not candidates_unique:
            return RetrievalResults([])

        # Stage 3: Prepare pairs for Cross-Encoder reranking
        cross_inp = [(query, c["chunk_text"]) for c in candidates_unique]

        # Stage 4: Predict rerank scores
        rerank_scores = self.cross_encoder.predict(cross_inp)

        # Attach rerank scores to candidates
        for c, score in zip(candidates_unique, rerank_scores):
            c["score"] = float(score)

        # Stage 5: Sort by rerank score descending
        candidates_unique.sort(key=lambda x: x["score"], reverse=True)
        #  ______Optional step end___
        print(candidates_unique[:k])
        return RetrievalResults(candidates_unique[:k])

#        emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
#        if emb.ndim == 1:
#            emb = emb.reshape(1, -1)
        #faiss.normalize_L2(emb) # only used for cosine similarity
#        D, I = self.index.search(emb,k)
        #use the top k indices returned by FAISS search to fetch the actual chunks from the saved dictionary containing all the chunks
        #return [self.id_to_chunk[i] for i in I[0]]

#        # Prepare results with metadata and scores
#       results = []
#        for idx, score in zip(I[0], D[0]):
#            metadata = self.id_to_chunk[idx]
#            #print("Metadata: \n", metadata)
#            results.append({
#                "chunk_text": metadata["chunk_text"],
#                "source": metadata["source"],
#                "pages": metadata["pages"],
#                "similarity": float(score)
#           })
#        return RetrievalResults(results)

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

        # Save BM25 index using pickle
        with open(os.path.join(dir_path,"bm25_index.pkl"), "wb") as f:
            pickle.dump(self.bm25_index, f)

        # Save chunks_with_metadata once for both FAISS and BM25
        with open(os.path.join(dir_path, "chunks_with_metadata.pkl"), "wb") as f:
            pickle.dump(self.id_to_chunk, f)

    def load(self, dir_path:  Union[str, Path]):
        """
        Load the FAISS index and chunk metadata from disk.

        Args:
            dir_path (Union[str, Path]): Directory path to load the index and metadata from.
        """
        dir_path = Path(dir_path)
        # Check if the directory path exists and is a directory
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"The directory '{dir_path}' does not exist.")

        faiss_path = os.path.join(dir_path, "faiss.index")
        bm25_path = os.path.join(dir_path, "bm25_index.pkl")
        meta_path = os.path.join(dir_path, "chunks_with_metadata.pkl")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError("faiss.index file not found.")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError("bm25_index.pkl file not found.")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("chunks_with_metadata.pkl file not found.")

        self.index = faiss.read_index(os.path.join(dir_path, "faiss.index"))

        # load BM25 index using pickle
        with open(os.path.join(dir_path, "bm25_index.pkl"), "rb") as f:
            self.bm25_index = pickle.load(f)

        # load chunks_with_metadata once for both FAISS and BM25
        with open(os.path.join(dir_path, "chunks_with_metadata.pkl"), "rb") as f:
            self.id_to_chunk = pickle.load(f)

        print(f"___...___\n[RetrieverSpeci]\nSuccessfully loaded FAISS and BM25 indexes from {dir_path}.\nFAISS index: {self.index }\nBM25 index: {self.bm25_index}\n___...___")
