from pathlib import Path
from typing import List, Union, Optional
from transformers import GenerationConfig
from baseline.retriever import Retriever
from specialization.generator_groq import GeneratorGroq
from specialization.retriever_speci import RetrieverSpeci
from baseline.generator import Generator
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class LogEntry:
    """
    A dataclass to represent a log entry for each query.

    Attributes:
        question (str): The user input question.
        retrieved_chunks (List[str]): List of retrieved document chunks.
        prompt (str): The final prompt sent to the generator model.
        generated_answer (str): The generated answer from the model.
        timestamp (str): ISO-formatted timestamp of when the query was processed.
        group_id (str): Identifier for the team/user.
    """
    question: str
    retrieved_chunks: List[str]
    retrieved_chunks_pages: Optional[List[str]]
    retrieved_chunks_source: Optional[List[str]]
    prompt: str
    generated_answer: str
    timestamp: str
    group_id: str

class PipelineSpeci:
    """
       A Retrieval-Augmented Generation (RAG) pipeline that combines a semantic retriever and a text generator
       to answer user queries using document context.

       Args:
           document_paths (Optional[List[Union[str, Path]]]): Paths to documents for indexing.
           index_save_path (Union[str, Path]): Path to save/load the FAISS index and metadata.
           chunk_size (int): Character-based chunking size (optional, used in retriever init).
           generator_model (str): model name for the generator (e.g., "google/flan-t5-base", "llama3-70b-8192").
           groq_model (bool): If True, uses the Groq-hosted generator (e.g., LLaMA 3); else uses local HuggingFace model.
           retriever_model (str): model name for the sentence encoder.
           generation_config (Optional[GenerationConfig]): Custom generation parameters (temperature, max_tokens, etc.).
           log_path (Union[str, Path]): Path to save logs of queries and generated answers.
           rebuild_index (bool): Whether to re-index documents from scratch or load a saved index.
       Raises:
           ValueError: If `rebuild_index` is True but no document_paths are provided.

       Example Usage:
            pipeline = PipelineSpeci(doc_path="data/findocs", index_save_path="./index"...)

       """
    def __init__(
            self,
            document_paths: Optional[List[Union[str, Path]]] = None,
            index_save_path: Union[str, Path] = "./vector_index_speci",
            chunk_size: int = 200,
            groq_model: bool = True,
            generator_model: Optional[str] = None,
            retriever_model: str = "sentence-transformers/all-mpnet-base-v2",
            generation_config: Optional[GenerationConfig] = None,
            log_path: Union[str, Path] = "rag_logs_spec.jsonl",
            rebuild_index: bool = False
    ) -> None:
        self.document_paths = document_paths
        self.index_save_path = index_save_path
        self.retriever = RetrieverSpeci(model_name=retriever_model)
        if generator_model is None:
            if groq_model:
                generator_model = "llama3-70b-8192"  # Default for Groq
            else:
                generator_model = "google/flan-t5-base"  # Default for standard
            print(
                f"Info: No generator_model provided. Defaulting to '{generator_model}'.")

        #self.generator = Generator(model_name=generator_model, generation_config=generation_config)
        self.generator = GeneratorGroq(model_name=generator_model) if groq_model else Generator(
            model_name=generator_model, generation_config=generation_config)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rebuild_index = rebuild_index

        if self.rebuild_index:
            print(self.document_paths)
            if not self.document_paths:
                raise ValueError("document_paths must be provided when rebuild_index=True")
            print("<=****Rebuilding index:****=>")
            self.retriever.add_documents(self.document_paths)
            if not self.index_save_path:
                raise ValueError("Index path must be provided to save the index.")
            self.retriever.save(self.index_save_path)
        else:
            if not index_save_path:
                raise ValueError("Index path must be provided to load the index.")
            self.retriever.load(index_save_path)

    def _log_query(self, entry: Dict[str, Any]):
        """
        Logs a query and its associated context, prompt, and response to a file in JSONL format.

        Args:
            entry (Dict[str, Any]): Dictionary containing metadata about a query, including
                                question, context, prompt, answer, etc.
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def query(self, question: str, k: int = 5) -> str:
        """
        Runs an end-to-end RAG query: retrieve context, generate answer, and log results.

        Args:
            question (str): The user's input question.
            k (int): Number of top retrieved chunks to use as context.

        Returns:
            str: The generated answer from the language model.
        """
       # if not self._index_loaded:
        #    raise ValueError("Load documents first using index_documents()")
        retrieved_results = self.retriever.query(question, k)
        #print(f"retrieved_results:{retrieved_results}")        #Print retrieved chunks
        #print(f"Question:{question}")                          #Print actual question asked
        #print(f"from pipeline_speci, retrieved: {retrieved_results.chunk_text}")
        answer, prompt = self.generator.generate_answer(question, retrieved_results.chunk_text)
        #print(f"Answer:{answer}")                             #Print answer generated
        # Create log entry
        log_entry = LogEntry(
            question=question,
            retrieved_chunks=retrieved_results.chunk_text,
            retrieved_chunks_pages=retrieved_results.pages,
            retrieved_chunks_source=retrieved_results.source,
            prompt=prompt,
            generated_answer=answer,
            timestamp=datetime.now().isoformat(),
            group_id="Team Oneironauts"
        )
        self._log_query(asdict(log_entry))

        return answer

    def add_documents(self, documents: List[Union[str, Path]], index_path: Union[str, Path]) -> None:
        """
        Adds new documents to the retriever index and updates the FAISS + BM25 store.

        This method:
            - Sets the document and index paths.
            - Validates inputs.
            - Re-chunks and re-indexes the documents.
            - Saves and reloads the updated index.

        Args:
            documents (List[Union[str, Path]]): List of file paths to documents to be indexed.
            index_path (Union[str, Path]): Directory path to save the FAISS and metadata index.

        Raises:
            ValueError: If documents or index_path are not provided.
        """
        if documents:
            self.document_paths = documents
        if index_path:
            self.index_save_path = index_path

        if not self.document_paths:
            raise ValueError("Path or paths must be provided to add documents.") #if document_path exists then no exception, continue
        if not self.index_save_path:
                raise ValueError("Index path must be provided to save the index.") #if index_save_path exists then no exception, continue

        print("<=****Rebuilding index:****=>")
        self.retriever.add_documents(self.document_paths)
        self.retriever.save(self.index_save_path)
        self.retriever.load(self.index_save_path)

# put the code that calls all the different components here
'''
def rag_pipeline(retriever, generator, question: str, k: int = 3) -> str:
    chunks = retriever.query(question, k=k)
    #context = "\n".join(chunks)
    context = chunks
    answer = generator.generate_answer(context, question)
    return answer
'''