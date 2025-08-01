from pathlib import Path
from typing import List, Union, Optional
from transformers import GenerationConfig
from baseline.retriever import Retriever
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

class Pipeline:
    def __init__(
            self,
            document_paths: List[Union[str, Path]],
            index_save_path: Union[str, Path] = "vector_index",
            rebuild_index: bool = False,
            chunk_size: int = 200,
            generator_model: str = "google/flan-t5-base",
            retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            generation_config: Optional[GenerationConfig] = None, log_path: Union[str, Path] = "rag_logs.jsonl") -> None:
        self.document_paths = document_paths
        self.index_save_path = index_save_path
        self.retriever = Retriever(chunk_size=chunk_size, model_name=retriever_model)
        self.generator = Generator(model_name=generator_model, generation_config=generation_config)
        self.rebuild_index = rebuild_index
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if rebuild_index:
            if not document_paths:
                raise ValueError("document_paths must be provided when rebuild_index=True")
            print("<=****Rebuilding index:****=>")
            self.retriever.add_documents(document_paths)
            self.retriever.save(index_save_path)
        else:
            self.retriever.load(index_save_path)

    def _log_query(self, entry: Dict[str, Any]):
        """Append log entry to file"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def index_documents(self, document_paths: List[Union[str, Path]]):
        """Add and index documents"""
        self.retriever.add_documents(document_paths)
        self.retriever.save(self.index_save_path)
        #self._index_loaded = True

    def query(self, question: str, k: int = 5) -> str:
        """End-to-end question answering"""
       # if not self._index_loaded:
        #    raise ValueError("Load documents first using index_documents()")
        retrieved_results = self.retriever.query(question, k)
        print(retrieved_results)
        answer, prompt = self.generator.generate_answer(question, retrieved_results.chunk_text)

        # Create log entry
        log_entry = LogEntry(
            question=question,
            retrieved_chunks=retrieved_results.chunk_text,
            retrieved_chunks_pages=retrieved_results.pages,
            #retrieved_chunks_source=retrieved_results.source,
            prompt=prompt,
            generated_answer=answer,
            timestamp=datetime.now().isoformat(),
            group_id="Team Oneironauts"
        )
        self._log_query(asdict(log_entry))

        return answer

    def save_index(self, save_path: Union[str, Path]):
        """Persist document index"""
        self.retriever.save(save_path)

    def load_index(self, load_path: Union[str, Path]):
        """Load existing document index"""
        self.retriever.load(load_path)
        #self._index_loaded = True



# put the code that calls all the different components here
'''
def rag_pipeline(retriever, generator, question: str, k: int = 3) -> str:
    chunks = retriever.query(question, k=k)
    #context = "\n".join(chunks)
    context = chunks
    answer = generator.generate_answer(context, question)
    return answer
'''