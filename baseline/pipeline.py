from pathlib import Path
from typing import List, Union, Optional
from transformers import GenerationConfig
from baseline.retriever import Retriever
from baseline.generator import Generator

class Pipeline:
    def __init__(
            self,
            document_paths: List[Union[str, Path]],
            index_save_path: Union[str, Path] = "sentence_embeddings_index",
            chunk_size: int = 500,
            generator_model: str = "google/flan-t5-base",
            retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            max_length: int = 512,
            generation_config: Optional[GenerationConfig] = None) -> None:
        self.document_paths = document_paths
        self.index_save_path = index_save_path
        self.retriever = Retriever(chunk_size=chunk_size, model_name=retriever_model)
        self.generator = Generator(model_name=generator_model, generation_config=generation_config)
        self._index_loaded = False
        self.retriever.add_documents(document_paths)
        self.retriever.save(index_save_path)

    def index_documents(self, document_paths: List[Union[str, Path]]):
        """Add and index documents"""
        self.retriever.add_documents(document_paths)
        self.retriever.save(self.index_save_path)
        self._index_loaded = True

    def query(self, question: str, k: int = 3) -> str:
        """End-to-end question answering"""
       # if not self._index_loaded:
        #    raise ValueError("Load documents first using index_documents()")
        contexts = self.retriever.query(question, k)
        return self.generator.generate_answer(question, contexts)

    def save_index(self, save_path: Union[str, Path]):
        """Persist document index"""
        self.retriever.save(save_path)

    def load_index(self, load_path: Union[str, Path]):
        """Load existing document index"""
        self.retriever.load(load_path)
        self._index_loaded = True



# put the code that calls all the different components here
'''
def rag_pipeline(retriever, generator, question: str, k: int = 3) -> str:
    chunks = retriever.query(question, k=k)
    #context = "\n".join(chunks)
    context = chunks
    answer = generator.generate_answer(context, question)
    return answer
'''