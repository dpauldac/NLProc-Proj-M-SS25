# put the code that calls all the different components here
def rag_pipeline(retriever, generator, question: str, k: int = 3) -> str:
    chunks = retriever.query(question, k=k)
    #context = "\n".join(chunks)
    context = chunks
    answer = generator.generate_answer(context, question)
    return answer

from pathlib import Path
from typing import List, Union
from retriever import Retriever
from generator import Generator

class Pipeline:
    def __init__(self, chunk_size: int = 500, generator_model: str = "google/flan-t5-base", retriever_model: str = "google/flan-t5-base-re-ranked", max_length: int = 512) -> None:
        self.retriever = Retriever(chunk_size=chunk_size, model_name=retriever_model)
        self.generator = Generator(max_length=max_length, model_name=generator_model)
        self._index_loaded = False