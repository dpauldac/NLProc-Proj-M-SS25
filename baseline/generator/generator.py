#implement your generator here
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from typing import List
#set_seed(42)

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length

    def build_prompt(self, question:str, contexts: List[str]) -> str:
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])
        return f"Context: {context_str}\n\nQuestion:{question}"

    def generate_answer(self, question:str, contexts: List[str]) -> str:
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        #t5 struggles if length is more than 512.
        output = self.model.generate(**inputs, max_length=self.max_length)
        #num_return_sequences

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
