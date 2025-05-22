#implement your generator here
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, set_seed
from typing import List, Optional
#set_seed(42)

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", generation_config: Optional[GenerationConfig] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig(
        #        max_length=512,
        #       num_beams=1,
        #       do_sample=False,
        #       early_stopping=True
           )
        self.generation_config = generation_config

    def build_prompt(self, question:str, contexts: List[str]) -> str:
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])
        return f"Context: {context_str}\n\nQuestion:{question}"

    def generate_answer(self, question:str, contexts: List[str]) -> str:
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        #t5 struggles if length is more than 512.
        output = self.model.generate(**inputs, generation_config = self.generation_config)
        #num_return_sequences

        return self.tokenizer.decode(output[0], skip_special_tokens=True)