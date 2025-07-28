from openai import OpenAI
from typing import Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, set_seed, AutoModel, AutoModelForCausalLM, AutoConfig
from typing import List, Optional
#set_seed(42)
from utils.utils import detect_task_type
import os

class GeneratorGroq:
    def __init__(self, model_name="llama3-70b-8192"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("GROQ_API_KEY") or "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")
        self.client.base_url = "https://api.groq.com/openai/v1"

    def generate_answer(self, question, contexts):
        # Join the context chunks into a single string (you can also summarize or limit length)
        combined_context = "\n---\n".join(contexts) if contexts else "No context provided."

        prompt = self.build_prompt(question, contexts)

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant who uses the provided context to answer questions."},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        prompt = "\n".join([m["content"] for m in messages])
        return answer, prompt

    def build_prompt(self, question:str,  contexts: List[str]) -> str:
        """Constructs a task-aware prompt using provided context and question.

            Combines base instructions with task-specific templates based on automatic
            task type detection. Supports QA, summarization, multiple choice, and
            classification tasks.

            Args:
                question (str): User's input question or instruction
                contexts (List[str]): Retrieved context passages for grounding

            Returns:
                str: Formatted prompt combining instructions, context, and task template
        """
        """
        #basic prompt
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])
        prompt = f"Context: {context_str}\n\nQuestion:{question}"
        """

        """Adaptive prompt construction with auto-detected task type"""
        task_type = detect_task_type(question, contexts)
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])

        base_prompt = (
            "Follow these rules:\n"
            "1. Use only the information from the context\n"
            "2. If unsure, say 'I don't know'\n"
            "3. Be concise\n\n"
            f"Context:\n{context_str}\n\n"
        )

        task_prompts = {
            "fact": f"Q: {question}\nA:",
            "reasoning": f"Q: {question}\nExplain your reasoning briefly, using context.\nA:",
            "yes-no": f"Q: {question}\nAnswer 'Yes', 'No', or 'I don't know', with brief justification from the context.\nA:",
            "comparative": f"Q: {question}\nCompare based on the context. Answer concisely.\nA:",
            "summarization": f"Summarize what the context says about: {question}\nSummary:",
            "classification": f"Q: {question}\nClassify based on context only.\nA:",
            "multiple_choice": f"Q: {question}\nChoose the best option (e.g., 'a', 'b', etc.) based only on context.\nA:",
            "hallucination": f"Q: {question}\nBased on the context, is this question answerable? Answer only 'I don't know' if not.\nA:",
            "qa": f"Q: {question}\nA:"
        }
        print(f"==>Question type: {task_type} \n")
        return base_prompt + task_prompts.get(task_type, f"Q: {question}\nA:")
