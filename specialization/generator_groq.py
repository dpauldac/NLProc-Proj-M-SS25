from openai import OpenAI
from typing import Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, set_seed, AutoModel, AutoModelForCausalLM, AutoConfig
from typing import List, Optional
#set_seed(42)
from utils.utils import detect_task_type
import os
from dotenv import load_dotenv


class GeneratorGroq:
    """
    Generator class for producing grounded answers using the Groq API and a large language model (e.g., LLaMA 3).

    This class detects the type of user question (e.g., fact, reasoning, summarization),
    constructs an appropriate prompt, sends it to the Groq-hosted LLM via the OpenAI-compatible API,
    and returns the generated response.

    Attributes:
        model_name (str): Name of the LLM to use via the Groq API (default: "llama3-70b-8192").
        client (OpenAI): OpenAI-compatible API client configured for Groq.

    Environment:
        Expects `GROQ_API_KEY` to be set in a `.env` file or in the environment.

    Example:
        >>> generator = GeneratorGroq()
        >>> answer, prompt = generator.generate_answer("What is Apple's net income?", contexts)
    """
    def __init__(self, model_name="llama3-70b-8192"):
        self.model_name = model_name
        load_dotenv()  # Automatically loads .env from project root
        self.client = OpenAI(api_key=os.getenv("GROQ_API_KEY") or "write_api_key_here_directly")
        self.client.base_url = "https://api.groq.com/openai/v1"

    def generate_answer(self, question:str, contexts):
        """
        Generates an answer to a given question based on provided context using the Groq LLM.

        Args:
            question (str): The question to be answered.
            contexts (List[str]): List of context strings retrieved for the question.

        Returns:
            Tuple[str, str]:
                - The generated answer string.
                - The final prompt sent to the model.
        """
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
        """
            Constructs a task-specific prompt using the given question and retrieved context.
            Uses automatic task type detection to apply appropriate instructions:
                - Fact-based QA
                - Yes/No
                - Comparative
                - Summarization
                - Multiple choice
                - Reasoning/inference
                - Classification
                - Hallucination detection

            Args:
                question (str): The user's question.
                contexts (List[str]): List of context passages used for grounding the answer.

            Returns:
                str: A formatted prompt tailored to the detected task type.
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
