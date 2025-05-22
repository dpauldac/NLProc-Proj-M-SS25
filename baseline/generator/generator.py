#implement your generator here
from typing import Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, set_seed
from typing import List, Optional
#set_seed(42)
from utils.utils import _detect_task_type

class Generator:
    """A text generation module for question answering tasks with adaptive prompting.

        Handles prompt construction and text generation using sequence-to-sequence models.
        Supports different task types through automatic task detection and adaptive prompting.

        Args:
            model_name (str): Pretrained model identifier from Hugging Face Hub.
                Default: "google/flan-t5-base"
            generation_config (Optional[GenerationConfig]): Configuration for text generation.
                If None, uses default parameters from the model. Default: None

        Attributes:
            tokenizer (AutoTokenizer): Tokenizer for the specified model
            model (AutoModelForSeq2SeqLM): Pretrained model for text generation
            generation_config (GenerationConfig): Configuration object for generation parameters
    """
    def __init__(self, model_name: str = "google/flan-t5-base", generation_config: Optional[GenerationConfig] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig(
        #       max_length=512,
        #        temperature = 0.3,  #a bit of creativity
       #         num_beams=2,    #Enables beam search. More beams = more exploration for best output, but slower.
       #         early_stopping=True, #Prevents unnecessarily long outputs with beam search.
       #         do_sample = True    #Randomly selects tokens based on probabilities.
           )
        self.generation_config = generation_config

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
        task_type = _detect_task_type(question, contexts)
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])

        base_prompt = (
            "Follow these rules:\n"
            "1. Use only the information from the context\n"
            "2. If unsure, say 'I don't know'\n"
            "3. Be concise\n\n"
            f"Context:\n{context_str}\n\n"
        )

        task_specific = {
            'qa': f"Question: {question}\nAnswer:",
            'summarization': (
                f"Summarize the key points from the context about '{question}'.\n"
                "Keep summary under 150 words.\nSummary:"
            ),
            'multiple_choice': (
                f"Question: {question}\n"
                "Choose the correct option using the context.\n"
                "Answer only the letter (e.g., 'a') or 'I don't know'.\nAnswer:"
            ),
            'classification': (
                f"Classify: {question}\n"
                "Use the policy guidelines in the context.\n"
                "Options: offensive/non-offensive\nAnswer:"
            ),
        }

        return base_prompt + task_specific[task_type]

    def generate_answer(self, question:str, contexts: List[str]) -> tuple[Any, str]:
        """Generates an answer using the provided question and contexts.

        Processes the inputs through the full pipeline:
        1. Constructs task-aware prompt
        2. Tokenizes inputs with length constraints
        3. Generates text using configured parameters
        4. Decodes and returns both answer and final prompt

        Args:
            question (str): User's input question or instruction
            contexts (List[str]): Retrieved context passages for grounding

        Returns:
            Tuple[str, str]: Generated answer string and full prompt used for generation
        """
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        #t5 struggles if length is more than 512.
        output = self.model.generate(**inputs, generation_config = self.generation_config)
       # output = self.model.generate(**inputs)
        #num_return_sequences
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return answer, prompt