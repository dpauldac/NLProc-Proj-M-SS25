#implement utility functions here
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import re
from typing import List, Union
from pathlib import Path
import os
import pymupdf  # For PDF handling (pip install pymupdf)

def read_file(file_path: Union[str, Path], max_pages: int = None) -> str:
  path = Path(file_path)

  text = ""
  if path.suffix == ".pdf":
    with pymupdf.open(path) as doc:
      for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
          break
        text += page.get_text()
  elif path.suffix in (".txt",'.md'):
      text = path.read_text()
  else:
      raise ValueError(f"Unsupported file type: {path.suffix}, only .txt, .pdf and .md are supported.")

  return text

def is_policy_related(text):
    lower_text = text.lower()
    policy_keywords = ["policy", "guideline"]
    rule_indicators = ["must", "should", "required", "prohibited", "permitted", "govern", "regulate", "rule", "regulation", "law", "compliance", "procedure", "protocol"]
    for keyword in policy_keywords:
        if keyword in lower_text:
            for indicator in rule_indicators:
                if indicator in lower_text:
                    return True
    return False

def detect_task_type(question: str, contexts: List[str]) -> str:
    q_lower = question.lower()

    # Check for Yes/No type
    if q_lower.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "should ", "has ", "have ", "had ")):
        return "yes-no"

    # Hallucination detection: If there's no clear signal in context
    if not any(any(keyword in ctx.lower() for keyword in question.lower().split()) for ctx in contexts):
        return "hallucination"

    # Check for summarization/explanation
    if any(word in q_lower for word in ['summarize', 'summary', 'overview', 'explain', 'describe']):
        return "summarization"

    # Check for multiple choice patterns
    if re.search(r"\b([a-z]\)|\([a-zA-Z]\)|[a-z]\.|[1-9]\d*\.|[ivx]+\.|[ivx]+\)|[A-Z]\))\s*", question):
        return "multiple_choice"

    # Classification-type signal
    if any(word in q_lower for word in ['classify', 'categorize', 'label']):
        return "classification"

    # Comparative or ranking
    if any(word in q_lower for word in ['higher', 'lower', 'compare', 'difference', 'rank', 'more', 'less']):
        return "comparative"

    # Inference or reasoning signal
    if any(word in q_lower for word in ['why', 'how', 'what caused', 'derive', 'conclude', 'suggest']):
        return "reasoning"

    # Paraphrased / extractive factual recall
    if any(word in q_lower for word in ['who', 'when', 'what', 'where', 'how much', 'how many']):
        return "fact"

    return "qa"  # fallback


def detect_task_type2(question: str, contexts: List[str]) -> str:
    """Automatically detect task type from question/content patterns"""
    # Check for multiple choice pattern
    """
    # Example usage:
    question1 = "Which of the following is true? a)Option A b) Option B"  # No space after a)
    question2 = "Select the best answer: b) This one is correct."
    question3 = "Question with choices like (c) Third possibility."
    question4 = "Choose one: a. Option 1 b. Option 2 c. Option 3"
    question5 = "Select one: (A) First choice (B) Second choice"
    question6 = "Choose the correct one: 1. First choice 2) Second choice"
    question7 = "Also consider: I. Option One II) Option Two"
    question8 = "Just a regular question."
    """

    # Check for multiple choice pattern
    if re.search(r"\b([a-z]\)|\([a-zA-Z]\)|[a-z]\.|[1-9]\d*\.|[ivx]+\.|[ivx]+\)|[A-Z]\))\s*", question):
        return 'multiple_choice'

    # Check for summarization keywords
    if any(word in question.lower() for word in ['summarize', 'summarization', 'overview', 'brief']):
        return 'summarization'

    # Check for classification patterns
    if any(word in question.lower() for word in ['categorize', 'classify']):
        return 'classification'

    # Check if context contains policy/rules
    if any('policy' in ctx.lower() or 'guideline' in ctx.lower() for ctx in contexts):
        return 'policy_based'

    return 'qa'  # default to question answering

# Ensure nltk punkt tokenizer is downloaded for sentence-based chunking
def dl_punkt():
    nltk.download('punkt')
    nltk.download('punkt_tab')

def chunk_text_fixed_size(text: str, chunk_size: int) -> List[str]:
    """
    Split text into fixed-size chunks without overlap.
    This is your original _chunk_text logic.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size
    return chunks


def chunk_text_fixed_size_overlap(text: str, chunk_size: int, overlap_size: int) -> List[str]:
    """
    Split text into fixed-size chunks with a specified overlap.
    """
    if overlap_size >= chunk_size:
        raise ValueError("Overlap size must be less than chunk size.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - overlap_size)  # Move by chunk_size minus overlap
    return chunks


def chunk_text_sentence_based(text: str, chunk_size: int, overlap_sentences: int = 0) -> List[str]:
    """
    Split text into chunks based on sentences, ensuring each chunk contains
    a certain number of sentences or fits within a character limit,
    with optional sentence overlap.

    Args:
        text (str): The input text.
        chunk_size (int): Max character length per chunk (approximate).
        overlap_sentences (int): Number of sentences to overlap between chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_length = 0

    for i, sentence in enumerate(sentences):
        # Check if adding the next sentence exceeds the chunk_size
        # +1 for a space between sentences
        if current_chunk_length + len(sentence) + (
        1 if current_chunk_sentences else 0) > chunk_size and current_chunk_sentences:
            # If current chunk is not empty, add it to chunks list
            chunks.append(" ".join(current_chunk_sentences))

            # Prepare for next chunk with overlap
            if overlap_sentences > 0:
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_chunk_length = len(" ".join(current_chunk_sentences))
            else:
                current_chunk_sentences = []
                current_chunk_length = 0

        current_chunk_sentences.append(sentence)
        current_chunk_length += len(sentence) + (1 if current_chunk_sentences else 0)  # Add 1 for space

    # Add any remaining sentences as the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def chunk_text_recursive_character(text: str, chunk_size: int, overlap_size: int = 0, separators: List[str] = None) -> List[
    str]:
    """
    Recursively split text by various separators trying to keep chunks
    within a certain size and maintain semantic boundaries.
    Inspired by LangChain's RecursiveCharacterTextSplitter.
    """
    dl_punkt()
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]  # Try paragraphs, lines, words, then characters

    final_chunks = []

    def _split_recursively(text_to_split: str, current_separators: List[str]):
        if not text_to_split:
            return

        if len(text_to_split) <= chunk_size:
            final_chunks.append(text_to_split)
            return

        if not current_separators:  # Fallback to fixed size if no more separators
            # If text is too long and no more separators, just split it
            # This is the base case if text cannot be split further by separators
            final_chunks.extend(chunk_text_fixed_size_overlap(text_to_split, chunk_size, overlap_size))
            return

        separator = current_separators[0]
        remaining_separators = current_separators[1:]

        # Split by the current separator
        parts = text_to_split.split(separator)

        current_part_accumulator = ""
        for i, part in enumerate(parts):
            # Check if adding this part would exceed chunk_size
            # Add 1 for separator length if it's not the first part
            part_len_with_sep = len(part) + (len(separator) if current_part_accumulator else 0)

            if len(current_part_accumulator) + part_len_with_sep <= chunk_size:
                if current_part_accumulator and separator:  # Add separator back if not first part
                    current_part_accumulator += separator
                current_part_accumulator += part
            else:
                # If adding this part exceeds, process the accumulated part
                if current_part_accumulator:
                    _split_recursively(current_part_accumulator, remaining_separators)

                    # Handle overlap if applicable. For recursive, overlap is complex.
                    # Here we take the end of the last chunk as the start of the next if needed
                    if 0 < overlap_size < len(current_part_accumulator):
                        current_part_accumulator = current_part_accumulator[-overlap_size:]
                    else:
                        current_part_accumulator = ""
                else:  # current_part_accumulator was empty, meaning the single 'part' is too big
                    # So, try to split this 'part' with the next separator
                    _split_recursively(part, remaining_separators)
                    # No overlap consideration for this case, handled by recursive call
                    current_part_accumulator = ""  # Reset for next accumulation

                # Start new accumulation if the 'part' was added recursively
                if not current_part_accumulator:  # If reset by recursive call or initially empty
                    current_part_accumulator = part

        # Process any remaining accumulated part
        if current_part_accumulator:
            _split_recursively(current_part_accumulator, remaining_separators)

    _split_recursively(text, separators)
    return final_chunks

