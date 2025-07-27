import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_similarity  # Embedding-based alternative
)
from ragas.embeddings import HuggingfaceEmbeddings
from transformers import pipeline
from ragas.llms import LangchainLLM
import torch
from langchain_community.llms import HuggingFacePipeline

# ===== CONFIGURE OPEN-SOURCE MODELS =====
# 1. For embedding-based metrics (context precision/recall, answer similarity)
embedding_model = HuggingfaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 2. For LLM-based metrics (if needed)
# Create a local LLM pipeline (requires GPU for reasonable speed)
def create_local_llm():
    model_name = "google/flan-t5-small"  # Lightweight model
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
        max_new_tokens=100
    )
    return HuggingFacePipeline(pipeline=pipe)


# ===== LOAD DATA =====
with open("test/test_report_speci_new.json", "r") as f:
    data = json.load(f)

# ===== REFORMAT DATA =====
records = []
for key, value in data.items():
    # ... (same data processing as before) ...
    records.append({
        "question": value["question"],
        "answer": value["answer_received"],
        "ground_truth": value.get("ground_truth", ""),
        "contexts":  literal_eval(value["context"]) if isinstance(value["context"], str) else value["context"]  # MUST be list of strings
    })

df = pd.DataFrame(records)
hf_dataset = Dataset.from_pandas(df)

# ===== FREE METRICS (NON-LLM BASED) =====
# These metrics work without any API calls
free_metrics = [
    context_precision,
    context_recall,
    answer_similarity  # Uses open-source embeddings
]

# Configure embeddings for answer_similarity
answer_similarity.embedding = embedding_model

# ===== RUN EVALUATION =====
if len(hf_dataset) > 0:
    try:
        # Evaluate with free metrics
        results = evaluate(
            hf_dataset,
            metrics=free_metrics
        )

        # Save results
        results_df = results.to_pandas()
        results_df.to_csv("free_ragas_results.csv", index=False)

        print("\n===== FREE METRICS RESULTS =====")
        print(f"Context Precision: {results_df['context_precision'].mean():.2f}")
        print(f"Context Recall:    {results_df['context_recall'].mean():.2f}")
        print(f"Answer Similarity: {results_df['answer_similarity'].mean():.2f}")

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
else:
    print("No valid records to evaluate")