import json
from typing import List, Dict
from pathlib import Path
from specialization.pipeline_speci import PipelineSpeci
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


class PipelineTester:
    """
       A utility class for evaluating the current RAG system pipeline.

       This tester loads test cases from a JSON file and evaluates the pipeline using:
         - Cosine similarity between generated and reference answers.
         - Semantic F1 score based on sentence embeddings.
         - Grounding checks to verify if the answer terms appear in the retrieved context.

       Attributes:
           pipeline (PipelineSpeci): An instance of the RAG pipeline to evaluate.
           test_cases (List[Dict]): A list of test case dictionaries with 'question' and 'ground_truth'.
           sim_model (SentenceTransformer): A model used for embedding-based similarity computations.
           sim_model (SentenceTransformer): A model used for embedding-based similarity computations.
       """

    def __init__(self, pipeline, test_file_path: str = "test_inputs_speci.json"):
        """
        Initialize the tester with a pipeline and load test cases from a file.

        Args:
            pipeline (PipelineSpeci): The RAG pipeline instance to test.
            test_file (str): Path to the JSON file containing test inputs.
        """
        self.pipeline = pipeline
        self.test_cases = self.load_test_cases(test_file_path)
        self.sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def load_test_cases(self, path: str) -> List[Dict]:
        """
        Load test cases from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            List[Dict]: A list of test case dictionaries.
        """
        with open(path, "r") as f:
            return json.load(f)

    def check_grounding(self, answer: str, contexts: List[str]) -> bool:
        """
        Check whether any terms from the answer appear in the retrieved context.

        Args:
            answer (str): The generated answer.
            contexts (List[str]): Retrieved document chunks.

        Returns:
            bool: True if any term in the answer is found in the context.
        """
        context_text = " ".join(contexts).lower()
        answer_terms = answer.lower().split()
        return any(term in context_text for term in answer_terms)

    def cosine_similarity(self, a: str, b: str) -> float:
        """
        Compute cosine similarity between two text strings using sentence embeddings.

        Args:
            a (str): First text string.
            b (str): Second text string.

        Returns:
            float: Cosine similarity score between the two texts.
        """
        emb_a = self.sim_model.encode(a, convert_to_tensor=True)
        emb_b = self.sim_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b)[0][0])

    def semantic_f1(self, pred: str, true: str) -> float:
        """
           Compute an approximate F1 score based on embedding similarity of tokens.

           Args:
               pred (str): Predicted/generated answer.
               true (str): Ground truth/reference answer.

           Returns:
               float: A semantic F1 score between 0 and 1.
        """
        pred_tokens = pred.split()
        true_tokens = true.split()

        if not pred_tokens or not true_tokens:
            return 0.0

        pred_embs = self.sim_model.encode(pred_tokens, convert_to_tensor=True)
        true_embs = self.sim_model.encode(true_tokens, convert_to_tensor=True)

        # Match predicted to true
        precision_matches = util.cos_sim(pred_embs, true_embs).max(dim=1).values
        recall_matches = util.cos_sim(true_embs, pred_embs).max(dim=1).values

        precision = (precision_matches > 0.7).float().mean().item()
        recall = (recall_matches > 0.7).float().mean().item()

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def run_tests(self) -> Dict[str, Dict]:
        """
        Run all loaded test cases on the pipeline and compute evaluation metrics.

        Returns:
             Dict[str, Dict]: A dictionary containing results per test case and an overall summary.
        """
        results = {}
        f1_scores = []

        for idx, test in enumerate(self.test_cases):
            answer = self.pipeline.query(test["question"])
            contexts = self.pipeline.retriever.query(test["question"])
            ground_truth = test["ground_truth"]

            sim_score = self.cosine_similarity(answer, ground_truth)
            sem_f1 = self.semantic_f1(answer, ground_truth)

            f1_scores.append(sem_f1)

            results[f"test_{idx}"] = {
                "question": test["question"],
                "ground_truth": ground_truth,
                "answer_received": answer,
        #        "grounding_check": self._check_grounding(answer, contexts),
                "cosine_similarity": round(sim_score, 4),
                "semantic_f1": round(sem_f1, 4)
            }

        # Aggregate scores
        avg_f1 = np.mean(f1_scores)
        results["summary"] = {
            "average_semantic_f1": round(avg_f1, 4),
            "total_tests": len(self.test_cases)
        }

        print(f"\n--- Evaluation Summary ---")
        print(f"Avg Semantic F1 Score      : {avg_f1:.4f}")
        print(f"Total Test Cases Evaluated : {len(self.test_cases)}")
        print(f"---------------------------\n")

        self._generate_report(results)
        return results

    def _generate_report(self, results: Dict):
        """
        Save evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results to save.
        """
        report_path = Path("test/test_report_speci.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test report saved to {report_path.absolute()}")
