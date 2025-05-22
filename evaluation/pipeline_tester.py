import json
from typing import List, Dict
from pathlib import Path
from baseline.pipeline import Pipeline


class PipelineTester:
    def __init__(self, pipeline, test_file: str = "test_inputs.json"):
        self.pipeline = pipeline
        self.test_cases = self._load_test_cases(test_file)

    def _load_test_cases(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return json.load(f)

    def _check_grounding(self, answer: str, contexts: List[str]) -> bool:
        """Check if answer terms appear in contexts"""
        context_text = " ".join(contexts).lower()
        answer_terms = answer.lower().split()
        return any(term in context_text for term in answer_terms)

    def run_tests(self) -> Dict[str, Dict]:
        results = {}

        for idx, test in enumerate(self.test_cases):
            answer = self.pipeline.query(test["question"])
            contexts = self.pipeline.retriever.query(test["question"])

            results[f"test_{idx}"] = {
                "question": test["question"],
                "expected_test_answer": test["expected_answer"],
                "answer_received": answer,
                "answer_valid": bool(answer.strip()),
                "grounding_check": self._check_grounding(answer, contexts),
            }

        self._generate_report(results)
        return results

    def _generate_report(self, results: Dict):
        report_path = Path("test_report.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Test report saved to {report_path.absolute()}")

# Usage
if __name__ == "__main__":
    # Load test documents
    documents_base_path = Path("../baseline/data")
    doc_paths = [
        documents_base_path / "demo.txt",
        documents_base_path / "demo.md",
        documents_base_path / "demo.pdf"
    ]

    pipeline = Pipeline(
        document_paths = doc_paths,
        index_save_path="./sentence_embeddings_index",
    )
  # Your existing initialization
    tester = PipelineTester(pipeline, "testing/test_inputs.json")
    test_results = tester.run_tests()