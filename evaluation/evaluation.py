#implement your evaluation code here
import os
import json
from datetime import datetime
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score
import re

from collections import defaultdict, Counter
from collections import Counter
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import List, Union, Any, Dict
from pathlib import Path

from baseline.pipeline import Pipeline
from specialization import PipelineSpeci
from utils import *
from utils.utils import get_doc_paths

model = SentenceTransformer("all-MiniLM-L6-v2")


class Evaluation:
    """
     A class to evaluate the performance of a question-answering (QA) pipeline.

     Attributes:
         pipeline (PipelineSpeci): The QA pipeline to evaluate.
         test_file_path (Union[str, Path]): Path to the JSON test file with QA pairs.
         test_cases (List[Dict]): Loaded list of test cases.
         embd_model (SentenceTransformer): Model used for computing semantic similarity.

     Methods:
         run_tests(): Executes test cases and stores the output.
         evaluate_model_performance(): Evaluates predictions using F1, ROUGE-L, BERTScore.
         semantic_similarity(): Computes cosine similarity between two text embeddings.
         evaluate_single_prediction(): Evaluates a single QA pair with multiple metrics.
         visualize_results(): Visualizes matched vs unmatched answers.
         run_pipeline(): Helper to manually run a pipeline query.
         print_testset(): Print human-readable format of the test set.
         print_result(): Print the result JSON in readable format.

     Example:
         >>> evaluation = Evaluation(pipeline, test_file_path="test_input_speci.json")
         >>> results = evaluation.run_tests()
         >>> evaluation.evaluate_model_performance("test_input_speci_result.json")
         >>> evaluation.visualize_results(matched, unmatched)
     """

    def __init__(self, pipeline: Union[PipelineSpeci], test_file_path: Union[str, Path] = "test_inputs_speci.json"):
        self.pipeline = pipeline
        self.test_file_path = test_file_path
        self.test_cases = self._load_test_cases(test_file_path)
        self.embd_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _load_test_cases(self, path: str) -> List[Dict]:
        """
        Load test cases from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            List[Dict]: A list of test case dictionaries.
        """
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"File {path} does not exist.")

    def run_tests(self, test_file_path: Union[str, Path] = None, result_path: Union[str, Path] = None) -> Dict[str, Dict]:
        """
        Run all loaded test cases on the pipeline and save the results.

        Args:
            test_file_path (Union[str, Path], optional): Path to a new test set (if different from the one loaded initially).
            result_path (Union[str, Path], optional): File path to store the generated results as JSON.

        Returns:
             Dict[str, Dict]: A dictionary containing the result for each test case.
        """
        results = {}
        f1_scores = []

        if test_file_path:
            if isinstance(test_file_path, str):
                # Convert the string to a Path object
                test_file_path = Path(test_file_path)
            self.test_file_path = Path(test_file_path)
            self.test_cases = self._load_test_cases(test_file_path)

        for idx, test in enumerate(self.test_cases):
            print(test["question"])
            answer = self.pipeline.query(test["question"])
            #contexts = self.pipeline.retriever.query(test["question"])
            ground_truth = test["ground_truth"]

            results[f"test_{idx}"] = {
                "question": test["question"],
                "ground_truth": ground_truth,
                "answer_received": answer,
                #"context": contexts.chunk_text,
                "source_type": test["source_type"],
                "difficulty": test["difficulty"],
                "question_type": test["question_type"],
            }

        print(f"\nTotal Test Cases Evaluated : {len(self.test_cases)}")
        print(f"---------------------------\n")

        self._generate_report(results, result_path)
        return results

    def _generate_report(self, results: Dict, result_path: Union[str, Path] = None):
        """
        Save evaluation results to a JSON file.

        Args:
            results (Dict): results to save.
            result_path (Union[str, Path], optional): Custom path to save the results file.
        """
        if result_path is None:
            os.makedirs("test", exist_ok=True)
            report_path = Path(f"test/{Path(self.test_file_path).stem}_result.json")
        else:
            #os.makedirs("test", exist_ok=True)
            report_path = Path(f"{result_path}.json")

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test result saved to {report_path.absolute()}")

    def semantic_similarity(self, expected: str, actual: str) -> float:
        """
        Compute cosine similarity between the embeddings of the expected and actual answers.

        Args:
            expected (str): Ground truth answer.
            actual (str): Model-generated answer.

        Returns:
            float: Cosine similarity score (0 to 1).
        """
        embeddings = model.encode([expected, actual], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    def _normalize_answer(self, s):
        """
        Normalize a string by lowercasing, removing punctuation, articles, and extra whitespace.

        Args:
            s (str): Input string.

        Returns:
            str: Normalized string.
        """

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            return re.sub(r'[^\w\s]', '', text)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _compute_f1_score(self, expected, predicted):
        """
        Compute token-level precision, recall, and F1 score.

        Args:
            expected (str): Ground truth answer.
            predicted (str): Model-generated answer.

        Returns:
            Tuple[float, float, float]: Precision, recall, and F1 score.
        """
        expected_tokens = self._normalize_answer(
            expected).split()  # F1 is a token-level metric, so we compare word-by-word, not characters or full sentences
        predicted_tokens = self._normalize_answer(predicted).split()

        common = Counter(expected_tokens) & Counter(predicted_tokens)
        num_same = sum(common.values())  # num_same counts how many tokens the model got right.

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(predicted_tokens)  # how much was correct
        recall = num_same / len(expected_tokens)  # how much did it cover
        f1 = 2 * precision * recall / (
                precision + recall)  # Harmonic mean balances precision and recall; high only if both are high
        return precision, recall, f1

    def evaluate_model_performance(self, test_report: Union[str, Path], output_dir: Union[str, Path] = "evaluation_results", threshold: float = 0.60):
        """
        Evaluate the QA pipeline using F1, ROUGE-L, and BERTScore. Group metrics by metadata fields.

        Args:
            test_report (Union[str, Path]): Path to the result JSON file generated by `run_tests`.
            output_dir (Union[str, Path], optional): Directory to save evaluation output files. Defaults to "evaluation_results".
            threshold (float, optional): BERTScore F1 threshold for matching. Defaults to 0.60.

        Returns:
            Tuple[int, int, List[Tuple], Dict]: Matched count, unmatched count, detailed results, and group-wise metrics.
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        test_report_data = self._load_test_cases(test_report)

        detailed_results = []
        total_precision = total_recall = total_f1 = total_rouge = 0
        matched = unmatched = 0
        question_pairs = []
        bert_expected = []
        bert_predicted = []

        group_metrics_raw = {
            "source_type": defaultdict(
                lambda: {"precision": 0, "recall": 0, "f1": 0, "rouge": 0, "bert_f1_sum": 0, "count": 0}),
            "difficulty": defaultdict(
                lambda: {"precision": 0, "recall": 0, "f1": 0, "rouge": 0, "bert_f1_sum": 0, "count": 0}),
            "question_type": defaultdict(
                lambda: {"precision": 0, "recall": 0, "f1": 0, "rouge": 0, "bert_f1_sum": 0, "count": 0})
        }

        for key, item in test_report_data.items():
            question = item["question"]
            ground_truth = item["ground_truth"]
            source_type = item["source_type"]
            difficulty = item["difficulty"]
            question_type = item["question_type"]
            generated_answer = item["answer_received"]

            precision, recall, f1 = self._compute_f1_score(ground_truth, generated_answer)
            rouge_score_val = scorer.score(ground_truth, generated_answer)["rougeL"].fmeasure

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_rouge += rouge_score_val

            bert_expected.append(ground_truth)
            bert_predicted.append(generated_answer)

            question_pairs.append(
                (question, ground_truth, generated_answer, source_type, difficulty, question_type, precision, recall,
                 f1, rouge_score_val)
            )

        P, R, F1 = bert_score(bert_predicted, bert_expected, lang='en', model_type='microsoft/deberta-xlarge-mnli',
                              verbose=True)
        bert_f1_list = F1.tolist()

        for i, (question, ground, generated, source_type, difficulty, question_type, precision, recall, f1,
                rouge) in enumerate(question_pairs):
            bert_f1 = bert_f1_list[i]
            if bert_f1 >= threshold:
                matched += 1
            else:
                unmatched += 1

            detailed_results.append(
                (question, ground, generated, source_type, difficulty, question_type, f1, rouge, bert_f1)
            )

            for category, value in [("source_type", source_type), ("difficulty", difficulty),
                                    ("question_type", question_type)]:
                group = group_metrics_raw[category][value]
                group["precision"] += precision
                group["recall"] += recall
                group["f1"] += f1
                group["rouge"] += rouge
                group["bert_f1_sum"] += bert_f1
                group["count"] += 1

        avg_precision = total_precision / len(detailed_results)
        avg_recall = total_recall / len(detailed_results)
        avg_f1 = total_f1 / len(detailed_results)
        avg_rouge = total_rouge / len(detailed_results)
        avg_bert_f1 = sum(bert_f1_list) / len(bert_f1_list)

        print(f"\nToken-level Evaluation Metrics:")
        print(f"Average Precision:    {avg_precision:.2f}")
        print(f"Average Recall:       {avg_recall:.2f}")
        print(f"Average F1 Score:     {avg_f1:.2f}")
        print(f"Average ROUGE-L:      {avg_rouge:.2f}")
        print(f"Average BERTScore F1: {avg_bert_f1:.2f}")

        group_metrics_avg = {"source_type": {}, "difficulty": {}, "question_type": {}}
        for category, values in group_metrics_raw.items():
            print(f"\n--- {category.upper()} ---")
            for label, metrics in values.items():
                count = metrics["count"]
                if count == 0:
                    continue
                avg_p = metrics["precision"] / count
                avg_r = metrics["recall"] / count
                avg_f1 = metrics["f1"] / count
                avg_rouge = metrics["rouge"] / count
                avg_bert_f1 = metrics["bert_f1_sum"] / count
                print(
                    f"{label:>12}: F1={avg_f1:.2f}, ROUGE={avg_rouge:.2f}, BERT F1={avg_bert_f1:.2f}, Precision={avg_p:.2f}, Recall={avg_r:.2f}")
                group_metrics_avg[category][label] = {
                    "precision": avg_p,
                    "recall": avg_r,
                    "f1": avg_f1,
                    "rouge": avg_rouge,
                    "bert_f1": avg_bert_f1,
                    "count": count
                }

        os.makedirs(output_dir, exist_ok=True)

        group_metrics_path = os.path.join(output_dir, "group_metrics.json")
        with open(group_metrics_path, "w", encoding="utf-8") as f:
            json.dump(group_metrics_avg, f, indent=4)
        print(f"\nGroup metrics saved to {group_metrics_path}")

        detailed_json_path = os.path.join(output_dir, "detailed_results.json")
        detailed_dicts = [
            {
                "question": q,
                "ground_truth": gt,
                "answer_received": gen,
                "source_type": srctyp,
                "difficulty": diffi,
                "question_type": quetyp,
                "f1": round(f1, 3),
                "rouge_l": round(rouge, 3),
                "bert_f1": round(bert, 3)
            }
            for q, gt, gen, srctyp, diffi, quetyp, f1, rouge, bert in detailed_results
        ]
        with open(detailed_json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_dicts, f, indent=4, ensure_ascii=False)
        print(f"Detailed results saved to {detailed_json_path}")

        summary_path = os.path.join(output_dir, "summary.json")
        summary_data = {
            "average_precision": round(avg_precision, 3),
            "average_recall": round(avg_recall, 3),
            "average_f1": round(avg_f1, 3),
            "average_rouge": round(avg_rouge, 3),
            "average_bert_f1": round(avg_bert_f1, 3),
            "matched": matched,
            "unmatched": unmatched
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        print(f"Summary saved to {summary_path}")

        return matched, unmatched, detailed_results, group_metrics_avg

    def visualize_results(self, matched: int, unmatched: int):
        """
         Generate a pie chart to visualize matched vs unmatched answers.

         Args:
             matched (int): Number of answers that passed the threshold.
             unmatched (int): Number of answers that failed the threshold.
         """
        labels = [f"Matched ({matched})", f"Unmatched ({unmatched})"]
        counts = [matched, unmatched]
        colors = ["#5cb85c", "#d95f02"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops={'fontsize': 12}
        )
        plt.title("LLM Evaluation")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def evaluate_single_prediction(self, question, expected_answer, generated_answer, threshold=0.25):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        precision, recall, f1 = self.compute_f1_score(expected_answer, generated_answer)

        rouge = scorer.score(expected_answer, generated_answer)["rougeL"].fmeasure

        P, R, F1 = bert_score([generated_answer], [expected_answer],
                              lang='en', model_type='microsoft/deberta-xlarge-mnli', verbose=False)
        bert_f1 = F1.tolist()[0]

        is_match = bert_f1 >= threshold

        return {
            "question": question,
            "ground_truth": expected_answer,
            "answer_received": generated_answer,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rouge_l": rouge,
            "bert_f1": bert_f1,
            "bert_match_threshold": threshold,
            "is_match": is_match
        }

    @staticmethod
    def print_testset(testset_path: Union[str, Path]):
        """
         Print each test case from a test set file.

         Args:
             testset_path (Union[str, Path]): Path to the JSON test file.
         """
        with open(testset_path, "r", encoding="utf-8") as f:
            testset_data = json.load(f)

        for entry in testset_data:
            question = entry["question"]
            ground_truth = entry["ground_truth"]
            source_type = entry["source_type"]
            difficulty = entry["difficulty"]
            question_type = entry["question_type"]

            print(
                f"Q: {question}\n A: {ground_truth}\n S: {source_type} \n D: {difficulty} \n QT: {question_type} \n")

    @staticmethod
    def print_result(testset_path: Union[str, Path]):
        """
        Print each result case from an evaluation result JSON file.

        Args:
            testset_path (Union[str, Path]): Path to the result JSON file.
        """
        with open(testset_path, "r", encoding="utf-8") as f:
            testset_data = json.load(f)

        for key, entry in testset_data.items():
            question = entry["question"]
            ground_truth = entry["ground_truth"]
            source_type = entry["source_type"]
            difficulty = entry["difficulty"]
            question_type = entry["question_type"]

            print(
                f"Q: {question}\n  A: {ground_truth}\n S: {source_type} \n D: {difficulty} \n QT: {question_type} \n")
            
# Usage
if __name__ == "__main__":
    # **************** Step1=> Initial Setup: Build the index or add the documents **************** #
    print(f"********Step2: Initial Setup ********\n")
    directory_path = Path("../baseline/data/findoc_xsm_samples")
    doc_path_list = get_doc_paths(directory_path)
    print(f"list of documents:{doc_path_list}\n")
    index_save_path = Path("vector_index_speci_xsm")
    pipeline = PipelineSpeci(
        document_paths=doc_path_list,
        index_save_path=index_save_path,
        groq_model=True,
        #rebuild_index=True    #to build index or add documents for the first time set the rebuild_index to True, default is false
    )
    #sample QA test, uncomment to check
    """
    query = "How many shares of Apple's common stock were issued and outstanding as of October 18, 2024?"
    query = input(f"Please enter your question: \n eg: {query}") 
    answer = pipeline.query(question=query)
    print(f"Q: {query}\nA: {answer}")
    """

    # **************** Step2=> Test and create result set **************** #
    print(f"********Step2: Use test set and create result set********\n")
    # Load test documents
    testset_path = Path("test/test_input_speci.json")
    #Evaluation.print_testset(testset_path)                                         # Uncomment to print and see how test set look,
    evaluation = Evaluation(pipeline, test_file_path=testset_path)
    result_path = Path(f"./test/test_input_speci_result")
    evaluation.run_tests(result_path=result_path, test_file_path=testset_path)     #uncomment and run to create a result using the test set

    # **************** Use the results to create evaluation report **************** #
    print(f"********Step2: Using the created result to create evaluation report ********\n")
    Evaluation.print_result(f"{result_path}.json")                                # Uncomment to print and see how the result set look, this will be used for evaluation
    matched, unmatched, detailed_results, group_metrics = evaluation.evaluate_model_performance(f"{result_path}.json")
    #evaluation.visualize_results(matched, unmatched)                              #uncomment and run to create a result using the test set