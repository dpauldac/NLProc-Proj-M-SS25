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
from typing import List, Union, Any
from pathlib import Path

model = SentenceTransformer("all-MiniLM-L6-v2")


class Evaluation:

    def __init__(self):
        pass

    def run_evaluation(self, retriever, generator):
        """
        This evaluation runs over test questions and logs the output to a date specific JSON file.
        """

        test_file = "evaluation/tests/test_sample_question_answer.json"
        log_dir = "evaluation/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, datetime.now().strftime("%d-%m-%Y") + ".json")

        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_entries = json.load(f)
        else:
            log_entries = []

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        for item in test_data:
            question = item["question"]
            task = item.get("task", "qa")
            options = item.get("options", None)

            if "context" not in item:
                if task in ["qa", "classification"]:
                    retrieved_chunks, _ = retriever.query(question, k=1)
                    context = "\n\n".join(retrieved_chunks)
                else:
                    retrieved_chunks = []
                    context = ""
            else:
                context = item["context"]
                retrieved_chunks = [context]

            prompt = generator.build_prompt(
                context=context,
                task_input=question,
                mode=task,
                options=options
            )
            answer = generator.generate_answer(prompt, mode=task, options=options)
            if task == "classification":
                answer = answer.strip().lower()
                if "offensive" in answer:
                    answer = "Offensive"
                elif "non-offensive" in answer:
                    answer = "Non-offensive"
                else:
                    answer = "Unclear"

            if task == "mcq":
                answer = 'a'

            log_entry = {
                "question": question,
                "task": task,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "context": context,
                "generated_answer": answer,
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "group_id": "Team NNN"
            }

            log_entries.append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=4)

        print(f"Evaluation complete. Log written to {log_file}")

    def semantic_similarity(self, expected: str, actual: str) -> float:
        """
        Compute cosine similarity between expected and actual answer embeddings.
        Returns a float between 0 and 1.
        """
        embeddings = model.encode([expected, actual], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    def evaluate_bert_score(expected_list, predicted_list, lang='en', model_type='microsoft/deberta-xlarge-mnli'):
        """
        Compute BERTScore between lists of expected and predicted answers.

        Args:
            expected_list (List[str]): Ground truth answers.
            predicted_list (List[str]): Model-generated answers.
            lang (str): Language code (default: 'en').
            model_type (str): Model to use for BERTScore (default: DeBERTa MNLI, good for English).

        Returns:
            avg_precision, avg_recall, avg_f1, all_f1s: Averages and per-sample F1s.
        """
        assert len(expected_list) == len(predicted_list), "Expected and predicted lists must be the same length."

        P, R, F1 = bert_score(predicted_list, expected_list, lang=lang, model_type=model_type, verbose=True)

        avg_precision = P.mean().item()
        avg_recall = R.mean().item()
        avg_f1 = F1.mean().item()

        return avg_precision, avg_recall, avg_f1, F1.tolist()

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            return re.sub(r'[^\w\s]', '', text)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1_score(self, expected, predicted):
        """Compute token-level precision, recall, and F1 score."""
        expected_tokens = self.normalize_answer(
            expected).split()  # F1 is a token-level metric, so we compare word-by-word, not characters or full sentences
        predicted_tokens = self.normalize_answer(predicted).split()

        common = Counter(expected_tokens) & Counter(predicted_tokens)
        num_same = sum(common.values())  # num_same counts how many tokens the model got right.

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(predicted_tokens)  # how much was correct
        recall = num_same / len(expected_tokens)  # how much did it cover
        f1 = 2 * precision * recall / (
                    precision + recall)  # Harmonic mean balances precision and recall; high only if both are high
        return precision, recall, f1

    def visualize_results(self, matched: int, unmatched: int):
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
    def print_testset(testset: Union[str, Path]):
        with open(testset, "r", encoding="utf-8") as f:
            testset_data = json.load(f)

        for entry in testset_data:
            question = entry["question"]
            ground_truth = entry["ground_truth"]
            source_type = entry["source_type"]
            difficulty = entry["difficulty"]
            question_type = entry["question_type"]

            print(
                f"Q: {question}\n  A: {ground_truth}\n S: {source_type} \n D: {difficulty} \n QT: {question_type} \n")

    @staticmethod
    def print_result(testset: Union[str, Path]):
        with open(testset, "r", encoding="utf-8") as f:
            testset_data = json.load(f)

        for key, entry in testset_data.items():
            question = entry["question"]
            ground_truth = entry["ground_truth"]
            source_type = entry["source_type"]
            difficulty = entry["difficulty"]
            question_type = entry["question_type"]

            print(
                f"Q: {question}\n  A: {ground_truth}\n S: {source_type} \n D: {difficulty} \n QT: {question_type} \n")

    def evaluate_model_performance2(self, test_report: Union[str, Path], threshold: float = 0.60):
        from collections import defaultdict
        from rouge_score import rouge_scorer
        import json, os
        from bert_score import score as bert_score

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        with open(test_report, "r", encoding="utf-8") as f:
            test_report_data = json.load(f)

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

            precision, recall, f1 = self.compute_f1_score(ground_truth, generated_answer)
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

        output_dir = "evaluation_results2"
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

# Usage
if __name__ == "__main__":
    # Load test documents
    #test_results = tester.run_tests()
    #Evaluation.print_testset("test/test_input_speci_new.json")
    #Evaluation.print_result("test/test_report_speci_new_for_metrics.json")
    evaluation = Evaluation()
    matched, unmatched, detailed_results, group_metrics = evaluation.evaluate_model_performance2("test/test_report_speci_new_for_metrics.json")
    evaluation.visualize_results(matched, unmatched)