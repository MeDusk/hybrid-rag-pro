"""RAGAS evaluation pipeline for HybridRAG-Pro."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from langchain_openai import ChatOpenAI
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import settings
from src.evaluation.test_dataset import load_test_dataset


class RAGASEvaluator:
    """Runs RAGAS evaluation on the HybridRAG-Pro pipeline.

    Evaluates four core metrics:
        - faithfulness       : Is the answer grounded in the retrieved context?
        - answer_relevancy   : Is the answer relevant to the question?
        - context_precision  : Are the retrieved chunks precise?
        - context_recall     : Are all relevant chunks retrieved?

    Args:
        rag_chain: HybridRAGChain instance to evaluate.
        dataset_path: Path to the JSON Q&A test dataset.
        results_dir: Directory where JSON + CSV reports will be saved.
    """

    METRICS = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    def __init__(
        self,
        rag_chain,
        dataset_path: str = settings.EVAL_DATASET_PATH,
        results_dir: str = settings.EVAL_RESULTS_PATH,
    ) -> None:
        self.rag_chain = rag_chain
        self.dataset_path = dataset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("RAGASEvaluator initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        max_samples: Optional[int] = None,
        save: bool = True,
    ) -> dict:
        """Execute the full RAGAS evaluation pipeline.

        Steps:
            1. Load test Q&A dataset.
            2. Run each question through the RAG chain.
            3. Build RAGAS Dataset with answers and retrieved contexts.
            4. Compute RAGAS metrics.
            5. Save JSON + CSV report.

        Args:
            max_samples: Limit evaluation to first N samples (None = all 20).
            save: Whether to persist results to disk.

        Returns:
            Dict of metric_name -> float score.
        """
        samples = load_test_dataset(self.dataset_path)
        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Running RAGAS eval on {len(samples)} samples...")

        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        ground_truths: List[str] = []

        for i, sample in enumerate(samples):
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            logger.info(f"[{i+1}/{len(samples)}] Querying: '{question[:60]}'")

            try:
                result = self.rag_chain.query(question)
                answer = result["answer"]
                retrieved_contexts = [
                    src["content"] for src in result.get("sources", [])
                ]
            except Exception as e:
                logger.error(f"Query failed for sample {i+1}: {e}")
                answer = "Error during generation."
                retrieved_contexts = ["No context retrieved."]

            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts if retrieved_contexts else ["No context"])
            ground_truths.append(ground_truth)

        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        logger.info("Computing RAGAS metrics...")
        result = evaluate(
            dataset=ragas_dataset,
            metrics=self.METRICS,
            llm=ChatOpenAI(
                model=settings.RAGAS_LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
            ),
        )

        scores = {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall": round(float(result["context_recall"]), 4),
            "evaluated_at": datetime.utcnow().isoformat(),
            "num_samples": len(samples),
        }

        logger.info(f"RAGAS Results: {scores}")

        if save:
            self._save_results(scores, questions, answers, contexts, ground_truths)

        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_results(
        self,
        scores: dict,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> None:
        """Save evaluation results to JSON summary and detailed CSV."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON summary
        json_path = self.results_dir / f"ragas_scores_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        logger.info(f"RAGAS scores saved -> {json_path}")

        # CSV detailed report
        csv_path = self.results_dir / f"ragas_details_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["question", "ground_truth", "answer", "num_contexts"],
            )
            writer.writeheader()
            for q, gt, a, ctx in zip(questions, ground_truths, answers, contexts):
                writer.writerow({
                    "question": q,
                    "ground_truth": gt,
                    "answer": a,
                    "num_contexts": len(ctx),
                })
        logger.info(f"RAGAS details saved -> {csv_path}")
