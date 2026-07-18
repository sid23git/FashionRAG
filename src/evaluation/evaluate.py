"""Evaluation module for the fashion retrieval system.

Runs a set of predefined queries through the full retrieval pipeline
(expansion, vector search, and reranking) and saves the results to a CSV file.
"""

import csv
import logging
from pathlib import Path
from typing import List

from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import AttributeReranker
from src.retrieval.text_retriever import TextRetriever

logger = logging.getLogger(__name__)


def evaluate_pipeline(output_csv: str = "outputs/evaluation_results.csv") -> None:
    """Evaluates the retrieval pipeline on a set of diverse queries.

    Args:
        output_csv: Path to save the evaluation results in CSV format.
    """
    # 1. Initialize pipeline components
    logger.info("Initializing evaluation pipeline...")
    expander = QueryExpander()
    retriever = TextRetriever()
    reranker = AttributeReranker()

    # 2. Define diverse queries
    queries: List[str] = [
        "blue shirt",
        "formal office wear",
        "casual outfit",
        "red dress",
        "denim jeans",
        "black jacket",
        "white sneakers",
        "hoodie",
        "summer clothing",
        "business attire",
    ]

    # 3. Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Run queries and save results
    logger.info("Running evaluation on %d queries...", len(queries))
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Original Query", "Expanded Query", "Rank", "Score", "Image Path"])

        for original_query in queries:
            print(f"\nEvaluating: '{original_query}'")
            
            # Pipeline execution
            expanded_query = expander.expand(original_query)
            print(f"Expanded Query: '{expanded_query}'")
            
            results = retriever.search(expanded_query, top_k=5)
            reranked_results = reranker.rerank(expanded_query, results)
            
            print(f"{'Rank':<6} {'Score':<10} {'Image Path'}")
            print("-" * 50)
            
            for res in reranked_results:
                rank = res["rank"]
                score = res["score"]
                image_path = res["image_path"]
                
                print(f"{rank:<6} {score:<10.6f} {image_path}")
                
                writer.writerow([
                    original_query,
                    expanded_query,
                    rank,
                    f"{score:.6f}",
                    image_path
                ])

    logger.info("Evaluation complete. Results saved to '%s'.", output_csv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    evaluate_pipeline()
