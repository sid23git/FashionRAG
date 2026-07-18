"""Main entry point for the fashion retrieval system."""

import argparse
import logging

from src.indexing.image_encoder import ImageEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def demo_encode() -> None:
    """Runs Milestone 1: encode a single image and print its embedding."""
    image_path = "data/raw_images/fashion_item_0.jpg"
    encoder = ImageEncoder()
    embedding = encoder.encode(image_path)

    print(f"\nEmbedding shape : {embedding.shape}")
    print(f"First 10 values : {embedding[:10]}\n")


def build_index() -> None:
    """Runs Milestone 2: build the FAISS index from raw images."""
    from src.indexing.build_index import build_index as _build_index

    _build_index()


def text_search(query: str, top_k: int = 5) -> None:
    """Runs Milestone 3 & 4: search the index with a text query, expand, and re-rank."""
    from src.retrieval.text_retriever import TextRetriever
    from src.retrieval.reranker import AttributeReranker
    from src.retrieval.query_expander import QueryExpander

    expander = QueryExpander()
    expanded_query = expander.expand(query)

    retriever = TextRetriever()
    results = retriever.search(expanded_query, top_k=top_k)
    
    reranker = AttributeReranker()
    results = reranker.rerank(expanded_query, results)

    print(f"\nOriginal Query: \"{query}\"")
    print(f"Expanded Query: \"{expanded_query}\"")
    print(f"{'Rank':<6} {'Score':<12} {'Image Path'}")
    print("-" * 60)
    for r in results:
        print(f"{r['rank']:<6} {r['score']:<12.6f} {r['image_path']}")
    print()


def main() -> None:
    """Parses CLI arguments and dispatches to the appropriate action."""
    parser = argparse.ArgumentParser(
        description="Glance — Multimodal Fashion Retrieval System",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the FAISS index from images in data/raw_images/.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text query for fashion image retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    args = parser.parse_args()

    if args.build_index:
        build_index()
    elif args.query:
        text_search(args.query, top_k=args.top_k)
    else:
        demo_encode()


if __name__ == "__main__":
    main()