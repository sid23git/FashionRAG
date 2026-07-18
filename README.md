# Glance Fashion Retrieval System

A state-of-the-art multi-modal fashion image retrieval system utilizing deep learning embeddings (e.g., CLIP, ResNet) and high-performance vector search (FAISS) to match text queries or reference images to fashion items.

---

## 🌟 Key Features

*   **Multi-Modal Retrieval**: Search for fashion items using either natural language text queries (e.g., "flowy red summer dress") or image queries (image-to-image search).
*   **Deep Learning Embeddings**: Utilizes pretrained models (like OpenAI's CLIP or ResNet backbones) to extract rich semantic features from fashion images.
*   **Fast Vector Search**: Integrates FAISS (Facebook AI Similarity Search) for sub-millisecond similarity search over large-scale catalogs.
*   **Reranking Pipeline**: Refines initial search results using high-accuracy cross-encoders or multi-attribute matching (e.g., color, category).
*   **Evaluation Framework**: Provides quantitative evaluation capabilities using standard retrieval metrics (Recall@K, MRR, NDCG).

---

## 📁 Repository Structure

```text
glance-fashion-retrieval/
│
├── data/
│   ├── raw_images/          # Original catalog fashion images
│   ├── processed_images/    # Resized, normalized, or augmented images
│   └── metadata/            # CSV/JSON catalog descriptions, colors, categories
│
├── src/
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── image_encoder.py # Model loader and image embedding extractor
│   │   └── build_index.py   # Script to generate & serialize the FAISS index
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_encoder.py # Multi-modal query encoder (Text / Image)
│   │   ├── search.py        # K-Nearest Neighbors search against index
│   │   └── reranker.py      # Rescoring and reranking matching candidates
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py      # Computation of MRR, Recall@K, etc.
│   │
│   └── utils/
│       └── __init__.py      # Helper utilities (logging, config loader, file I/O)
│
├── indexes/                 # Serialized FAISS indices (.index files)
│
├── notebooks/
│   └── exploration.ipynb   # Jupyter notebook for exploratory data analysis
│
├── outputs/                 # Query results, visual plots, and evaluation reports
│
├── tests/                   # Pytest suite
│
├── requirements.txt         # Dependencies list
├── README.md                # Project documentation
├── .gitignore               # Git exclusion settings
└── main.py                  # CLI entry point to run indexing, search, or evaluation
```

---

## 🚀 Getting Started

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

1. Place your raw fashion images under `data/raw_images/`.
2. Add your product metadata (e.g. `catalog.csv` containing product IDs, paths, and text descriptions) under `data/metadata/`.

### 3. Build the Index

Run the indexing pipeline to preprocess images, generate embeddings, and build the FAISS vector index:

```bash
python main.py --mode index --data_dir data/raw_images --output_dir indexes/
```

### 4. Perform Search

Search the catalog using a text query or a path to a query image:

```bash
# Text query
python main.py --mode search --query "blue denim jacket" --top_k 5

# Image query
python main.py --mode search --query_image path/to/query_shoe.jpg --top_k 5
```

### 5. Evaluate the Model

Run the evaluation suite against a labeled validation set to compute metrics:

```bash
python main.py --mode evaluate --test_labels data/metadata/val_pairs.csv
```

---

## 🧠 Technology Stack

*   **PyTorch**: Framework for deep learning computations.
*   **HuggingFace Transformers**: Used for loading CLIP (Contrastive Language-Image Pretraining) models.
*   **FAISS**: Facebook AI Similarity Search for efficient dense vector search.
*   **Pillow**: For image loading and preprocessing.
*   **Pandas & NumPy**: For metadata handling and vector operations.
