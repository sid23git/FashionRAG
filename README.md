# FashionRAG: A Modular Multimodal Fashion Retrieval System

A multimodal fashion retrieval system that retrieves relevant fashion images from a catalog using natural language queries. Built using **OpenAI CLIP** for image and text encoding, **FAISS** for efficient vector similarity search, and a fashion-specific **Query Expansion** module to improve retrieval quality.

---

## Features

- **CLIP Image Encoder** — Encodes fashion images into 512-dimensional normalized embeddings
- **CLIP Text Encoder** — Encodes natural language queries into the same embedding space
- **Offline Image Indexing** — Scans and indexes all images in a directory recursively
- **FAISS Vector Search** — Fast inner-product search using `IndexFlatIP` on L2-normalized embeddings
- **Natural Language Image Retrieval** — Retrieve fashion images using free-text queries
- **Fashion Query Expansion** — Expands query keywords with fashion-domain synonyms before encoding
- **Attribute-Aware Re-ranking** — Combines CLIP similarity with keyword-based attribute scores
- **Evaluation Pipeline** — Runs 10 diverse fashion queries and saves results to CSV
- **Modular Python Architecture** — Each pipeline stage is a standalone, reusable module
- **Logging** — Structured logging throughout all modules
- **Type Hints** — Fully annotated with Python type hints throughout the codebase

---

## Architecture

**Offline Indexing Pipeline**

```text
Images (data/raw_images/)
       ↓
CLIP Image Encoder
       ↓
Image Embeddings (float32, normalized)
       ↓
FAISS IndexFlatIP
       ↓
Saved: indexes/fashion.index
       indexes/image_metadata.pkl
```

**Online Retrieval Pipeline**

```text
User Query ("blue shirt")
       ↓
Fashion Query Expansion → "blue shirt apparel clothing top"
       ↓
CLIP Text Encoder
       ↓
FAISS Search (IndexFlatIP)
       ↓
Attribute-Aware Re-ranking
(final_score = 0.8 × CLIP score + 0.2 × attribute score)
       ↓
Top-K Retrieved Images
```

---

## Folder Structure

```text
glance-fashion-retrieval/
├── data/
│   └── raw_images/           # Input fashion images (.jpg, .jpeg, .png)
├── indexes/
│   ├── fashion.index         # FAISS index (auto-generated)
│   └── image_metadata.pkl    # Image metadata mapping (auto-generated)
├── outputs/
│   └── evaluation_results.csv
├── src/
│   ├── evaluation/
│   │   └── evaluate.py       # Evaluation pipeline
│   ├── indexing/
│   │   ├── build_index.py    # Offline indexing pipeline
│   │   └── image_encoder.py  # CLIP image encoder
│   └── retrieval/
│       ├── query_expander.py # Fashion keyword expansion
│       ├── reranker.py       # Attribute-aware re-ranker
│       └── text_retriever.py # FAISS-based text retrieval
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/glance-fashion-retrieval.git
cd glance-fashion-retrieval
pip install -r requirements.txt
```

---

## Requirements

```text
numpy>=1.22.0
pandas>=1.4.0
pillow>=9.0.0
torch>=1.11.0
torchvision>=0.12.0
transformers>=4.18.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.2
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
ipykernel>=6.9.0
```

---

## Usage

**Step 1 — Place your images in `data/raw_images/`** (`.jpg`, `.jpeg`, `.png` supported)

**Step 2 — Build the FAISS index:**

```bash
python main.py --build-index
```

**Step 3 — Search using a natural language query:**

```bash
python main.py --query "blue shirt"
```

Optionally control the number of results:

```bash
python main.py --query "red dress" --top-k 10
```

**Step 4 — Run the evaluation pipeline:**

```bash
python -m src.evaluation.evaluate
```

Results are saved to `outputs/evaluation_results.csv`.

---

## Example Output

```text
Original Query: "casual shirt"
Expanded Query: "casual everyday relaxed shirt apparel clothing top"

Rank   Score        Image Path
------------------------------------------------------------
1      0.186217     data\raw_images\fashion_item_1.jpg
2      0.165560     data\raw_images\fashion_item_3.jpg
3      0.165111     data\raw_images\fashion_item_4.jpg
4      0.161358     data\raw_images\fashion_item_0.jpg
5      0.161248     data\raw_images\fashion_item_2.jpg
```

---

## Technologies Used

| Technology              | Role                                   |
|-------------------------|----------------------------------------|
| Python 3.12             | Core language                          |
| PyTorch                 | Tensor operations and inference        |
| Hugging Face Transformers | CLIP model loading and processing    |
| OpenAI CLIP             | Image and text embedding               |
| FAISS                   | Approximate nearest-neighbor search    |
| NumPy                   | Embedding matrix operations            |
| Pillow                  | Image loading and preprocessing        |
| tqdm                    | Indexing progress bar                  |

---

## Improvement over Vanilla CLIP

The baseline CLIP retrieval approach was enhanced by introducing a lightweight, fashion-specific **Query Expansion** module that executes before text encoding. When a user submits a query, the expander maps known fashion keywords to a curated set of synonyms, producing a richer query sentence that better covers the semantic space of fashion images.

For example:
- `"casual shirt"` → `"casual everyday relaxed shirt apparel clothing top"`
- `"hoodie"` → `"hoodie sweatshirt"`
- `"jeans"` → `"jeans denim pants trousers"`

This improves the alignment between the text query and image embeddings without requiring any model fine-tuning or additional training data.

An attribute-aware re-ranking step further refines the results by combining the raw CLIP similarity score with a lightweight attribute match score:

```
final_score = 0.8 × CLIP score + 0.2 × attribute score
```

---

## Future Work

- Fine-tune CLIP on large-scale fashion datasets
- Support for larger, real-world fashion catalogs
- Streamlit web interface for interactive search
- Automatic attribute extraction from image metadata

---

## Author

**Siddharth Hefa**  
K. J. Somaiya School of Engineering
