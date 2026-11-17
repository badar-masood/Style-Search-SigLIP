# Style Search – Hybrid Information Retrieval Engine

A multimodal furniture discovery and recommendation system that combines classical information retrieval, visual similarity search, and vision-language model reasoning to find furniture that matches both a user's search query and the visual style of their room.

## Overview

Traditional product search engines primarily rely on keywords and product metadata. However, finding furniture that matches a particular room requires more than matching product names or descriptions.

**Style Search** addresses this problem through a three-stage hybrid retrieval pipeline:

1. **Classical Retrieval (BM25):** Finds products relevant to the user's text query using product names, categories, and keywords.
2. **Visual Retrieval (SigLIP):** Ranks the retrieved products according to their visual similarity to the user's room.
3. **Agentic Reranking (Qwen2-VL):** Evaluates the compatibility of shortlisted furniture with the room, considering both style and functionality.

The result is a ranked list of furniture recommendations that considers what the user wants and what visually fits their space.

## Architecture

<pre>
User Text Query + Room Image
             |
             v
    Stage 1: BM25 Retrieval
    (Keyword & Category Filtering)
             |
       Top 200 Candidates
             |
             v
    Stage 2: SigLIP Retrieval
    (Visual Similarity Ranking)
             |
        Top 20 Candidates
             |
             v
    Stage 3: Qwen2-VL Style Agent
    (Visual Compatibility Analysis)
             |
             v
       Final Top 5 Results
</pre>

## Key Components

### 1. Classical Retrieval – BM25

The first stage uses the BM25 ranking algorithm to retrieve relevant furniture products.

The search index is built using:
- Product categories
- Product names
- Product keywords

Text preprocessing includes lowercasing, punctuation removal, and whitespace normalization.

An optional category filter helps prevent irrelevant product types from appearing in search results.

**Purpose:** Identify functionally relevant products before visual ranking.

### 2. Visual Retrieval – SigLIP

The second stage uses Google's pretrained SigLIP model:

`google/siglip-so400m-patch14-384`

The system compares the user's room image against precomputed furniture image embeddings.

Key operations include:
- Image preprocessing and feature extraction.
- L2 normalization of image embeddings.
- Cosine similarity calculation.
- Ranking products by visual similarity.
- Filtering the search space using candidates from Stage 1.

The visual retrieval stage selects the top 20 candidates for further analysis.

**Purpose:** Find furniture that visually complements the user's room.

### 3. Agentic Reranking – Qwen2-VL

The final stage uses `Qwen/Qwen2-VL-2B-Instruct` with a fine-tuned LoRA adapter.

The model receives:
- The user's room image.
- A candidate furniture image.
- A prompt asking it to analyze the stylistic compatibility of the furniture with the room.

It generates a relevance score and a textual rationale for each candidate.

Implementation details:
- Four-bit quantization using BitsAndBytes.
- NF4 quantization configuration.
- LoRA adapter integration through PEFT.
- Batched inference for candidate evaluation.
- Structured score and rationale extraction.
- Final ranking based on agent-generated scores.

**Purpose:** Go beyond visual similarity and assess contextual furniture compatibility.

## Evaluation

The notebook includes an evaluation pipeline comparing SigLIP-only visual ranking against the Qwen2-VL agent's reranking.

The evaluation uses:
- A labeled dataset containing room-product pairs.
- Teacher-generated relevance scores from Qwen-7B.
- Random sampling of up to 30 rooms.
- NDCG@10 to compare ranking quality.

Two approaches are evaluated:

| Approach | Ranking method |
|---|---|
| Stage 2 | SigLIP visual similarity |
| Stage 3 | Qwen2-VL agent reranking |

The evaluation calculates the average NDCG@10 for both approaches and their relative improvement.

*Note: Actual evaluation scores should be added after running the complete evaluation cell.*

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Classical Retrieval | BM25, rank-bm25 |
| Vision Model | SigLIP |
| Vision-Language Model | Qwen2-VL |
| Model Fine-tuning | LoRA, PEFT |
| Quantization | BitsAndBytes, NF4 |
| Data Processing | Pandas, NumPy |
| Image Processing | OpenCV, Pillow |
| Evaluation | scikit-learn, NDCG |
| Environment | Kaggle Notebooks |

## Datasets and Model Dependencies

The notebook uses the following Kaggle input resources:

- `style-search-siglip-generated-pairs`
  - `clean_furniture_data.csv`
  - `furniture_style_db.pt`
  - `clean_room_data.csv`

- `style-search-qwen2-5vl-generated-pairs`
  - Fine-tuned Qwen2-VL LoRA adapter
  - `labeled_dataset_shard_1.jsonl`

- `house-rooms`
  - Room image dataset

The pretrained Qwen2-VL model is loaded from Hugging Face.

## Installation

Install the required dependencies:

```bash
pip install rank_bm25
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate bitsandbytes peft datasets trl qwen-vl-utils
pip install pandas numpy torch pillow opencv-python
pip install scikit-learn scipy matplotlib seaborn tqdm
```

A CUDA-enabled GPU is recommended for running the SigLIP and Qwen2-VL inference stages.

## Running the Project

To run the project:

1. Open the Kaggle notebook.
2. Select **Copy & Edit**.
3. Attach the required Kaggle datasets.
4. Enable GPU acceleration.
5. Run the notebook cells in sequence.
6. Explore the retrieval pipeline and compare the final furniture recommendations.

## Example Search

A user can provide a room image and a text query such as:

`I need a modern Chair`

The system then:
1. Retrieves relevant chair products using BM25.
2. Ranks the candidates according to their visual compatibility with the room using SigLIP.
3. Uses Qwen2-VL to evaluate the shortlisted furniture and generate relevance scores with explanations.
4. Returns the top-ranked recommendations.

## Project Objective

The goal of Style Search is to demonstrate how combining classical information retrieval, multimodal embeddings, and vision-language model reasoning can create a more context-aware product discovery experience.

Rather than relying on keywords or visual similarity alone, the system combines product relevance, room aesthetics, and contextual reasoning to improve furniture recommendations.
