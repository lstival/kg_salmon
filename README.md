# Fish Knowledge Graph Recommendation System

A high-performance research pipeline for Knowledge Graph (KG) analysis, Link Prediction, and Recommendation, with a focus on comparing traditional translational/bilinear models against modern Transformer-based and Self-Supervised Learning (SSL) architectures.

## Project Overview

This project implements a comprehensive comparison framework for Knowledge Graph Embeddings (KGE). It integrates the **PyKeen** library for state-of-the-art KGE models and **Sentence-Transformers (SBERT)** for semantic initialization based on entity labels.

The goal is to evaluate the quality of entity embeddings for recommendation tasks across different datasets (e.g., UMLS, FB15k-237, or custom CSV data).

## Key Features

- **Multi-Model Benchmark:** Compare TransE, RotatE, DistMult, ComplEx, and AutoSF.
- **NLP Integration:** Use Hugging Face's SBERT (`paraphrase-MiniLM-L6-v2`) to generate semantic feature vectors for entities.
- **Transformer Baseline:** A minimal Transformer Encoder architecture to serve as a pure NLP-based baseline.
- **Generic Data Loading:** Support for standard PyKeen datasets and custom `.csv` files in `(head, relation, tail)` format.
- **Advanced SSM Encoder:** A specialized State Space Model (SSM) based encoder for capturing deep structural path patterns via Self-Supervised Learning.

## Encoder Architectures

### 1. Translational Models
- **TransE:** The foundation of distance-based models. It treats relations as translations in the vector space ($\mathbf{h} + \mathbf{r} \approx \mathbf{t}$). Optimal for hierarchical structures.
- **RotatE:** Defines relations as rotations in a complex vector space ($\mathbf{t} = \mathbf{h} \circ \mathbf{r}$). Capable of modeling symmetry, anti-symmetry, and inversion.

### 2. Bilinear & Semantic Matching Models
- **DistMult:** Uses a diagonal weight matrix for relations, resulting in a symmetric scoring function. Efficient but limited in capturing directed information.
- **ComplEx:** Extends DistMult to the complex space, allowing it to model asymmetric relations by utilizing the Hermitian inner product.

### 3. Automated & Hybrid Models
- **AutoSF:** A model discovered via AutoML that utilizes a search space of bilinear scoring functions to find the optimal representation for a specific dataset.

### 4. Transformer & SSM (Our Approach)
- **Transformer Baseline:** Uses a standard multi-head attention mechanism to aggregate information across entity sequences.
- **SSM (State Space Model):** Our core research component. It uses a selective scan (similar to Mamba/S4) to capture long-range dependencies in KG paths with $O(L)$ complexity, significantly more efficient than Transformers for long sequences.

## How to Run

1. **Environment Setup:**
   ```bash
   conda activate ts2vec
   ```

2. **Configuration:**
   Modify `kg_project/config.py` to select your dataset and model parameters.

3. **Execution:**
   ```bash
   python main.py
   ```

4. **Custom CSV Data:**
   Pass the path to your `.csv` file in the `dataset_name` configuration field. The CSV should have three columns: `head`, `relation`, `tail`.

## Metrics
The project tracks standard KGE metrics:
- **MRR (Mean Reciprocal Rank)**
- **Hits@1, Hits@3, Hits@10**
- **Training duration and GFLOPS estimation**
