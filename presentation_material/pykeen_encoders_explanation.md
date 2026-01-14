# PyKeen Encoder Integration and Text-to-Embedding Pipeline

This document provides a technical overview of the Knowledge Graph (KG) encoders imported from PyKeen and describes the internal mechanics of how PyKeen transforms raw KG data (textual labels) into numerical representations suitable for Machine Learning and NLP tasks.

## 1. Encoders Imported from PyKeen

The current project utilizes several core Knowledge Graph Embedding (KGE) models from the PyKeen library, as defined in [kg_project/models/encoders.py](kg_project/models/encoders.py):

*   **TransE (Translational Embeddings):** A distance-based model where relations are represented as translations in the embedding space: $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$.
*   **RotatE:** Models relations as rotations in a complex vector space: $\mathbf{t} = \mathbf{h} \circ \mathbf{r}$, effectively capturing symmetry, anti-symmetry, and inversion patterns.
*   **DistMult:** A bilinear model using a diagonal matrix for relations, capturing symmetric interactions: $f(h, r, t) = \mathbf{h}^\top \text{diag}(\mathbf{r}) \mathbf{t}$.
*   **ComplEx (Complex Embeddings):** Extends DistMult to the complex domain to handle asymmetric relations by using the Hermitian inner product.
*   **AutoSF:** A model discovered via Automated Machine Learning (AutoML) that searches for optimal scoring functions for relation representation.

## 2. Text-to-Embedding Pipeline: From Labels to Tensors

For an NLP and ML specialist, the transition from discrete string labels to continuous vector spaces in PyKeen follows a structured multi-stage pipeline.

### A. The Mapping Phase (`TriplesFactory`)
PyKeen uses a `TriplesFactory` to manage the conversion of raw text labels into a canonical integer-based representation.
1.  **Vocabulary Construction:** All unique entity and relation strings are collected from the input dataset.
2.  **Bijective Mapping:** A dictionary mapping $ \phi : \text{Label} \to \{0, \dots, N-1\} $ is created.
3.  **Integer Triples:** The raw triples $(s, p, o)$ are converted into a tensor of shape $(M, 3)$ containing these mapped IDs.

### B. Numerical Representation (The Embedding Layer)
Once IDs are generated, PyKeen utilizes different strategies to convert these integers into vectors:

#### 1. Shallow Embeddings (Lookup Tables)
By default, the models imported above (TransE, DistMult, etc.) use **Shallow Embeddings**.
*   **Mechanism:** An `nn.Embedding(num_entities, dim)` layer acts as a weight matrix $\mathbf{E} \in \mathbb{R}^{|\mathcal{E}| \times d}$.
*   **Access:** For an entity ID $i$, the vector is retrieved via a simple indexing operation (one-hot multiplication optimization): $\mathbf{e}_i = \text{Lookup}(\mathbf{E}, i)$.

#### 2. Text-Based Representations (LM-based Encoders)
In more advanced NLP-centric configurations (often used when entity names or descriptions are informative), PyKeen integrates with HuggingFace `transformers`:
*   **Tokenization:** Entity labels are passed through a tokenizer (e.g., BERT's WordPiece or RoBERTa's BPE).
*   **Encoding:** The sequence of tokens is processed by a pre-trained Language Model (LM).
*   **Pooling:** The resulting contextualized hidden states are pooled (e.g., taking the `[CLS]` token or mean pooling) to produce a single vector $\mathbf{z} \in \mathbb{R}^{d_{LM}}$.
*   **Projection:** Often, a linear layer projects $d_{LM}$ to the desired KG embedding dimension $d$.

### C. The Workflow Summary
1.  **Input:** Textual triples, e.g., `("Salmon", "is_a", "Fish")`.
2.  **Mapping:** `(42, 7, 105)` based on the global index.
3.  **Retrieval:** The `forward` pass of the model uses these indices to fetch vectors from the `Representation` modules.
4.  **Scoring:** The specific encoder logic (e.g., TransE's $\| \mathbf{h} + \mathbf{r} - \mathbf{t} \|$) computes the plausibility score.

## 3. Implementation in This Project

The project wraps these PyKeen components in [kg_project/engine/trainer.py](kg_project/engine/trainer.py) using the High-Level `pipeline` entry point, which abstracts the `TriplesFactory` creation and model initialization. The `KGEncoderFactory` in [models/encoders.py](models/encoders.py) provides a unified interface to instantiate these models with specific `embedding_dim` configurations.
