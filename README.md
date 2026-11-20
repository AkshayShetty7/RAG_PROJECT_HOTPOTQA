# Multi-Hop Question Answering using Retrieval-Augmented Generation (RAG)

## Project Overview

This project implements and evaluates a Retrieval-Augmented Generation (RAG) pipeline for multi-hop question answering using 100 questions from the HotpotQA dataset.

The system retrieves relevant evidence using multiple retrieval strategies and generates answers using a Groq-hosted Large Language Model (LLM).

Four retrieval approaches are evaluated:

- Dense Retrieval
- BM25 Retrieval
- Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
- Hybrid RRF with BGE Reranking

The objective is to compare retrieval effectiveness and evaluate answer quality using retrieval metrics, Exact Match, Token F1 and RAGAS.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Dataset | HotpotQA |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Vector Index | FAISS |
| Sparse Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Reranking | BGE Reranker |
| LLM | Groq API |
| Evaluation | Custom metrics and RAGAS |

## System Workflow

1. Load the 100 HotpotQA questions and their associated contexts.
2. Preprocess documents and extract sentences.
3. Split documents into retrieval chunks while preserving sentence identifiers.
4. Generate embeddings using BGE.
5. Build FAISS and BM25 indexes.
6. Retrieve relevant chunks using Dense, BM25 and Hybrid RRF methods.
7. Apply BGE reranking to improve retrieval ranking.
8. Generate answers using the Groq LLM.
9. Evaluate retrieval performance and generated answers.

## Experimental Results

**Dataset: HotpotQA | Evaluation Questions: 100**

### 1. Retrieval Performance

| Retrieval Method | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense Retrieval | 0.9292 | 0.9792 | 0.9525 | 0.9154 |
| BM25 | 0.7092 | 0.8908 | 0.8354 | 0.7499 |
| Hybrid RRF | 0.8942 | 0.9742 | 0.8992 | 0.8605 |
| **Hybrid RRF + BGE Reranker** | **0.9592** | 0.9742 | **0.9708** | **0.9391** |

**Best-performing retrieval configuration:** Hybrid RRF + BGE Reranker.

It achieved the highest Recall@5, MRR and nDCG@10 among the evaluated methods.


### 2. RAGAS Evaluation

| Metric | Mean Score |
|---|---:|
| Faithfulness | 0.8383 |
| Factual Correctness (Raw RAGAS) | 0.8702 |
| Context Recall | 0.9500 |




## Project Structure

```text
RAG_PROJECT/
│
├── data_processing/
│   ├── create_benchmark.py
│   ├── prepare_benchmark_documents.py
│   └── ...
│
├── rag/
│   ├── benchmark_chunker.py
│   ├── build_benchmark_indexes.py
│   ├── groq_generator.py
│   └── ...
│
├── evaluation/
│   ├── run_benchmark.py
│   ├── run_ragas.py
│   └── ...
│
├── datasets/
│   ├── hotpotqa/
│   └── benchmark_hotpotqa_100.json
│
├── data/
│   └── benchmark_indexes/
│
├── results/
│
├── tests/
│
├── requirements.txt
└── README.md
```



## Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG_PROJECT
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv rag_env
.\rag_env\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit your `.env` file or expose your API key publicly.

### 5. Prepare the HotpotQA Dataset

Place the HotpotQA development distractor dataset at:

```text
datasets/hotpotqa/dev_distractor.json
```

The 100-question benchmark should be available at:

```text
datasets/benchmark_hotpotqa_100.json
```

Use the benchmark creation script if you need to generate it from the source dataset.

### 6. Prepare Benchmark Documents

```bash
python -m data_processing.prepare_benchmark_documents
```

### 7. Run Chunking

```bash
python -m rag.benchmark_chunker
```

### 8. Run Unit Tests

```bash
python -m unittest discover -s tests -v
```

### 9. Build Retrieval Indexes

```bash
python -m rag.build_benchmark_indexes
```

### 10. Run the RAG Benchmark

```bash
python -m evaluation.run_benchmark --benchmark datasets/benchmark_hotpotqa_100.json --generate --output-dir results/hotpotqa_100/raw --sleep 2.2
```

This executes the retrieval evaluation and answer-generation pipeline.

### 11. Run RAGAS Evaluation

Use the generated answer file as input:

```bash
python -m evaluation.run_ragas --input results/hotpotqa_100/raw/answers_hybrid_rerank.jsonl --output results/hotpotqa_100/ragas/ragas_scores.csv
```

The RAGAS evaluation produces scores for Faithfulness, Factual Correctness and Context Recall.

## Generated Outputs

The benchmark execution produces:

| Output | Description |
|---|---|
| `retrieval_metrics.jsonl` | Retrieval metrics for each question and retrieval method |
| `answers_hybrid_rerank.jsonl` | Generated answers using hybrid retrieval and reranking |
| `ragas_scores.csv` | RAGAS evaluation results |
| `faiss.index` | Dense vector index |
| `bm25.pkl` | BM25 index |
| `chunks.json` | Indexed retrieval chunks and metadata |

## Evaluation Metrics

- **Recall@K:** Measures the proportion of relevant supporting evidence retrieved within the top K results.
- **MRR:** Measures how highly the first relevant result is ranked.
- **nDCG@10:** Measures ranking quality with relevance and position taken into account.
- **Exact Match:** Measures exact agreement between generated and reference answers.
- **Token F1:** Measures token-level precision and recall between generated and reference answers.
- **Faithfulness:** Measures whether generated claims are supported by retrieved context.
- **Factual Correctness:** Evaluates factual agreement with the reference answer.
- **Context Recall:** Measures how much reference information is represented in retrieved context.

