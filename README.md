# Projects

Fine-tuning and benchmarking a retrieval model for European funded projects using the FP7, H2020, and HORIZON corpora.

The goal of this repository is to adapt a `SentenceTransformer` model based on ModernBERT so it retrieves project summaries better than a general-purpose encoder on this domain.

## What This Repo Does

This project covers the full retrieval pipeline:

1. Build an evaluation set from HORIZON projects.
2. Build a synthetic training set from FP7, H2020, and HORIZON data.
3. Fine-tune `joe32140/ModernBERT-base-msmarco` with cached MNRL plus Matryoshka training.
4. Benchmark baseline and fine-tuned checkpoints with `Recall@k` and `MRR@k`.

## Dataset Assumptions

The raw corpus is expected under a `data/` directory with this structure:

```text
data/
|-- fp7_projects/
|   `-- project.csv
|-- h2020_projects/
|   `-- project.csv
`-- horizon_projects/
    `-- project.csv
```

The loaders expect semicolon-separated CSVs and use the project fields already present in the official exports, especially columns such as `id`, `objective`, `keywords`, `topics`, and `frameworkProgramme`.

## Project Layout

```text
.
|-- benchmark.py
|-- generate_eval_dataset.py
|-- generate_train_dataset.py
|-- train.py
|-- utils.py
|-- pyproject.toml
|-- .env.template
`-- data/
```

## Environment Setup

Python `3.12+` is required.

Using `uv`:

```bash
uv sync
```

## Configuration

Create a local `.env` from `.env.template` and fill in the values you need:

```env
WANDB_API_KEY=
WANDB_PROJECT=
WANDB_ENTITY=
WANDB_MODE=
VLLM_ADDRESS=
```

Notes:

- `VLLM_ADDRESS` should point to the OpenAI-compatible vLLM server used to generate synthetic queries.
- `WANDB_*` variables are used by `train.py` for experiment tracking.

## End-to-End Workflow

### 1. Generate the Evaluation Set

This script samples HORIZON projects, generates one realistic search query per project with a local LLM, then expands positives using lexical search (BM25) plus a reranker.

```bash
python generate_eval_dataset.py
```

Output:

- `eval.jsonl`

Each line looks like:

```json
{"query": "battery recycling for electric vehicles", "positives": [101044526, 101064365]}
```

### 2. Generate the Training Set

This script excludes evaluation positives from the training corpus, samples a subset of projects, generates three types of synthetic user queries, and finds additional positives with BM25 plus reranking.

The synthetic queries are intentionally diverse:

- `short`: compact keyword-style searches, close to what a user would type when they already know the area they want.
- `medium`: more descriptive natural searches with clearer intent and a bit more semantic structure.
- `problem-oriented`: queries framed around the practical problem a user wants to solve rather than around formal project terminology.

This is useful because it exposes the retriever to different search behaviors instead of overfitting to a single query style. In practice, that usually leads to a model that is more robust to terse keyword searches, richer natural-language searches, and problem-driven retrieval.

```bash
python generate_train_dataset.py -p eval.jsonl -d data -k 20 -n 3 -pr 0.5
```

Arguments:

- `-p`, `--path`: path to the evaluation set.
- `-d`, `--path_data`: path to the raw corpus root.
- `-k`, `--k`: top-k candidates retrieved before reranking.
- `-n`, `--n_pos`: number of positives to keep per query.
- `-pr`, `--prop`: proportion of the corpus to sample for synthetic data generation.

Output:

- `train.jsonl`
- `trainset_pipe.log`

Each line looks like:

```json
{"query": "AI methods to detect phishing emails in small businesses", "positives": [24891, 11802, 8841]}
```

### 3. Fine-Tune the Retriever

Training is done with `SentenceTransformerTrainer`, `CachedMultipleNegativesRankingLoss`, and `MatryoshkaLoss`.

We use MNRL because it is a very effective objective for retrieval and contrastive representation learning. More specifically, we use the cached version of MNRL so we can handle large effective batch sizes without running into memory problems. This matters because large batch sizes are especially beneficial in contrastive learning setups like MNRL, where more in-batch negatives usually produce a stronger training signal.

On top of that, we add `MatryoshkaLoss` to encourage the model to remain strong even when we trim the generated embeddings to smaller dimensions at inference time. This helps ensure that the retriever still performs well when we need smaller vectors for cheaper storage, lower latency, or faster search.

```bash
python train.py -pe eval.jsonl -pt train.jsonl -d data -po outputs/run_01 -rn run_01 -bs 32 -cbs 8
```

Arguments:

- `-pe`, `--path_eval`: path to the evaluation set.
- `-pt`, `--path_train`: path to the training set.
- `-d`, `--path_data`: path to the raw corpus root.
- `-po`, `--path_output`: output directory for checkpoints and trainer artifacts.
- `-rn`, `--run_name`: run name for Weights & Biases.
- `-bs`, `--batch_size`: per-device train batch size.
- `-cbs`, `--cached_batch_size`: mini-batch size used by cached loss.

Output:

- checkpointed model artifacts under `outputs/...`

### 4. Benchmark a Model

You can benchmark either:

- a hub model with `--model`
- a local fine-tuned checkpoint with `--weights`

Benchmark a hub model:

```bash
python benchmark.py -p eval.jsonl -pd data -k 10 -m joe32140/ModernBERT-base-msmarco
```

Benchmark local weights:

```bash
python benchmark.py -p eval.jsonl -pd data -k 10 -w outputs/run_01
```

Run Matryoshka evaluation across dimensions:

```bash
python benchmark.py -p eval.jsonl -pd data -k 10 -w outputs/run_01 -mt
```

Reported metrics:

- `Recall@k`
- `MRR@k`

## Why This Is Useful

Even though we are seeing more and more capable foundation models, domain-specific fine-tuning is still a very good option and often improves results in practice.

General embedding models are often decent at semantic search, but domain-specific retrieval usually improves when:

- the training pairs resemble real user search behavior,
- the query distribution includes different search styles instead of a single synthetic pattern,
- positives are selected inside the target corpus,
- and the encoder is tuned on the language and structure of the target documents.

This repository is built around exactly that idea for European funded project discovery: adapt a strong base retriever to the language, topics, and search patterns that actually appear in this domain, while also making sure it keeps performing well when embeddings are trimmed for more efficient deployment.

## Disclaimer

This README was drafted with the help of Codex and reviewed by the author. The project code itself was written manually.