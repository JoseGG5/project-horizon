import argparse
import logging
from typing import Iterable, Sequence

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import load_projects, load_eval_set


def plot_mrl_metric(y_values: list[float], metric_name: str, dims: list[int]):
    """
    Plot MRL dimensions vs metric.

    Parameters
    ----------
    y_values : list[float]
        Metric values for dimensions [128, 384, 512, 768]
    metric_name : str
        "Recall" or "MRR"
    dims : list[int]
        List containing the dimensions of matryoshka used in training
    """

    if len(y_values) != len(dims):
        raise ValueError("y_values must contain exactly 4 values.")

    if "Recall" not in metric_name and "MRR" not in metric_name:
        raise ValueError("metric_name incorrect, should contain Recall or MRR")
    
    ylabel = metric_name + " (higher is better)"
    title = f"MRL Dimensions vs {metric_name}"

    plt.figure(figsize=(10, 6))

    plt.plot(
        dims,
        y_values,
        marker="o",
        markersize=10,
        linewidth=2.5,
        alpha=0.9
    )

    # Labels over points
    for x, y in zip(dims, y_values):
        plt.text(
            x,
            y + 0.005,
            f"{y:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.xticks(dims)
    plt.xlabel("Embedding dimensions", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=18, weight="bold")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def compute_recall_at_k(top_k_ids: Sequence[int], positives: set[int]) -> float:
    """
    Compute Recall@k for a ranked list of retrieved document ids.

    Parameters
    ----------
    top_k_ids : Sequence[int]
        Ranked list of retrieved document ids truncated at k.
    positives : set[int]
        Set of relevant document ids for the query.

    Returns
    -------
    float
        Recall@k score. Returns 0.0 if no positives are provided.
    """
    if not positives:
        return 0.0
    retrieved_relevant = len(set(top_k_ids) & positives)
    return retrieved_relevant / len(positives)


def compute_mrr_at_k(top_k_ids: Sequence[int], positives: set[int]) -> float:
    """
    Compute MRR@k (Mean Reciprocal Rank contribution) for one query.

    Parameters
    ----------
    top_k_ids : Sequence[int]
        Ranked list of retrieved document ids truncated at k.
    positives : set[int]
        Set of relevant document ids for the query.

    Returns
    -------
    float
        Reciprocal rank of the first relevant retrieved document.
        Returns 0.0 if no relevant document is found in top-k.
    """
    for rank, doc_id in enumerate(top_k_ids, start=1):
        if doc_id in positives:
            return 1.0 / rank
    return 0.0


def get_top_k_ids(
    model,
    query: str,
    doc_ids: Sequence[int],
    doc_embs,
    k: int,
    truncate_dim: int | None = None,
) -> list[int]:
    """
    Retrieve top-k document ids for a query using cosine similarity.

    Parameters
    ----------
    model :
        SentenceTransformer-compatible model with encode_query method.
    query : str
        Input query text.
    doc_ids : Sequence[int]
        Ordered list of document ids aligned with doc_embs rows.
    doc_embs :
        Matrix of normalized document embeddings.
    k : int
        Number of top documents to retrieve.
    truncate_dim : int | None
        If provided, truncate query and document embeddings to this dimension
        for Matryoshka evaluation.

    Returns
    -------
    list[int]
        Ranked list of top-k retrieved document ids.
    """
    query_emb = model.encode_query([query], normalize_embeddings=True)

    if truncate_dim is not None:
        query_emb = query_emb[:, :truncate_dim]
        doc_embs = doc_embs[:, :truncate_dim]

    scores = (query_emb @ doc_embs.T)[0]
    ranked = sorted(zip(doc_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


def evaluate_dataset(
    model,
    dataset: Iterable[dict],
    doc_ids: Sequence[int],
    doc_embs,
    k: int,
    truncate_dim: int | None = None,
    show_progress: bool = True,
    ) -> tuple[float, float]:
    """
    Evaluate retrieval performance over a dataset using Recall@k and MRR@k.

    Parameters
    ----------
    model :
        SentenceTransformer-compatible model with encode_query method.
    dataset : Iterable[dict]
        Iterable containing evaluation records. Each record must contain:
        - "query": query text
        - "positives": list of relevant document ids
    doc_ids : Sequence[int]
        Ordered list of document ids aligned with doc_embs rows.
    doc_embs :
        Matrix of normalized document embeddings.
    k : int
        Number of retrieved documents considered for evaluation.
    truncate_dim : int | None
        If provided, truncate query and document embeddings to this dimension
        for Matryoshka evaluation.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    tuple[float, float]
        Mean Recall@k and Mean MRR@k across the dataset.
    """
    recalls = []
    reciprocal_ranks = []

    iterator = tqdm(dataset, desc="record", total=len(dataset)) if show_progress else dataset

    for record in iterator:
        query = record["query"]
        positives = set(record["positives"])

        top_k_ids = get_top_k_ids(
            model=model,
            query=query,
            doc_ids=doc_ids,
            doc_embs=doc_embs,
            k=k,
            truncate_dim=truncate_dim,
        )

        recalls.append(compute_recall_at_k(top_k_ids, positives))
        reciprocal_ranks.append(compute_mrr_at_k(top_k_ids, positives))

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return mean_recall, mean_mrr



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("-p", "--path", type=str, help="Path to eval set", required=True)
    parser.add_argument("-w", "--weights", type=str, help="The path to model weights", required=False)
    parser.add_argument("-k", "--k", type=int, help="top k for metrics computation", required=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="the model to benchmark (if no weights are provided) (Alibaba-NLP/gte-modernbert-base or joe32140/ModernBERT-base-msmarco)",
        required=False
        )
    parser.add_argument("-mt", "--matry", action="store_true", help="compute matryoshka benchmarking or not")
    
    args = parser.parse_args()
    
    # basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # load horizon projects
    projects = load_projects("data", mono=True)
    
    # load the evaluation set
    dataset = load_eval_set(path_set=args.path)
    
    # check what we want to benchmark
    if not args.weights and args.model:
        logging.info(f"Benchmarking model {args.model}")
        model = SentenceTransformer(args.model)
        name_model = args.model.split('/')[-1]
        model.save(f"models/{name_model}")
        
    elif args.weights and not args.model:
        logging.info(f"Benchmarking our finetuned model {args.weights}")
        model = SentenceTransformer(args.weights)
    
    elif args.weights and args.model:
        logging.error("You must not set a custom model and a model from the hub.")
        raise Exception()
        
    else:
        logging.error("You need to set --weights or --model")
        raise Exception()
    
    # prepare corpus
    documents = list(projects["objective"])
    doc_ids = list(projects["id"])
    
    # encode corpus once
    doc_embs = model.encode_document(documents, normalize_embeddings=True)
    
    recalls = []
    reciprocal_ranks = []
    
    # if not mrl simply eval on a fixed dim
    if not args.matry:
        mean_recall, mrr = evaluate_dataset(
            model=model,
            dataset=dataset,
            doc_ids=doc_ids,
            doc_embs=doc_embs,
            k=args.k,
        )
    
        print(f"Recall@{args.k}: {mean_recall:.4f}")
        print(f"MRR@{args.k}: {mrr:.4f}")
      
    # else eval for all mrl dims
    else:
        matryoshka_dims=[128, 384, 512, 768]
        
        recalls_dims, mrrs_dims =  [], []
        
        for d in tqdm(matryoshka_dims, desc="matryoshka", total=len(matryoshka_dims)):
            mean_recall, mrr = evaluate_dataset(
                model=model,
                dataset=dataset,
                doc_ids=doc_ids,
                doc_embs=doc_embs,
                k=args.k,
                truncate_dim=d,
                show_progress=False,
            )
    
            recalls_dims.append(mean_recall)
            mrrs_dims.append(mrr)         
        
        plot_mrl_metric(recalls_dims, f"Recall@{args.k}", matryoshka_dims)
        plot_mrl_metric(mrrs_dims, f"MRR@{args.k}", matryoshka_dims)
            
            
    