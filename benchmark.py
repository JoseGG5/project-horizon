import argparse

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from utils import load_projects, load_eval_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("-p", "--path", type=str, help="Path to eval set", required=True)
    parser.add_argument("-w", "--weights", type=str, help="The path to model weights", required=False)
    parser.add_argument("-k", "--k", type=int, help="top k for metrics computation", required=True)
    
    args = parser.parse_args()
    
    # load horizon projects
    projects = load_projects("data", mono=True)
    
    # load the evaluation set
    dataset = load_eval_set(path_set=args.path)
    
    # if not fine tuned we benchmark the gte base which is suited for retrieval
    if not args.weights:
        # model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
        model = SentenceTransformer("joe32140/ModernBERT-base-msmarco")
    else:
        model = SentenceTransformer(args.weights)
    
    documents = list(projects["objective"])
    doc_ids = list(projects["id"])
    
    # encode corpus once
    doc_embs = model.encode_document(documents, normalize_embeddings=True)
    
    recalls = []
    reciprocal_ranks = []
    
    for record in tqdm(dataset, desc="record", total=len(dataset)):
        query = record["query"]
        positives = set(record["positives"])
    
        query_emb = model.encode_query([query], normalize_embeddings=True)
    
        scores = (query_emb @ doc_embs.T)[0]
        ranked = sorted(zip(doc_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        top_k_ids = [doc_id for doc_id, _ in ranked[:args.k]]
    
        # Recall@k
        retrieved_relevant = len(set(top_k_ids) & positives)
        recall = retrieved_relevant / len(positives) if positives else 0.0
        recalls.append(recall)
    
        # MRR@k
        rr = 0.0
        for rank, doc_id in enumerate(top_k_ids, start=1):
            if doc_id in positives:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    
    mean_recall = sum(recalls) / len(recalls)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    print(f"Recall@{args.k}: {mean_recall:.4f}")
    print(f"MRR@{args.k}: {mrr:.4f}")