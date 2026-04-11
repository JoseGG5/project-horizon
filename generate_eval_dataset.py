import os
import random
import json
from pprint import pprint

import pandas as pd
from tqdm import tqdm
from ollama import Client
import bm25s
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from utils import load_projects

K = 20
n_positives_select = 3

def setup_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_reranker(model_name: str, device: str):
    """ Sets up the tokenizer and reranker """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    return tokenizer, model


if __name__ == "__main__":
    
    # order among chaos
    random.seed(42)

    # for eval only horizon projects
    projects = load_projects("data", mono=True)

    # drop all with keywords missing
    projects = projects.dropna(subset=['keywords'])
    projects.reset_index(drop=True, inplace=True)

    # now we want to select 100 different projects and generate a query per project
    # we stratify this with the call to ensure that we capture different problems and types of projects
    n_projects = 100
    samples = []
    for topic, group in projects.groupby("topics"):
        sampled_group = group.sample(frac=0.1, random_state=42)
        samples.append(sampled_group)
    
    stratified_sample = pd.concat(samples, ignore_index=True)
    
    stratified_sample = stratified_sample.sample(
        n=min(n_projects, len(stratified_sample)),
        random_state=42
    ).reset_index(drop=True)

    # now we need to find all relevant projects to a given query (we do this via bm25 + reranking)
    # originally I used topic heuristics also but in my experience similar topics can appear in differet calls and being in the same call sometimes means nothing
    eval_set = []
    
    # prepare reranker and tokenizer for using it later in the loop
    device = setup_device()
    tokenizer, model = setup_reranker(model_name="BAAI/bge-reranker-v2-m3", device=device)

    # prepare a full bm25
    corpus_tokens = bm25s.tokenize(projects["objective"], stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # set ollama client for query generation
    client = Client(host='http://192.168.2.12:11434')
    
    # start iterating
    for i, row in tqdm(stratified_sample.iterrows(), desc="eval construction"):
        
        # generate query for the project
        resp = client.generate(
            model='gpt-oss:20b',
            prompt=f"""You are generating a search query for an information retrieval benchmark.

                        Given the following project summary, produce ONE realistic search query that a user might type into a search engine to find projects like this.

                        Requirements:
                        - It must sound like something a human would actually search for.
                        - It should capture the main goal, technology, or application of the project.
                        - Prefer natural search wording over paper-title wording.
                        - Do NOT write a full sentence.
                        - Do NOT copy long phrases verbatim from the summary.
                        - Do NOT mention "project", "research", "study", or funding-related terms.
                        - Do NOT invent information.
                        - Return ONLY the query text.

                        Good query style examples:
                        - AI for autonomous robots
                        - battery recycling for electric vehicles
                        - digital tools for rural communities
                        - gender equality in agriculture
                        - small RNA structure prediction methods

                        Project summary: {row["objective"]}
                        """,
            stream=False
        )
        query = resp["response"]

        print(f"query: {query}")

        # use all horizon corpus and a full bm25
        projects_related = projects  # avoid overwriting projects
        
        # # exclude the one that originated the query (we will append it by hand to ensure it appears) and reset index to avoid misconfig
        # projects_related = projects_related[projects_related["id"] != row["id"]].reset_index(drop=True)
        
        # get keywords from our topic and tokenize via BM25 tokenizer
        keywords_query_project = bm25s.tokenize(row["keywords"])  # we could also tokenize the query, but keywords will typically be more complete
        
        # grab docs and indexes selected by bm25
        results, scores = retriever.retrieve(keywords_query_project, k=K)        
        top_k_idx = results[0, :]
        docs_bm25 = projects_related["objective"].iloc[top_k_idx]
        idxs_bm25 = projects_related["id"].iloc[top_k_idx]
            
        # use the reranker to enhance the results
        pairs = [[query, doc] for doc in docs_bm25]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            rerank_scores = model(**inputs).logits.squeeze(-1)
        rerank_scores = rerank_scores.detach().cpu().tolist()
        
        ranking_df = pd.DataFrame({
            "doc_id": idxs_bm25,
            "doc_text": docs_bm25,
            "rerank_score": rerank_scores
        }).sort_values("rerank_score", ascending=False).reset_index(drop=True)
        
        
        # to select the positives we filter by a quality threshold and then select the k first
        threshold = 0.5  # manually adjusted by looking at the distribution
        filtered_df = ranking_df[
            ranking_df["rerank_score"] >= threshold
        ].reset_index(drop=True)
        
        # we get the n - 1 from ranking_df and also append by hand the one that originated the query (to ensure its always there and it's the best positive)
        top_valid = filtered_df[filtered_df["doc_id"] != row["id"]].head(n_positives_select - 1)
        top_valid = pd.concat(
            [pd.DataFrame([{"doc_id": row["id"]}]), top_valid[["doc_id"]]],
            ignore_index=True
        )
        
        # create benchmark record
        record = {"query": query, "positives": top_valid["doc_id"].values.tolist()}
        eval_set.append(record)
        
    


    with open("eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")