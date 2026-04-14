# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:26:38 2026

@author: Jose Antonio
"""

import os
import argparse
from string import Template
import json

import pandas as pd
from ollama import Client
from tqdm import tqdm
import torch
import bm25s
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import load_projects, load_eval_set


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


def infer(pairs: list[list], model, threshold: float, row: pd.Series, n_positives: int) -> pd.DataFrame:
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
    filtered_df = ranking_df[
        ranking_df["rerank_score"] >= threshold
    ].reset_index(drop=True)
    
    # we get the n - 1 from ranking_df and also append by hand the one that originated the query (to ensure its always there and it's the best positive)
    top_valid = filtered_df[filtered_df["doc_id"] != row["id"]].head(args.n_pos - 1)
    top_valid = pd.concat(
        [pd.DataFrame([{"doc_id": row["id"]}]), top_valid[["doc_id"]]],
        ignore_index=True
    )
    
    return top_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train dataset")
    parser.add_argument("-p", "--path", type=str, help="Path to eval set", required=True)
    parser.add_argument("-d", "--path_data", type=str, help="Path to raw corpus", required=True)
    parser.add_argument("-k", "--k", type=int, help="top k for metrics computation", required=True)
    parser.add_argument("-n", "--n_pos", type=int, help="number of positives to select", required=True)
    parser.add_argument("-pr", "--prop", type=float, help="The proportion of samples from the full corpus to use (eg 0.5)", required=True)
    
    args = parser.parse_args()
    
    # load full dataset
    data = load_projects(args.path_data)
    
    # load eval set to exclude all projects in eval from train
    eval_set = load_eval_set(args.path)
    eval_projects = [project_id for record in eval_set for project_id in record['positives']]    
    eval_projects = list(set(eval_projects))
    data = data[~data["id"].isin(eval_projects)]
    
    # get the sampled dataset
    data_process = data.groupby(
        "frameworkProgramme",
        group_keys=False
        ).apply(lambda x: x.sample(frac=args.prop, random_state=42))
    
    # drop all with objective and keywords missing
    data_process = data_process.dropna(subset=['keywords', 'objective'])
    data_process.reset_index(drop=True, inplace=True)
    
    
    """same idea as in eval, generate anchors (queries) via local llm
    and extract positives with bm25 + ranker """
    
    # prepare reranker and tokenizer for using it later in the loop
    device = setup_device()
    tokenizer, model = setup_reranker(model_name="BAAI/bge-reranker-v2-m3", device=device)
    
    corpus_tokens = bm25s.tokenize(data_process["objective"].values, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    
    """ for the training, we will use three types of queries to enhance robustness:
        1. short human queries you would type into a search engine. They lack complex
        semantics, just short sentences with a bunch of keywords (eg. AI for autonomous robots) 
        
        2. medium human queries. They are more elaborated than the first one
        but have essentially the same nature
        (eg. AI methods to detect phishing emails in small businesses)
        
        3. Problem-oriented query. These are the hardest for the models, but are really
        useful for people that searchs straight how to solve a problem
        (eg. how to prevent phishing attacks in company email systems)"""
    
    # prepare prompts
    short_template = Template(""" You are generating a search query for an information retrieval benchmark.

                Given the following project summary, produce ONE realistic search query that a user might type into a search engine to find projects like this.

                Requirements:
                - It must sound like something a human would actually search for.
                - Maximum 6 words.
                - Avoid prepositions unless necessary.
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

                Project summary: $objective""")
                
    medium_template = Template(""" You are generating a search query for an information retrieval benchmark.
                        Given the following project summary, produce ONE realistic medium-length
                        natural search query that a user might type into a search engine to find projects like this.
                    
                        Requirements:
                        - It must sound natural and realistic, like a human search query.
                        - It should reflect a clear search intent.
                        - Capture the main objective, technology, or application of the project.
                        - Use natural search phrasing rather than title-like or academic wording.
                        - Keep it moderately detailed: target length 8 to 20 words.
                        - Prefer specific and discriminative wording over vague generic expressions.
                        - Do NOT write a complete grammatical sentence.
                        - Do NOT copy long phrases verbatim from the summary.
                        - Do NOT mention "project", "research", "study", or funding-related terms.
                        - Do NOT invent information.
                        - Return ONLY the query text.
                        
                        Good query style examples:
                        - AI methods to detect phishing emails in small businesses
                        - technologies for monitoring marine biodiversity with underwater sensors
                        - blockchain solutions for improving food supply chain traceability
                        - machine learning tools for early cancer image diagnosis
                        
                        Project summary: $objective
                        """)
    
    problem_template = Template(""" You are generating a search query for an information retrieval benchmark.
    
        Given the following project summary, produce ONE realistic problem-oriented search query that a user might type when trying to solve the problem addressed by this project.
        
        Requirements:
        - The query must express a real user need, challenge, or problem to solve.
        - Focus on the problem the project addresses, not just the technology used.
        - It should sound like something a person would type when looking for solutions.
        - Use natural human search wording.
        - Target length: 8 to 18 words.
        - Prefer concrete problem descriptions over vague general wording.
        - Do NOT write a full sentence with punctuation at the end.
        - Do NOT copy long phrases verbatim from the summary.
        - Do NOT mention "project", "research", "study", or funding-related terms.
        - Do NOT invent information.
        - Return ONLY the query text.
        
        Good query style examples:
        - how to prevent phishing attacks in company email systems
        - how to improve food traceability in agricultural supply chains
        - ways to monitor endangered marine species underwater
        - how to reduce battery waste from electric vehicles
        
        Project summary: $objective
        """)
    
    client = Client(host='http://192.168.2.12:11434')
    
    train_set = []
    for i, row in tqdm(data_process.iterrows(), desc="train construction"):
        
        short_prompt = short_template.substitute(objective=row["objective"])
        medium_prompt = medium_template.substitute(objective=row["objective"])
        problem_prompt = problem_template.substitute(objective=row["objective"])
        
        # generate query for the project
        short_resp = client.generate(
            model='gpt-oss:20b',
            prompt=short_prompt,
            stream=False
        )["response"]
        
        medium_resp = client.generate(
            model='gpt-oss:20b',
            prompt=medium_prompt,
            stream=False
        )["response"]
        
        problem_resp = client.generate(
            model='gpt-oss:20b',
            prompt=problem_prompt,
            stream=False
        )["response"]
        
        
        print(f"Project: {row['objective']}\n")
        print(f"Short query: {short_resp}")
        print(f"Medium query: {medium_resp}")
        print(f"Problem query: {problem_resp}\n")
        
        # get keywords from our topic and tokenize via BM25 tokenizer
        keywords_query_project = bm25s.tokenize(row["keywords"])  # we could also tokenize the query, but keywords will typically be more complete
        
        # grab docs and indexes selected by bm25
        results, scores = retriever.retrieve(keywords_query_project, k=args.k)        
        top_k_idx = results[0, :]
        docs_bm25 = data_process["objective"].iloc[top_k_idx]
        idxs_bm25 = data_process["id"].iloc[top_k_idx]
            
        # use the reranker to enhance the results
        pairs_short = [[short_resp, doc] for doc in docs_bm25]
        pairs_medium = [[medium_resp, doc] for doc in docs_bm25]
        pairs_problem = [[problem_resp, doc] for doc in docs_bm25]
        
        top_valid_short = infer(pairs=pairs_short, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        top_valid_medium = infer(pairs=pairs_medium, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        top_valid_problem = infer(pairs=pairs_problem, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        
        # create train records
        record_short = {"query": short_resp, "positives": top_valid_short["doc_id"].values.tolist()}
        record_medium = {"query": medium_resp, "positives": top_valid_medium["doc_id"].values.tolist()}
        record_problem = {"query": problem_resp, "positives": top_valid_problem["doc_id"].values.tolist()}
    
        print(f"\nShort ones: {record_short}\n")
        print(f"Medium ones: {record_medium}\n")
        print(f"Problem ones: {record_problem}\n")
        
        # append the records
        train_set.append(record_short)
        train_set.append(record_medium)
        train_set.append(record_problem)
        

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")    
        
    
    