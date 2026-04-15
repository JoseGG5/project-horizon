# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:26:38 2026

@author: Jose Antonio
"""

import os
import argparse
from string import Template
import json
import logging

import pandas as pd
from ollama import Client
from tqdm import tqdm
import torch
import bm25s
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
import tiktoken

from utils import load_projects, load_eval_set


class QueryResponse(BaseModel):
  short_query: str
  medium_query: str
  problem_query: str


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
    
    # set logging
    logging.basicConfig(
        filename="trainset_pipe.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8"
    )
    
    
    # load full dataset
    data = load_projects(args.path_data)
    
    # load eval set to exclude all projects in eval from train
    eval_set = load_eval_set(args.path)
    eval_projects = [project_id for record in eval_set for project_id in record['positives']]    
    eval_projects = list(set(eval_projects))
    data = data[~data["id"].isin(eval_projects)]
    
    # drop all with objective and keywords missing
    data = data.dropna(subset=['keywords', 'objective'])
    data.reset_index(drop=True, inplace=True)
    
    # get the sampled dataset
    data_process = data.groupby(
        "frameworkProgramme",
        group_keys=False
        ).apply(lambda x: x.sample(frac=args.prop, random_state=42))
    
    print(data_process.iloc[91])
    
     
    print(f"There are {len(data_process)} projects to be processed")
    
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
    
    # prepare the mega prompt :)
    combined_template = Template("""
        Generate exactly 3 realistic search engine queries from the project summary.
        
        These queries are for an information retrieval benchmark.
        They must resemble real Google searches written by humans.
        
        Return ONLY one valid JSON object exactly in this format:
        {
          "short_query": "...",
          "medium_query": "...",
          "problem_query": "..."
        }
        
        STRICT GLOBAL RULES:
        - Output must be valid JSON only.
        - Do NOT add explanations, markdown, comments, or extra text.
        - Queries must sound like real human search engine searches.
        - NEVER generate chatbot-style prompts.
        - NEVER generate academic question sentences.
        - NEVER generate full grammatical explanatory sentences.
        - NEVER start any query with:
          How, What, Why, When, Where, Who, Can, Could, Should, Does, Do
        - If a query sounds like a question, rewrite it into compact search format.
        - Prefer noun phrases and keyword search style.
        - Keep wording concise and retrieval-oriented.
        - Do NOT invent facts not present in the summary.
        - Do NOT mention these words:
          project, research, study, funding, proposal
        - Do NOT copy long phrases verbatim from the summary.
        - The three queries must be clearly different in wording and intent.
        - No punctuation at the end.
        - Avoid paper-title style phrasing.
        - Avoid secondary contextual details like geographic regions unless central to the scientific problem.
        
        FIELD RULES:
        
        short_query:
        - Maximum 6 words
        - Compact keyword-style query
        - Minimal and highly concise
        
        medium_query:
        - 8 to 14 words
        - Natural descriptive search query
        - Must express clear search intent
        - Must NOT be phrased as a question
        - Avoid title-like academic phrasing; prefer natural search wording.
        
        problem_query:
        - 8 to 14 words
        - Express a practical problem someone is trying to solve
        - Must be compact keyword search style
        - Must NOT be phrased as a question
        - Prefer formats like:
          "identify high risk melanoma patients after UV exposure"
        
        STYLE EXAMPLES:
        
        GOOD short_query:
        UV melanoma risk genetics
        
        GOOD medium_query:
        genetic susceptibility factors in melanoma caused by ultraviolet radiation exposure
        
        GOOD problem_query:
        identify high risk melanoma patients after UV exposure using genetic markers
        
        BAD examples:
        How does UV radiation increase melanoma risk
        What strategies can identify melanoma patients
        Why is UV exposure linked to melanoma
        
        Project summary:
        $objective
        """)
          
    # setup client
    client = Client(host='http://192.168.2.12:11434')
    
    # additionally get the tokenizer used in gpt-oss to check that we are in feasible context length
    encoding = tiktoken.get_encoding("o200k_harmony")
    
    for i, row in tqdm(data_process.iterrows(), desc="train construction", total=len(data_process)):
      
        combined_prompt = combined_template.substitute(objective=row["objective"])
        
        tokens = encoding.encode(combined_prompt)
        if len(tokens) > 4096:
            logging.info(
                f"Exceeded context window | project_id={row['id']} | programme={row['frameworkProgramme']}"
            )
            continue
        
        # generate query for the project
        combined_resp = client.chat(
            model='gpt-oss:20b',
            messages=[{"role": "user", "content": combined_prompt}],
            stream=False,
            format=QueryResponse.model_json_schema(),
            options={
                "temperature": 0,
                "num_ctx": 4096  # we should be around 1k tokens more or less
            }
        )
        
        # parse the json
        raw = combined_resp.message.content
        try:
            parsed = QueryResponse.model_validate_json(raw)
        except Exception as e:
            logging.error(
                f"JSON parse failed | project_id={row['id']} | programme={row['frameworkProgramme']} | error={str(e)} | raw_output={raw}"
            )
            continue
        
        # get keywords from our topic and tokenize via BM25 tokenizer
        keywords_query_project = bm25s.tokenize(row["keywords"])  # we could also tokenize the query, but keywords will typically be more complete
        
        # grab docs and indexes selected by bm25
        results, scores = retriever.retrieve(keywords_query_project, k=args.k)        
        top_k_idx = results[0, :]
        docs_bm25 = data_process["objective"].iloc[top_k_idx]
        idxs_bm25 = data_process["id"].iloc[top_k_idx]
            
        # use the reranker to enhance the results
        pairs_short = [[parsed.short_query, doc] for doc in docs_bm25]
        pairs_medium = [[parsed.medium_query, doc] for doc in docs_bm25]
        pairs_problem = [[parsed.problem_query, doc] for doc in docs_bm25]
        
        top_valid_short = infer(pairs=pairs_short, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        top_valid_medium = infer(pairs=pairs_medium, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        top_valid_problem = infer(pairs=pairs_problem, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        
        # create train records
        record_short = {"query": parsed.short_query, "positives": top_valid_short["doc_id"].values.tolist()}
        record_medium = {"query": parsed.medium_query, "positives": top_valid_medium["doc_id"].values.tolist()}
        record_problem = {"query": parsed.problem_query, "positives": top_valid_problem["doc_id"].values.tolist()}
        
        # write the three to a train jsonl file
        with open("train.jsonl", "a", encoding="utf-8") as f:            
            f.write(json.dumps(record_short, ensure_ascii=False) + "\n")
            f.write(json.dumps(record_medium, ensure_ascii=False) + "\n")
            f.write(json.dumps(record_problem, ensure_ascii=False) + "\n")
            
            
    
    