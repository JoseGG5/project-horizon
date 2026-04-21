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
import random

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import torch
import bm25s
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
import tiktoken
from dotenv import load_dotenv

from utils import load_projects, load_set


class QueryResponse(BaseModel):
    query: str


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
    
    load_dotenv()

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
    eval_set = load_set(args.path)
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
    
    # prepare the three mega prompts :)
    
    short_template = Template("""
        Generate exactly 1 realistic search engine query from the project summary.
        
        This query is for an information retrieval benchmark.
        It must resemble a real Google search written by a human.
        
        Return ONLY one valid JSON object exactly in this format:
        {
          "query": "..."
        }
        
        STRICT GLOBAL RULES:
        - Output must be valid JSON only.
        - Do NOT add explanations, markdown, comments, or extra text.
        - Query must sound like a real human search engine search.
        - NEVER generate chatbot-style prompts.
        - NEVER generate academic question sentences.
        - NEVER generate full grammatical explanatory sentences.
        - NEVER start the query with:
          How, What, Why, When, Where, Who, Can, Could, Should, Does, Do
        - If the query sounds like a question, rewrite it into compact search format.
        - Prefer noun phrases and keyword search style.
        - Keep wording concise and retrieval-oriented.
        - Do NOT invent facts not present in the summary.
        - Do NOT mention these words:
          project, research, study, funding, proposal
        - Do NOT copy long phrases verbatim from the summary.
        - No punctuation at the end.
        - Avoid paper-title style phrasing.
        - Avoid secondary contextual details like geographic regions unless central to the scientific problem.
        
        FIELD RULES:
        
        short_query:
        - Maximum 6 words
        - Compact keyword-style query
        - Minimal and highly concise
        
        STYLE EXAMPLES:
        
        GOOD:
        UV melanoma risk genetics
        
        BAD:
        How does UV radiation increase melanoma risk
        What strategies can identify melanoma patients
        
        Project summary:
        $objective
        """)
    
    medium_template = Template("""
        Generate exactly 1 realistic search engine query from the project summary.
        
        This query is for an information retrieval benchmark.
        It must resemble a real Google search written by a human.
        
        Return ONLY one valid JSON object exactly in this format:
        {
          "query": "..."
        }
        
        STRICT GLOBAL RULES:
        - Output must be valid JSON only.
        - Do NOT add explanations, markdown, comments, or extra text.
        - Query must sound like a real human search engine search.
        - NEVER generate chatbot-style prompts.
        - NEVER generate academic question sentences.
        - NEVER generate full grammatical explanatory sentences.
        - NEVER start the query with:
          How, What, Why, When, Where, Who, Can, Could, Should, Does, Do
        - If the query sounds like a question, rewrite it into compact search format.
        - Prefer natural search wording.
        - Keep wording concise and retrieval-oriented.
        - Do NOT invent facts not present in the summary.
        - Do NOT mention these words:
          project, research, study, funding, proposal
        - Do NOT copy long phrases verbatim from the summary.
        - No punctuation at the end.
        - Avoid paper-title style phrasing.
        - Avoid secondary contextual details like geographic regions unless central to the scientific problem.
        
        FIELD RULES:
        
        medium_query:
        - 8 to 14 words
        - Natural descriptive search query
        - Must express clear search intent
        - Must NOT be phrased as a question
        - Avoid title-like academic phrasing
        
        STYLE EXAMPLES:
        
        GOOD:
        genetic susceptibility factors in melanoma caused by ultraviolet radiation exposure
        
        BAD:
        What causes melanoma after UV radiation exposure
        How can genetics predict melanoma risk
        
        Project summary:
        $objective
        """)
        
    problem_template = Template("""
        Generate exactly 1 realistic search engine query from the project summary.
        
        This query is for an information retrieval benchmark.
        It must resemble a real Google search written by a human.
        
        Return ONLY one valid JSON object exactly in this format:
        {
          "query": "..."
        }
        
        STRICT GLOBAL RULES:
        - Output must be valid JSON only.
        - Do NOT add explanations, markdown, comments, or extra text.
        - Query must sound like a real human search engine search.
        - NEVER generate chatbot-style prompts.
        - NEVER generate academic question sentences.
        - NEVER generate full grammatical explanatory sentences.
        - NEVER start the query with:
          How, What, Why, When, Where, Who, Can, Could, Should, Does, Do
        - If the query sounds like a question, rewrite it into compact search format.
        - Prefer compact practical search wording.
        - Keep wording concise and retrieval-oriented.
        - Do NOT invent facts not present in the summary.
        - Do NOT mention these words:
          project, research, study, funding, proposal
        - Do NOT copy long phrases verbatim from the summary.
        - No punctuation at the end.
        - Avoid paper-title style phrasing.
        - Avoid secondary contextual details like geographic regions unless central to the scientific problem.
        
        FIELD RULES:
        
        problem_query:
        - 8 to 14 words
        - Express a practical problem someone is trying to solve
        - Must be compact keyword search style
        - Must NOT be phrased as a question
        - Prefer formats like:
          identify high risk melanoma patients after UV exposure
        
        STYLE EXAMPLES:
        
        GOOD:
        identify high risk melanoma patients after UV exposure using genetic markers
        
        BAD:
        How to identify melanoma patients after UV exposure
        What methods detect melanoma risk
        
        Project summary:
        $objective
        """)
    
    # we will randomly select one per project
    templates = [short_template, medium_template, problem_template]
          
    # setup vLLM client
    client = OpenAI(
        base_url=os.getenv("VLLM_ADDRESS"),   
        api_key="dummy"  # unauth server
    )
    
    # additionally get the tokenizer used in gpt-oss to check that we are in feasible context length
    encoding = tiktoken.get_encoding("o200k_harmony")
    
    for i, row in tqdm(data_process.iterrows(), desc="train construction", total=len(data_process)):
        
        # randomly select a template and interpolate
        template = random.choice(templates)
        prompt = template.substitute(objective=row["objective"])
        
        tokens = encoding.encode(prompt)
        if len(tokens) > 4096:
            logging.info(
                f"Exceeded context window | project_id={row['id']} | programme={row['frameworkProgramme']}"
            )
            continue
        
        # generate query for the project
        combined_resp = client.chat.completions.create(
            model='openai/gpt-oss-20b',
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_response",
                    "schema": QueryResponse.model_json_schema(),
                },
            },
        )
        
        # parse the json
        raw = combined_resp.choices[0].message.content
        try:
            parsed = QueryResponse.model_validate_json(raw)
        except Exception as e:
            logging.error(
                f"JSON parse failed | project_id={row['id']} | error={str(e)} | raw_output={raw}"
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
        pairs = [[parsed.query, doc] for doc in docs_bm25] 
        top_valid = infer(pairs=pairs, model=model, threshold=0.5, row=row, n_positives=args.n_pos)
        
        # create train records
        record = {"query": parsed.query, "positives": top_valid["doc_id"].values.tolist()}
        
        # write the three to a train jsonl file
        with open("train.jsonl", "a", encoding="utf-8") as f:            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            
    
    