# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:38:07 2026

@author: Jose Antonio
"""

import os
import json
import argparse

from dotenv import load_dotenv
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardData
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
import wandb

from utils import load_projects, load_eval_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-pe", "--path_eval", type=str, help="Path to eval set", required=True)
    parser.add_argument("-pt", "--path_train", type=str, help="Path to train set", required=True)
    parser.add_argument("-d", "--path_data", type=str, help="Path to raw corpus", required=True)
    parser.add_argument("-po", "--path_output", type=str, help="Path to raw corpus", required=True)
    parser.add_argument("-rn", "--run_name", type=str, help="Name of the run for W&B logging", required=True)
    parser.add_argument("-bs", "--batch_size", type=int, help="Total batch size per device", required=True)
    parser.add_argument("-cbs", "--cached_batch_size", type=int, help="Size of chunks for cached loss", required=True)
    
    args = parser.parse_args()    
    
    load_dotenv()
    
    # load full dataset
    data = load_projects(args.path_data)
    
    # load eval set to exclude all projects in eval from train
    eval_set = load_eval_set(args.path_eval)
    eval_projects = [project_id for record in eval_set for project_id in record['positives']]    
    eval_projects = list(set(eval_projects))
    data = data[~data["id"].isin(eval_projects)]
    
    # drop all with objective and keywords missing
    data = data.dropna(subset=['keywords', 'objective'])
    data.reset_index(drop=True, inplace=True)    
    
    # get train data
    train_set = load_eval_set(args.path_train)  # should unify this into a single function
    
    # convert the trainset to a good shape for MNRL training and setup HF Dataset
    train_set_v2 = []
    for record in train_set:
        for positive in record["positives"]:
            train_set_v2.append({"anchor": record["query"], "positive": data[data["id"] == positive]["objective"].values[0]})
    train_dataset = Dataset.from_list(train_set_v2)
    
    # setup the model
    model = SentenceTransformer(
        "joe32140/ModernBERT-base-msmarco",
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="A retriver version of ModernBERT specifically finetuned for project retrieval. It is build on top of joe32140/ModernBERT-base-msmarco",
        )
    )
    
    # setup loss (cached for larger batches that is beneficial in mnrl and matryoshka for multiple dims without performnace loss)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.cached_batch_size)
    loss = MatryoshkaLoss(model, loss, matryoshka_dims=[768, 512, 384, 128])
    
    # setup wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=args.run_name
    )
    
    # setup train args
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.path_output,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=10,
        save_strategy="steps",
        save_steps=0.1,
        report_to="wandb",
        run_name=args.run_name,
    )
    
    # setup trainer and train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    
    trainer.train()
        