# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:38:07 2026

@author: Jose Antonio
"""

import json
import argparse


from utils import load_projects, load_eval_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-pe", "--patheval", type=str, help="Path to eval set", required=True)
    parser.add_argument("-pt", "--pathtrain", type=str, help="Path to train set", required=True)
    parser.add_argument("-d", "--path_data", type=str, help="Path to raw corpus", required=True)
    
    args = parser.parse_args()    
    
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
    
    # get train data
    train_set = load_eval_set(args.pt)  # should unify this into a single function