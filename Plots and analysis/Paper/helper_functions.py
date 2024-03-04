# This file contains some helper functions that will be used 
# in the jupyter notebook in the same directory. These functions are 
# collected from the jupyter notebooks that were used to create plots for
# my MSc thesis (in ../MSc thesis), with suitable modifications. 

import pandas as pd
import numpy as np
import copy, random
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

def add_asa_result_to_df(df, output_path, drop_class=[]):
    """
    params:
        df: pdd.Dataframe, the whole df that contains all instances (of all folds)
        output_path: str, path to file that contains the prediction results
    return: 
        df of this fold only, containing the predictions made by the best model
    """
    fold = int(output_path.split("asa_output_")[-1][0])
    fold_df = df[df.split == fold]
    fold_df = fold_df[~fold_df.cefr_mean.isin(drop_class)]
    
    # get results from output path 
    labels = []
    predictions = []

    with open(output_path, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line: 
                break 
            if "label" in line:
                labels.append(int(line.strip()[-1]) + 1)
            elif "pred" in line:
                predictions.append(int(line.strip()[-1]) + 1)

    assert len(fold_df) == len(labels), f"{len(fold_df)}, {len(labels)}" 
    assert all(fold_df.cefr_mean == np.array(labels))
    
    fold_df.insert(4, "Prediction", predictions, True)
    return fold_df

def get_aggregated_asa_df(df, paths, drop_class):
    """
    params: 
        df: the df 
    """
    dfs = []
    for path in paths:
        fold_df = add_asa_result_to_df(df, path, drop_class)
        dfs.append(fold_df)    
    return pd.concat(dfs)

def get_asa_metrics(y_true, y_pred, average="macro"):
    """
    params: 
        y_true: (N,) 1-d array, labels 
        y_pred: (N,) 1-d array, predictions
        average: str, use macro average by default, as all classes are equally important
    returns: 
        (float, float, float, float): the evaluation metrics based on y_true and y_pred
    """

    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
    kappa = cohen_kappa_score(y1=y_true, y2=y_pred, weights="quadratic")
    
    return precision,recall, f1, kappa

def resample(train_dataset,
             alpha=2, # 2 means doubling the dataset
             criterion: str = "rating"):
    """Resample data to balanced out the data based on the chosen rating (default = cefr_mean)"""
    
    train_copy = copy.deepcopy(train_dataset) # always create a copy 
    ratings = train_copy[criterion].tolist()

    # calculate samlping rate
    group_counts = Counter(ratings)
    n_group = len(group_counts)
    n_copy = alpha 
    avg_n_samples_per_gp = len(train_copy)*n_copy/n_group
    n_samples = [(group, avg_n_samples_per_gp - count) for group, count in group_counts.items()]
    assert all([n > 0 for _, n in n_samples]), f"Please increase alpha."
    weights = {group: round(100*n/group_counts[group]) for group, n in n_samples}
    resample_weights = [weights[r] for r in ratings]

    # resample data based on weights
    n = random.choices(range(len(train_copy)), weights=resample_weights, k=int(len(train_copy)*(alpha-1)))
    train_copy = train_copy.iloc[n]

    return train_copy