import os
import json

import pandas as pd

def load_projects(path_csvs: str, mono: bool = False) -> pd.DataFrame:
    """Load all projects on a single DataFrame

    Args:
        path_csvs (str): The path to the data folder where the .zip file has been decompressed.
        mono (bool): If true, we only return the horizon projects data.

    Returns:
        pd.DataFrame: The merged DataFrame
    """
    horizon_proj = pd.read_csv(
        os.path.join(path_csvs, 'horizon_projects', 'project.csv'),
        encoding="utf-8",
        sep=";",
        on_bad_lines="warn",
        engine="python"
    )
    
    h2020_proj = pd.read_csv(
        os.path.join(path_csvs, 'h2020_projects', 'project.csv'),
        encoding="utf-8",
        sep=";",
        on_bad_lines="warn",
        engine="python"
    )
    
    fp7_proj = pd.read_csv(
        os.path.join(path_csvs, 'fp7_projects', 'project.csv'),
        encoding="utf-8",
        sep=";",
        on_bad_lines="skip",
        engine="python"
    )
    
    if mono:
        return horizon_proj
    
    projs = pd.concat([horizon_proj, fp7_proj, h2020_proj])
    projs.reset_index(drop=True, inplace=True)
    return projs


def load_eval_set(path_set: str) -> list:
    """
    Loads the eval set

    Parameters
    ----------
    path_set : str
        Path to the evaluation set generated.

    Returns
    -------
    list
        The eval set.

    """
    
    data = []

    with open(path_set, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            
    return data