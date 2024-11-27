import os

import networkx as nx
import pandas as pd

def get_graph() -> nx.Graph:
    data_path = os.path.join(os.path.dirname(__file__), "facebook_combined.txt/facebook_combined.txt")
    df = pd.read_csv(data_path, sep=" ", header=None)
    
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    return G