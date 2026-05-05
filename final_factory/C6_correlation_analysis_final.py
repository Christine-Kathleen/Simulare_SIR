import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, kendalltau, pearsonr

def compute_correlations(summary_df, centralities):
    """
    Corelează datele matematice cu cele stocastice pentru fiecare rețea și probabilitate.
    Versiunea FINALĂ.
    """
    results = []
    
    # Iterăm prin fiecare rețea unică din rezultatele simulării
    for network in summary_df["network"].unique():
        if network not in centralities:
            continue
            
        for p in summary_df["infection_probability"].unique():
            # Filtrăm rezultatele simulării pentru probabilitatea curentă
            subset = summary_df[(summary_df["network"] == network) & (summary_df["infection_probability"] == p)]
            
            for ctype in ["degree", "closeness", "katz"]:
                if ctype in centralities[network]:
                    classical = centralities[network][ctype]
                    
                    # Aliniem datele folosind indexul nodului (source_vertex)
                    # stochastic = media infecțiilor din simulare
                    # classical = scorul de centralitate calculat
                    merged = pd.DataFrame({
                        "stochastic": subset.set_index("source_vertex")["mean_total_infected"],
                        "classical": classical
                    }).dropna()
                    
                    # Avem nevoie de cel puțin 3 puncte pentru o corelație validă
                    if len(merged) > 2:
                        x = merged["classical"]
                        y = merged["stochastic"]
                        
                        # Verificăm dacă există variație în date (std > 0)
                        if x.std() == 0 or y.std() == 0:
                            pear = spear = kend = np.nan
                        else:
                            pear, _ = pearsonr(x, y)
                            spear, _ = spearmanr(x, y)
                            kend, _ = kendalltau(x, y)
                        
                        results.append({
                            "network": network, 
                            "p": p, 
                            "centrality": ctype,
                            "pearson": pear, 
                            "spearman": spear,
                            "kendall": kend
                        })
                        
    return pd.DataFrame(results)

def load_centralities_for_analysis(centrality_folder, network_map):
    """
    Încarcă toate fișierele de centralitate clasică din folderul specificat.
    """
    data = {}
    centrality_types = ["degree", "closeness", "katz"]

    for net_key, net_name in network_map.items():
        data[net_key] = {}
        for ctype in centrality_types:
            filename = f"{ctype}_{net_name}_output.csv"
            path = os.path.join(centrality_folder, filename)

            if os.path.exists(path):
                # Citim doar coloana de scoruri (fără index, deoarece e sortat)
                df = pd.read_csv(path, header=None, names=["score"])
                
                # Ajustăm indexarea: dacă rețeaua e naturală pornește de la 1, altfel 0
                if "natural" in net_name:
                    df.index = df.index + 1
                else:
                    df.index = df.index
                    
                data[net_key][ctype] = df["score"]
    
    return data