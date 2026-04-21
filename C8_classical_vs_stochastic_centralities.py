import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

CENTRALITY_FOLDER = "scoruri_centralitati_clasice"
STOCHASTIC_FILE = "stochastic_centrality_summary.csv"

NETWORK_MAP = {
    "graf_generat_100_99": "99muchii_100noduri_generat",
    "graf_generat_275_100": "275muchii_100noduri_generat",
    "graf_generat_100_1000": "1000muchii_100noduri_generat",
    "graf_real_100_99": "99muchii_100noduri_natural",
    "graf_real_100_275": "275muchii_100noduri_natural",
    "graf_real_100_1000": "1000muchii_100noduri_natural",
}

CENTRALITY_TYPES = ["degree", "closeness", "katz"]

def load_centralities():
    data = {}

    for net_key, net_name in NETWORK_MAP.items():
        data[net_key] = {} #pt fiecare retea se considera un sub-dictionar 

        for ctype in CENTRALITY_TYPES:
            filename = f"{ctype}_{net_name}_output.csv"
            path = os.path.join(CENTRALITY_FOLDER, filename)

            if not os.path.exists(path):
                alt_name = filename.replace("100noduri", "100node")
                alt_path = os.path.join(CENTRALITY_FOLDER, alt_name)

                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    raise FileNotFoundError(f"Nu exista nici {filename}, nici {alt_name}")

            df = pd.read_csv(path, header=None, names=["score"])

            if "generat" in net_name: #pt grafurile generate sintetic, val nodurilor incep de la 0
                df["node"] = df.index 
            else:
                df["node"] = df.index + 1

            df = df.set_index("node") #nodul e setat ca index 

            data[net_key][ctype] = df["score"]

    return data

def load_stochastic_summary():
    df = pd.read_csv(STOCHASTIC_FILE)

    #se pastreaza doar col de care e nevoie din analiza statistica 
    return df[["network", "infection_probability", "source_vertex", "mean_total_infected", "max_total_infected"]]

def compute_correlations(summary_df: pd.DataFrame, centralities):
    results = []

    for network in summary_df["network"].unique():
        for p in summary_df["infection_probability"].unique():

            #se retin toate nodurile pt reteaua si prob curente 
            subset = summary_df[(summary_df["network"] == network) & (summary_df["infection_probability"] == p)]

            for ctype in CENTRALITY_TYPES:
                classical = centralities[network][ctype]

                merged = pd.DataFrame({
                    "mean_inf": subset.set_index("source_vertex")["mean_total_infected"],
                    "max_inf": subset.set_index("source_vertex")["max_total_infected"],
                    "classical": classical
                }).dropna()

                if len(merged) < 2:
                    continue

                for target in ["mean_inf", "max_inf"]:
                    x = merged["classical"]
                    y = merged[target]

                    if x.std() == 0 or y.std() == 0: #daca toate val sunt identice, corelatia nu def 
                        pearson = spearman = kendall = np.nan
                    else:
                        pearson, _ = pearsonr(x, y) #daca degree e mare => corelatie pozitiva mare 
                        spearman, _ = spearmanr(x, y)
                        kendall, _ = kendalltau(x, y)

                    results.append({
                        "network": network,
                        "p": p,
                        "centrality": ctype,
                        "target_metric": target, #"mean_inf" sau "max_inf"
                        "pearson": pearson,
                        "spearman": spearman,
                        "kendall": kendall
                    })

    return pd.DataFrame(results)

def main():
    centralities = load_centralities()
    summary_df = load_stochastic_summary()
    
    corr_df = compute_correlations(summary_df, centralities)
    
    corr_df.to_csv("correlations_all_metrics.csv", index=False)

if __name__ == "__main__":
    main()
