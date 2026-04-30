import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
        data[net_key] = {}

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

            if "generat" in net_name:
                df["node"] = df.index
            else:
                df["node"] = df.index + 1

            df = df.set_index("node")

            data[net_key][ctype] = df["score"]

    return data

def load_stochastic_summary():
    df = pd.read_csv(STOCHASTIC_FILE)

    return df[["network", "infection_probability", "source_vertex", "mean_total_infected"]]

def compute_correlations(summary_df: pd.DataFrame, centralities):
    results = []

    for network in summary_df["network"].unique():
        for p in summary_df["infection_probability"].unique():

            subset = summary_df[(summary_df["network"] == network) & (summary_df["infection_probability"] == p)]

            for ctype in CENTRALITY_TYPES:
                classical = centralities[network][ctype]

                merged = pd.DataFrame({
                    "mean_inf": subset.set_index("source_vertex")["mean_total_infected"],
                    "classical": classical
                }).dropna()

                if len(merged) < 2:
                    continue

                if merged["mean_inf"].std() == 0 or merged["classical"].std() == 0:
                    corr = np.nan
                else:
                    corr, _ = pearsonr(merged["classical"], merged["mean_inf"])

                results.append({"network": network, "p": p, "centrality": ctype, "pearson": corr})

    return pd.DataFrame(results)

def plot_pearson_vs_p(corr_df: pd.DataFrame):
    for network in corr_df["network"].unique():
        subset = corr_df[corr_df["network"] == network]

        plt.figure()

        for ctype in CENTRALITY_TYPES:
            sub = subset[subset["centrality"] == ctype]
            plt.plot(sub["p"], sub["pearson"], marker='o', label=ctype)

        plt.title(f"Pearson vs. p - {network}")
        plt.xlabel("Probabilitate p")
        plt.ylabel("Pearson")
        plt.legend()
        plt.grid()

        plt.savefig(f"plot_pearson_{network}.png")
        plt.close()

def plot_bar_comparison(corr_df, p_value=0.1):
    subset = corr_df[corr_df["p"] == p_value]

    pivot = subset.pivot(index="network", columns="centrality", values="pearson")

    pivot.plot(kind="bar")
    plt.title(f"Comparatie centralitati (p={p_value})")
    plt.ylabel("Pearson")
    plt.xticks(rotation=45)
    plt.grid()

    plt.savefig(f"comparison_p_{p_value}.png")
    plt.close()

def plot_scatter(summary_df, centralities, network, p_value, ctype):
    subset = summary_df[(summary_df["network"] == network) & (summary_df["infection_probability"] == p_value)]

    classical = centralities[network][ctype]

    merged = pd.DataFrame({
        "mean_inf": subset.set_index("source_vertex")["mean_total_infected"],
        "classical": classical
    }).dropna()

    if len(merged) < 2:
        return

    plt.figure()
    plt.scatter(merged["classical"], merged["mean_inf"])

    plt.xlabel("Centralitate clasica")
    plt.ylabel("Media infectarilor")
    plt.title(f"{network}, p={p_value}, {ctype}")

    plt.grid()
    plt.savefig(f"scatter_{network}_p{p_value}_{ctype}.png")
    plt.close()

def main():
    centralities = load_centralities()
    summary_df = load_stochastic_summary()

    corr_df = compute_correlations(summary_df, centralities)

    corr_df.to_csv("pearson_mean_correlations.csv", index=False)

    plot_pearson_vs_p(corr_df)
    plot_bar_comparison(corr_df, p_value=0.1)

    for network in corr_df["network"].unique():
        plot_scatter(summary_df, centralities, network, 0.1, "degree")

if __name__ == "__main__":
    main()
