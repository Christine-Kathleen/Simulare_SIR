import os
import csv
import time
import yaml
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

# Importăm motorul de simulare din fișierul vecin
import csv_generator 

# ═════════════════════════════════════════════════════════════════════════════
# 1. ÎNCĂRCARE CONFIGURAȚIE ȘI DETECTARE HARDWARE
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAFURI_DIR = os.path.join(BASE_DIR, "grafuri_naturale_si_generate_curatate")

cale_config = os.path.join(BASE_DIR, "config.yaml")

if not os.path.exists(cale_config):
    raise FileNotFoundError(f"Eroare: Fisierul config.yaml nu a fost gasit in {BASE_DIR}")

with open(cale_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# --- Logica de detectare nuclee (Hardware Autodiscovery) ---
n_cores_sistem = os.cpu_count() or 1
n_jobs_config = config["optiuni"].get("n_jobs", 8)

if n_jobs_config == -1:
    N_JOBS = n_cores_sistem
else:
    N_JOBS = min(n_jobs_config, n_cores_sistem)

# Parametri din YAML
VALORI_P = config["valori_p"]
RULARI_PE_SETARE = config["rulari_pe_setare"]
RETELE = config["retele"]
TICKS_MAX_CONFIG = config["optiuni"].get("ticks_maxim", 200)

# Output LOCAL
OUTPUT_BASE = os.path.join(BASE_DIR, config["output"]["director_rezultate"])
CSV_BRUT_PATH = os.path.join(OUTPUT_BASE, config["output"]["csv_brut"])

# ═════════════════════════════════════════════════════════════════════════════
# 2. COORDONARE STUDIU
# ═════════════════════════════════════════════════════════════════════════════

def run_study(graf, nume_retea, probabilitati, nr_rulari, tip_graf):
    noduri_s = sorted(graf.nodes())
    noduri_f = frozenset(noduri_s)
    vecini = {nd: frozenset(graf.neighbors(nd)) for nd in noduri_s}
    g_max = max(d for _, d in graf.degree())

    os.makedirs(os.path.dirname(CSV_BRUT_PATH), exist_ok=True)
    
    if not os.path.exists(CSV_BRUT_PATH):
        with open(CSV_BRUT_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["retea", "sursa", "p", "run_id", "ticks", "nr_infectati"])

    for p in probabilitati:
        print(f"\n>>> Simulăm p={p} | Rețea: {nume_retea}")
        prob_dir = os.path.join(OUTPUT_BASE, nume_retea, f"Prob_{p}")
        os.makedirs(prob_dir, exist_ok=True)
        
        prob_t = csv_generator.tabel_prob(p, g_max)

        for run_id in range(1, nr_rulari + 1):
            t0 = time.perf_counter()
            rez_batch = {}
            
            # Paralelizare folosind nucleele detectate
            with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                futures = {
                    executor.submit(
                        csv_generator.ruleaza_simulare_core, 
                        vecini, noduri_f, s, prob_t, TICKS_MAX_CONFIG
                    ): s for s in noduri_s
                }
                
                with open(CSV_BRUT_PATH, "a", newline="", encoding="utf-8") as f_brut:
                    w_brut = csv.writer(f_brut)
                    for future in as_completed(futures):
                        res = future.result()
                        rez_batch[res["sursa"]] = res
                        w_brut.writerow([nume_retea, res["sursa"], p, run_id, res["ticks"], len(res["infection_tick"])])
                    f_brut.flush()

            csv_generator.export_nomenclator_csv(graf, rez_batch, p, run_id, tip_graf, prob_dir)
            print(f"   -> Run {run_id}/{nr_rulari} gata ({time.perf_counter()-t0:.2f}s)")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PUNCT DE INTRARE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"════════════════════════════════════════════════════════")
    print(f" START PIPELINE MODULAR (SIR)")
    print(f" Nuclee utilizate: {N_JOBS} (Total sistem: {n_cores_sistem})")
    print(f" Destinație: {OUTPUT_BASE}")
    print(f"════════════════════════════════════════════════════════")

    for r_info in RETELE:
        nume_fisier = os.path.basename(r_info["cale"])
        cale_fisier = os.path.join(GRAFURI_DIR, nume_fisier)
        
        if os.path.exists(cale_fisier):
            print(f"\n[OK] Încărcare rețea: {r_info['nume']}")
            graf = nx.Graph()
            with open(cale_fisier, "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    if len(r) >= 2:
                        try:
                            graf.add_edge(int(r[0]), int(r[1]))
                        except: continue
            
            run_study(graf, r_info["nume"], VALORI_P, RULARI_PE_SETARE, "experiment")
        else:
            print(f"\n[EROARE] Fișierul nu a fost găsit în {GRAFURI_DIR}: {nume_fisier}")

if __name__ == "__main__":
    main()