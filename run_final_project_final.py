import os
import sys
import pandas as pd
import networkx as nx

# Importuri din pachetul final_factory
from final_factory.C6_graph_utils_final import load_graph_from_csv
from final_factory.C6_centrality_master_final import CentralityMaster
from final_factory.C6_simulation_engine_final import SIRSimulationEngine
from final_factory.C6_correlation_analysis_final import compute_correlations, load_centralities_for_analysis
from final_factory.C6_visualizer_final import SIRVisualizer
from final_factory.C0_graph_generator_final import genereaza_toate_grafurile
from final_factory.C2_preprocessor_final import GraphPreprocessor

# --- CONFIGURARE EXPERIMENT ---
INPUT_DIR = "grafuri_naturale_si_generate_curatate"
CENTRALITY_DIR = "scoruri_centralitati_clasice"
SIM_SUMMARY_FILE = "stochastic_centrality_summary.csv"
REPORT_DIR = "raport_final_export"

# Parametri Simulări
P_VALUES = [0.01, 0.04, 0.1, 0.3, 0.5]
NR_RULARI = 100

# Maparea rețelelor pentru corelație
NETWORK_MAP = {
    "graf_generat_100_99": "99muchii_100noduri_generat",
    "graf_generat_275_100": "275muchii_100noduri_generat",
    "graf_generat_100_1000": "1000muchii_100noduri_generat",
    "graf_real_100_99": "99muchii_100noduri_natural",
    "graf_real_100_275": "275muchii_100noduri_natural",
    "graf_real_100_1000": "1000muchii_100noduri_natural"
}

# ============================================================================
# FUNCȚII UTILITARE
# ============================================================================

def asigura_folder(folder):
    """Creează folderul dacă nu există."""
    os.makedirs(folder, exist_ok=True)


def preproceseaza_graf(G):
    """
    Aplică preprocesarea completă asupra unui graf:
    - Păstrează doar componenta conexă maximă
    """
    G_clean = GraphPreprocessor.get_max_component(G)
    return G_clean


def preproceseaza_toate_grafurile(input_dir, output_dir=None):
    """
    Preprocesează toate grafurile dintr-un folder.
    Dacă output_dir este None, suprascrie fișierele originale.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        asigura_folder(output_dir)
    
    print("\n🔧 PREPROCESARE GRAFURI...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    
    files = [f for f in os.listdir(input_dir) if f.endswith("_output.csv")]
    preprocesate = []
    
    for file in files:
        cale_intrare = os.path.join(input_dir, file)
        cale_iesire = os.path.join(output_dir, file)
        
        print(f"   → Preprocesare {file}...")
        G = load_graph_from_csv(cale_intrare)
        if G is None or G.number_of_nodes() == 0:
            print(f"      ⚠️ Eroare la încărcare, sărit...")
            continue
        
        G_curatat = preproceseaza_graf(G)
        
        # Salvare
        with open(cale_iesire, 'w') as f:
            for u, v in sorted(G_curatat.edges()):
                f.write(f"{u},{v}\n")
        
        print(f"      ✅ {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii → "
              f"{G_curatat.number_of_nodes()} noduri, {G_curatat.number_of_edges()} muchii")
        preprocesate.append(cale_iesire)
    
    print(f"✅ Preprocesare completă! {len(preprocesate)} fișiere procesate.")
    return preprocesate


def afiseaza_meniu():
    """Afișează meniul interactiv."""
    print("\n" + "═" * 60)
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "PIPELINE ANALIZĂ CENTRALITĂȚI" + " " * 16 + "║")
    print("╠" + "═" * 58 + "╣")
    print("║  1. Generează grafuri NOI (sintetice + reale)              ║")
    print("║  2. Continuă cu grafurile EXISTENTE (fără generare)        ║")
    print("║  3. Doar PREPROCESEAZĂ grafurile existente                 ║")
    print("║  4. Generează NOI + PREPROCESEAZĂ automat                  ║")
    print("║  5. Ieșire                                                  ║")
    print("╚" + "═" * 58 + "╝")
    print()


def ask_yes_no(question, default="da"):
    """Întreabă utilizatorul da/nu."""
    opts = "da/nu"
    if default == "da":
        opts = "Da/nu"
    elif default == "nu":
        opts = "da/Nu"
    
    raspuns = input(f"{question} ({opts}): ").strip().lower()
    if raspuns in ["da", "d", "yes", "y"]:
        return True
    if raspuns in ["nu", "n", "no"]:
        return False
    return default == "da"


# ============================================================================
# FUNCȚIA PRINCIPALĂ DE ANALIZĂ (refactorizată)
# ============================================================================

def ruleaza_analiza(graf_dir, report_dir, centrality_dir, summary_file):
    """
    Rulează analiza completă pe grafurile dintr-un director dat.
    """
    print("\n" + "═" * 60)
    print("🚀 PORNIRE ANALIZĂ")
    print(f"   Director grafuri: {graf_dir}")
    print(f"   Director rapoarte: {report_dir}")
    print("═" * 60)
    
    sim_engine = SIRSimulationEngine()
    analyzer = CentralityMaster()
    master_stochastic_data = []
    
    asigura_folder(centrality_dir)
    asigura_folder(report_dir)
    
    # 1. Parcurgere rețele
    files = [f for f in os.listdir(graf_dir) if f.endswith("_output.csv")]
    
    if not files:
        print(f"❌ Nu s-au găsit fișiere CSV în {graf_dir}")
        return None
    
    for file in files:
        path = os.path.join(graf_dir, file)
        network_key = file.replace("_output.csv", "")
        
        print(f"\n[PROCESARE] Rețeaua: {network_key}")
        
        G = load_graph_from_csv(path)
        if G is None or G.number_of_nodes() == 0:
            print(f"   ⚠️ Graf gol sau invalid, sărit...")
            continue
        
        # A. Simulări SIR
        print(f"   -> Pasul 1: Rulăm {NR_RULARI} simulări SIR...")
        raw_sim_results = sim_engine.run_full_experiment(G, P_VALUES, NR_RULARI)
        
        for r in raw_sim_results:
            r["network"] = network_key
            master_stochastic_data.append(r)
        
        # B. Centralități matematice
        print(f"   -> Pasul 2: Calculăm centralități matematice...")
        analyzer.compute_all(path, os.path.join(centrality_dir, network_key))
    
    # 2. Rezumat stocastic
    print("\n[PAS 3] Generare centralitate stocastică...")
    df_raw = pd.DataFrame(master_stochastic_data)
    
    if df_raw.empty:
        print("❌ Nu există date de simulare!")
        return None
    
    summary_df = df_raw.groupby(
        ["network", "infection_probability", "source_vertex"],
        as_index=False
    ).agg(mean_total_infected=("total_infected", "mean"))
    
    summary_df.to_csv(summary_file, index=False)
    print(f"   ✅ Rezumat salvat în: {summary_file}")
    
    # 3. Corelații
    print("\n[PAS 4] Calcul corelații...")
    centralities_data = load_centralities_for_analysis(centrality_dir, NETWORK_MAP)
    df_corr = compute_correlations(summary_df, centralities_data)
    
    if 'target_metric' not in df_corr.columns:
        df_corr['target_metric'] = 'mean_inf'
    
    corr_report_path = os.path.join(report_dir, "corelatii_finale_sir.csv")
    df_corr.to_csv(corr_report_path, index=False)
    print(f"   ✅ Corelații salvate în: {corr_report_path}")
    
    # 4. Vizualizări
    print("\n[PAS 5] Generare vizualizări...")
    vis = SIRVisualizer(output_dir=os.path.join(report_dir, "visuals"))
    
    # Grafice simple
    for net in df_corr['network'].unique():
        vis.plot_correlations(df_corr, net)
        vis.plot_infection_trend(summary_df, net)
    
    # Grafice avansate
    vis.generate_all_advanced_plots(df_corr)
    
    # Animație
    try:
        vis.generate_video(df_corr, video_format='mp4')
    except:
        try:
            vis.generate_video(df_corr, video_format='gif')
        except:
            print("   ⚠️ Animația nu a putut fi generată")
    
    # Raport text
    vis.generate_text_report(df_corr)
    
    print("\n" + "═" * 60)
    print("✨ ANALIZĂ COMPLETĂ!")
    print(f"   Rezultate în: {report_dir}")
    print("═" * 60)
    
    return df_corr


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Directorul implicit pentru grafuri
    graf_dir_default = INPUT_DIR
    graf_dir_nou = "grafuri_noi_generate"
    
    while True:
        afiseaza_meniu()
        optiune = input("Alege opțiunea (1/2/3/4/5): ").strip()
        
        if optiune == "1":
            # Generează grafuri NOI (fără preprocesare)
            print("\n📁 Generare grafuri NOI...")
            preproceseaza_opt = ask_yes_no("Dorești preprocesare automată după generare?", "nu")
            
            genereaza_toate_grafurile(
                output_dir=graf_dir_nou,
                preproceseaza=preproceseaza_opt,
                elimina_cicluri=False
            )
            
            # Întrebăm dacă vrem să continuăm cu analiza
            if ask_yes_no("\nDorești să rulezi analiza pe grafurile nou generate?", "da"):
                ruleaza_analiza(
                    graf_dir=graf_dir_nou,
                    report_dir=REPORT_DIR + "_nou",
                    centrality_dir=CENTRALITY_DIR + "_nou",
                    summary_file=f"stochastic_summary_nou.csv"
                )
            else:
                print("   → Poți rula analiza mai târziu cu opțiunea 2, specificând directorul corect.")
        
        elif optiune == "2":
            # Continuă cu grafurile existente
            graf_folder = input(f"Director cu grafuri (ENTER pentru '{graf_dir_default}'): ").strip()
            if not graf_folder:
                graf_folder = graf_dir_default
            
            if not os.path.exists(graf_folder):
                print(f"❌ Directorul '{graf_folder}' nu există!")
                continue
            
            report_folder = input(f"Director rapoarte (ENTER pentru '{REPORT_DIR}'): ").strip()
            if not report_folder:
                report_folder = REPORT_DIR
            
            ruleaza_analiza(
                graf_dir=graf_folder,
                report_dir=report_folder,
                centrality_dir=CENTRALITY_DIR,
                summary_file=SIM_SUMMARY_FILE
            )
        
        elif optiune == "3":
            # Doar preprocesează
            graf_folder = input(f"Director cu grafuri de preprocesat (ENTER pentru '{graf_dir_default}'): ").strip()
            if not graf_folder:
                graf_folder = graf_dir_default
            
            output_folder = input(f"Director output (ENTER pentru suprascriere): ").strip()
            if not output_folder:
                output_folder = graf_folder
            
            preproceseaza_toate_grafurile(graf_folder, output_folder)
        
        elif optiune == "4":
            # Generează NOI + PREPROCESEAZĂ automat
            print("\n📁 Generare grafuri NOI cu preprocesare...")
            genereaza_toate_grafurile(
                output_dir=graf_dir_nou,
                preproceseaza=True,
                elimina_cicluri=False
            )
            
            print("\n🔧 Aplicare preprocesare suplimentară...")
            preproceseaza_toate_grafurile(graf_dir_nou, graf_dir_nou)
            
            # Analiză automată
            if ask_yes_no("\nDorești să rulezi analiza pe grafurile preprocesate?", "da"):
                ruleaza_analiza(
                    graf_dir=graf_dir_nou,
                    report_dir=REPORT_DIR + "_preprocesat",
                    centrality_dir=CENTRALITY_DIR + "_preprocesat",
                    summary_file=f"stochastic_summary_preprocesat.csv"
                )
        
        elif optiune == "5":
            print("👋 La revedere!")
            break
        
        else:
            print("❌ Opțiune invalidă! Alege 1-5.")
        
        print("\n" + "─" * 40)


if __name__ == "__main__":
    main()