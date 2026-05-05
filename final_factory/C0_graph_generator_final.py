"""
C0_graph_generator_final.py
Modul pentru generarea celor 6 grafuri (3 sintetice + 3 reale)
Generează:
  - 99muchii_100noduri_generat
  - 275muchii_100noduri_generat
  - 1000muchii_100noduri_generat
  - 99muchii_100noduri_natural
  - 275muchii_100noduri_natural
  - 1000muchii_100noduri_natural
"""

import os
import random
import networkx as nx
import numpy as np

# ============================================================================
# CONFIGURARE
# ============================================================================

NODURI = 100
SEMILLA_ALEATORIU = 42  # pentru reproductibilitate

# Definim structurile reale (pe baza fișierelor existente)
# Pentru rețelele reale, am observat că:
# - 99muchii_natural: nodul 1 conectat la toate celelalte (99 muchii)
# - 275muchii_natural: stea + muchii suplimentare
# - 1000muchii_natural: stea + multe muchii suplimentare

# Vom recrea aceste structuri din fișierele existente
# (sau le vom încărca dacă există deja)

def incarca_muchii_din_fisier_existent(filepath):
    """Încarcă muchiile dintr-un fișier CSV existent (pentru reproductibilitate)."""
    muchii = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        muchii.append((u, v))
                    except ValueError:
                        continue
    return muchii


# ============================================================================
# GENERARE GRAFURI SINTETICE (Erdős–Rényi cu număr fix de muchii)
# ============================================================================

def genereaza_graf_sintetic(noduri, numar_muchii, seed=SEMILLA_ALEATORIU):
    """
    Generează un graf Erdős–Rényi cu număr fix de muchii.
    Folosește metoda de eșantionare uniformă a muchiilor.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Numărul maxim posibil de muchii pentru un graf complet
    max_muchii = noduri * (noduri - 1) // 2
    
    if numar_muchii > max_muchii:
        raise ValueError(f"Prea multe muchii! Maxim: {max_muchii}")
    
    # Creăm un graf gol
    G = nx.Graph()
    G.add_nodes_from(range(noduri))
    
    # Generăm toate muchiile posibile și le amestecăm
    toate_muchiile = [(i, j) for i in range(noduri) for j in range(i + 1, noduri)]
    random.shuffle(toate_muchiile)
    
    # Adăugăm primele `numar_muchii` muchii
    muchii_ales = toate_muchiile[:numar_muchii]
    G.add_edges_from(muchii_ales)
    
    return G


# ============================================================================
# GENERARE GRAFURI REALE (pe baza structurii din fișierele existente)
# ============================================================================

def genereaza_graf_real_stea_plus_muchii(noduri, numar_muchii, seed=SEMILLA_ALEATORIU):
    """
    Generează un graf real care respectă structura rețelelor originale:
    - Nodul 1 (indexat de la 1) este conectat la multe noduri (stea parțială)
    - Se adaugă muchii suplimentare între nodurile 2..N
    
    Pentru 99 muchii: doar stea completă (nodul 1 legat de toate celelalte)
    Pentru 275/1000: stea + muchii aleatoare suplimentare
    """
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    noduri_list = list(range(1, noduri + 1))
    G.add_nodes_from(noduri_list)
    
    # Pas 1: Adăugăm steaua cu nodul 1 conectat la toate celelalte (99 muchii aici)
    for i in range(2, noduri + 1):
        G.add_edge(1, i)
    
    # Pas 2: Dacă avem nevoie de mai multe muchii, le adăugăm aleator
    muchii_existente = set(G.edges())
    muchii_curente = len(muchii_existente)
    
    if numar_muchii > muchii_curente:
        # Generăm muchii suplimentare între nodurile 2..N
        noduri_fara_1 = list(range(2, noduri + 1))
        toate_muchiile_posibile = [(i, j) for i in noduri_fara_1 
                                   for j in noduri_fara_1 if i < j]
        random.shuffle(toate_muchiile_posibile)
        
        muchii_de_adaugat = numar_muchii - muchii_curente
        for u, v in toate_muchiile_posibile:
            if (u, v) not in muchii_existente and (v, u) not in muchii_existente:
                G.add_edge(u, v)
                muchii_existente.add((u, v))
                muchii_curente += 1
                if muchii_curente >= numar_muchii:
                    break
    
    # Dacă am adăugat prea multe (caz rar), eliminăm aleator
    while G.number_of_edges() > numar_muchii:
        muchii_list = list(G.edges())
        muchie_de_eliminat = random.choice(muchii_list)
        # Nu eliminăm muchiile din stea (cele care implică nodul 1)
        if muchie_de_eliminat[0] != 1 and muchie_de_eliminat[1] != 1:
            G.remove_edge(*muchie_de_eliminat)
    
    return G


# ============================================================================
# SALVARE GRAF ÎN FORMAT CSV
# ============================================================================

def salveaza_graf_csv(G, nume_fisier, output_dir):
    """
    Salvează graful în format CSV (muchii, câte una pe linie).
    """
    os.makedirs(output_dir, exist_ok=True)
    cale = os.path.join(output_dir, nume_fisier)
    
    # Obținem nodurile sortate pentru consistență
    noduri = sorted(G.nodes())
    
    # Dacă nodurile sunt indexate de la 0, le păstrăm așa
    # Dacă sunt indexate de la 1, le păstrăm așa
    with open(cale, 'w') as f:
        for u, v in sorted(G.edges()):
            f.write(f"{u},{v}\n")
    
    print(f"   ✅ Salvat: {cale} ({G.number_of_edges()} muchii, {G.number_of_nodes()} noduri)")
    return cale


# ============================================================================
# CURĂȚARE GRAF (preprocesare)
# ============================================================================

def preproceseaza_graf(G, remove_cycles=False):
    """
    Aplică preprocesarea asupra grafului:
    - Păstrează doar componenta conexă maximă
    - Opțional, elimină ciclurile (transformă în arbore)
    """
    from C2_preprocessor_final import GraphPreprocessor
    
    # Pas 1: Componenta maximă
    G_clean = GraphPreprocessor.get_max_component(G)
    
    # Pas 2: Dacă se cere, elimină ciclurile
    if remove_cycles:
        G_clean = GraphPreprocessor.remove_cycles_dsu(G_clean)
    
    return G_clean


# ============================================================================
# GENERARE TOATE GRAFURILE
# ============================================================================

def genereaza_toate_grafurile(output_dir="grafuri_noi_generate", 
                              preproceseaza=False, 
                              elimina_cicluri=False):
    """
    Generează toate cele 6 grafuri.
    
    Parametri:
        output_dir: folderul unde se salvează grafurile
        preproceseaza: dacă True, aplică preprocesarea (componentă maximă)
        elimina_cicluri: dacă True și preproceseaza=True, elimină și ciclurile
    """
    print("\n" + "=" * 60)
    print("🔧 GENERARE GRAFURI")
    print("=" * 60)
    print(f"   Output directory: {output_dir}")
    print(f"   Preprocesare: {'DA' if preproceseaza else 'NU'}")
    print(f"   Eliminare cicluri: {'DA' if elimina_cicluri else 'NU'}")
    print("=" * 60)
    
    # Configurații pentru cele 6 grafuri
    config_sintetice = [
        (99, "99muchii_100noduri_generat_output.csv"),
        (275, "275muchii_100noduri_generat_output.csv"),
        (1000, "1000muchii_100noduri_generat_output.csv"),
    ]
    
    config_reale = [
        (99, "99muchii_100noduri_natural_output.csv"),
        (275, "275muchii_100noduri_natural_output.csv"),
        (1000, "1000muchii_100noduri_natural_output.csv"),
    ]
    
    grafuri_generate = []
    
    # 1. Generare grafuri sintetice
    print("\n📊 Generare grafuri SINTETICE (Erdős–Rényi)...")
    for muchii, nume_fisier in config_sintetice:
        print(f"   → Generare {nume_fisier} ({muchii} muchii)...")
        G = genereaza_graf_sintetic(NODURI, muchii)
        
        if preproceseaza:
            G = preproceseaza_graf(G, remove_cycles=elimina_cicluri)
        
        cale = salveaza_graf_csv(G, nume_fisier, output_dir)
        grafuri_generate.append(cale)
    
    # 2. Generare grafuri reale
    print("\n🏛️ Generare grafuri REALE (stea + muchii suplimentare)...")
    for muchii, nume_fisier in config_reale:
        print(f"   → Generare {nume_fisier} ({muchii} muchii)...")
        G = genereaza_graf_real_stea_plus_muchii(NODURI, muchii)
        
        if preproceseaza:
            G = preproceseaza_graf(G, remove_cycles=elimina_cicluri)
        
        cale = salveaza_graf_csv(G, nume_fisier, output_dir)
        grafuri_generate.append(cale)
    
    print("\n" + "=" * 60)
    print(f"✅ Toate cele {len(grafuri_generate)} grafuri au fost generate!")
    print(f"   Director: {output_dir}")
    print("=" * 60)
    
    return grafuri_generate


# ============================================================================
# TESTARE DIRECTĂ
# ============================================================================

if __name__ == "__main__":
    # Testare rapidă
    genereaza_toate_grafurile(
        output_dir="grafuri_test_generate",
        preproceseaza=False,
        elimina_cicluri=False
    )

