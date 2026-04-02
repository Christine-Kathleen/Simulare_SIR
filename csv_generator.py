import os
import csv
import networkx as nx
import numpy as np

def tabel_prob(p, grad_maxim):
    return [1.0 - (1.0 - p) ** k for k in range(grad_maxim + 1)]

def ruleaza_simulare_core(vecini, noduri, sursa, prob_tabel, ticks_maxim=200):
    """Execută o singură propagare SIR."""
    infectati = {sursa}
    recuperati = set()
    infection_tick = {sursa: 0} 

    for tick in range(1, ticks_maxim + 1):
        potentiali = noduri - infectati - recuperati
        # Logica de infecție bazată pe probabilitatea p cumulată
        noi_I = {nd for nd in potentiali 
                 if (k := len(vecini[nd] & infectati)) > 0 
                 and np.random.random() < prob_tabel[k]}
        
        for nd in noi_I: 
            infection_tick[nd] = tick
            
        recuperati |= infectati
        infectati = noi_I
        if len(infectati) == 0: break

    return {"sursa": sursa, "infection_tick": infection_tick, "ticks": tick}

def export_nomenclator_csv(graf, rezultate_batch, p, run_id, tip_graf, output_dir):
    """Generează fișierul CSV detaliat (Nomenclator)."""
    noduri_s = sorted(graf.nodes())
    nume_fisier = f"Graf_{tip_graf}_{len(noduri_s)}_p{p}_run{run_id}.csv"
    cale = os.path.join(output_dir, nume_fisier)
    
    max_ticks_batch = max(res["ticks"] for res in rezultate_batch.values())

    with open(cale, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Clock Tick"] + [f"Sursa_{nd}" for nd in noduri_s]
        w.writerow(header)

        for t in range(1, max_ticks_batch + 1):
            rand = [f"t{t}"]
            for sursa in noduri_s:
                inf_dict = rezultate_batch[sursa]["infection_tick"]
                noduri_la_t = [str(nd) for nd, tick_val in inf_dict.items() if tick_val == t]
                rand.append(",".join(noduri_la_t))
            w.writerow(rand)