import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

class SIRSimulationEngine:
   

    @staticmethod
    def generate_prob_table(p, max_degree):
        """Pre-calculează probabilitatea de infectare pentru k vecini (formula 1-(1-p)^k)."""
        return [1.0 - (1.0 - p) ** k for k in range(max_degree + 1)]

    @staticmethod
    def run_single_sir(vecini, noduri, sursa, prob_tabel, ticks_maxim=200):
        """Execută o singură propagare SIR pornind de la un singur nod sursă."""
        infectati = {sursa}
        recuperati = set()
        infection_count = {sursa: 0} # Momentan reținem doar faptul că sursa e infectată

        for tick in range(1, ticks_maxim + 1):
            # Determinăm cine poate fi infectat (vecinii nodurilor infectate care sunt încă sănătoși)
            potentiali = noduri - infectati - recuperati
            
            noi_I = set()
            for nd in potentiali:
                # Numărul de vecini ai nodului 'nd' care sunt în starea INFECTAT
                k = len(vecini[nd] & infectati)
                if k > 0:
                    if np.random.random() < prob_tabel[k]:
                        noi_I.add(nd)
            
            # Trecerea în starea RECUPERAT (imun)
            recuperati |= infectati
            infectati = noi_I
            
            # Dacă nu mai avem infectați activi, epidemia s-a oprit
            if not infectati:
                break

        return {
            "sursa": sursa, 
            "total_infectati": len(recuperati | infectati), 
            "ticks": tick
        }

    def run_full_experiment(self, G, p_values, n_runs, n_jobs=-1):
        """
        Rulează experimentul complet pe un graf dat.
        Returnează o listă de rezultate pentru fiecare rulare.
        """
        noduri_list = sorted(G.nodes())
        noduri_f = frozenset(noduri_list)
        vecini = {nd: frozenset(G.neighbors(nd)) for nd in noduri_list}
        
        # Calculăm gradul maxim pentru tabelul de probabilități
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 0
        
        # Detecție hardware pentru paralelizare
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        all_results = []

        for p in p_values:
            print(f"   -> Simulăm p={p}...")
            prob_t = self.generate_prob_table(p, max_deg)
            
            for run_id in range(1, n_runs + 1):
                # Paralelizăm execuția pentru fiecare nod sursă
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(self.run_single_sir, vecini, noduri_f, s, prob_t) 
                        for s in noduri_list
                    ]
                    
                    for fut in as_completed(futures):
                        res = fut.result()
                        all_results.append({
                            "infection_probability": p,
                            "source_vertex": res["sursa"],
                            "run_id": run_id,
                            "ticks": res["ticks"],
                            "total_infected": res["total_infectati"]
                        })
        return all_results
