# Simulare SIR pe Retele Complexe

Simularea propagarii unui agent infectios folosind **modelul SIR** (Susceptibil-Infectat-Recuperat)
pe retele reale si generate, cu analiza corelatiei dintre **centralitati clasice** si **impactul stochastic** al nodurilor.

**Framework:** Python + NetworkX + SciPy  
**Tip retele:** 6 grafuri (3 reale + 3 generate), 100 noduri fiecare  
**Total simulari:** 30.000 (6 retele × 5 valori p × 10 rulari × 100 surse)

---

## Echipa

| Membru | Contributie |
|--------|-------------|
| Deva Dan | Generare grafuri, implementare algoritm GNM conex |
| Hirceaga Andreea | Generare grafuri, preprocesare date |
| Pascu Lorena | Generare grafuri, curatare date |
| Racz Christine | Generare grafuri, coordonare repository |

---

## Modelul SIR

Fiecare nod din retea se afla in una din 3 stari:

- **S (Susceptibil)** — poate fi infectat
- **I (Infectat)** — infectat si poate infecta vecinii Susceptibili cu probabilitatea `p`
- **R (Recuperat)** — nu mai poate fi infectat sau infecta

Simularea porneste cu un singur nod sursa in starea I. La fiecare pas de timp, nodurile Infectate
incearca sa-si infecteze vecinii Susceptibili. Simularea se opreste cand nu mai exista noduri Infectate
sau se atinge limita de 200 de pasi.

---

## Structura Proiectului

```
Simulare_SIR/
├── main.py                              <- generare grafuri aleatorii conexe
├── programCuratare.py                   <- preprocesare grafuri reale (componenta maxima)
├── runner.py                            <- coordonare simulare SIR paralela
├── csv_generator.py                     <- motorul simularii SIR
├── config.yaml                          <- configuratia experimentului
├── full_pipeline_centrality_and_correlations.py  <- pipeline centrality + correlations
├── C2_preprocesare_graf_01.py           <- preprocesare graful 1
├── C2_preprocesare_graf_02.py           <- preprocesare graful 2
├── C6_graph_utils.py                    <- utilitare grafuri
├── C6_degree_centrality_module.py       <- calcul Degree Centrality
├── C6_closeness_centrality_module.py    <- calcul Closeness Centrality
├── C6_katz_centrality_module.py         <- calcul Katz Centrality
├── C6_main_classical_centralities.py    <- script principal centralitati clasice
├── C8_classical_vs_stochastic_centralities.py  <- corelatie clasic vs. stochastic
├── run_centralities_sorted.py           <- rulare centralitati cu output sortat
├── explicatie_config.txt                <- explicatii detaliate configuratie
├── grafuri_naturale_si_generate_curatate/   <- grafuri in format CSV
├── scoruri_centralitati_clasice/        <- scoruri degree, closeness, katz per retea
├── simulation_results/                  <- rezultate simulari brute
├── documentatii/                        <- documentatie teoretica per centralitate
└── graphs/                              <- grafice generate
```

---

## Instalare

```bash
# Cloneaza repository-ul
git clone https://github.com/Christine-Kathleen/Simulare_SIR.git
cd Simulare_SIR

# Creeaza si activeaza mediu virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Instaleaza dependentele
pip install networkx matplotlib scipy numpy pandas pyyaml
```

---

## Rulare

### 1. Generare graf nou (optional)
```bash
python main.py
# Varfuri: 100
# Muchii: 275
```
Genereaza un graf aleatoriu conex si il salveaza in `graph5.csv`.

### 2. Preprocesare graf real
```bash
python programCuratare.py
```
Citeste un graf real din CSV, pastreaza componenta maxima conexa si salveaza rezultatul in format standardizat.

### 3. Rulare simulare SIR
```bash
python runner.py
```
Ruleaza simularea SIR pe toate cele 6 retele configurate in `config.yaml`.
Rezultatele se salveaza in `REZULTATE_EXPERIMENT/raw_results.csv`.

> **Durata estimata:** depinde de hardware. Cu `n_jobs = -1` (toate nucleele), simularea ruleaza in paralel.

### 4. Calcul centralitati clasice
```bash
python C6_main_classical_centralities.py
```
Calculeaza Degree, Closeness si Katz Centrality pentru toate cele 6 retele.
Rezultatele se salveaza in `scoruri_centralitati_clasice/`.

### 5. Corelatie clasic vs. stochastic
```bash
python C8_classical_vs_stochastic_centralities.py
```
Calculeaza coeficientii Pearson, Spearman si Kendall intre centralitati clasice si impactul SIR.
Rezultatele se salveaza in `correlations_all_metrics.csv`.

### 6. Pipeline complet (alternativa)
```bash
python full_pipeline_centrality_and_correlations.py
```
Ruleaza intregul pipeline: incarcare graf → centralitati → corelatie → salvare rezultate.

---

## Configuratie

Fisierul `config.yaml` controleaza intregul experiment:

```yaml
valori_p:         [0.01, 0.04, 0.10, 0.30, 0.50]  # probabilitati de infectare
rulari_pe_setare: 10                                # repetitii per combinatie
optiuni:
  n_jobs: -1        # -1 = toate nucleele procesorului
  ticks_maxim: 200  # limita maxima de pasi per simulare
  seed_random: 42   # seed pentru reproductibilitate
```

---

## Retele Utilizate

| Retea | Tip | Noduri | Muchii | Densitate |
|-------|-----|--------|--------|-----------|
| 99muchii_100noduri_generat | Generata | 100 | 99 | ~2% |
| 99muchii_100noduri_natural | Reala | 100 | 99 | ~2% |
| 275muchii_100noduri_generat | Generata | 100 | 275 | ~5.6% |
| 275muchii_100noduri_natural | Reala | 100 | 275 | ~5.6% |
| 1000muchii_100noduri_generat | Generata | 100 | 1000 | ~20% |
| 1000muchii_100node_natural | Reala | 100 | 1000 | ~20% |

---

## Rezultate

### Centralitati calculate
- **Degree Centrality** — numarul normalizat de conexiuni directe ale unui nod
- **Closeness Centrality** — inversul sumei distantelor minime catre toate celelalte noduri
- **Katz Centrality** — influenta globala ponderata, incluzand vecini de toate ordinele (factor atenuare α)

### Metrici stochastice (din simulare)
- **mean_total_infected** — media nodurilor infectate pe toate rurarile cand nodul este sursa
- **max_total_infected** — maximul nodurilor infectate observat cand nodul este sursa

### Concluzii principale
Toate cele trei centralitati sunt predictori puternici pentru **influenta medie** (`mean_inf`)
doar in regimul intermediar al lui p (~0.04-0.10), unde procesul SIR este sensibil la structura retelei.
Pentru **focarele maxime** (`max_inf`), niciuna nu ofera predictii consistente —
evenimentele extreme sunt dominate de dinamica stohastica, nu de pozitia structurala a nodului.

---

## Dependente

| Librarie | Utilizare |
|----------|-----------|
| `networkx` | Structuri de date graf, calcul centralitati |
| `scipy` | Coeficienti de corelatie (Pearson, Spearman, Kendall) |
| `numpy` | Calcule numerice, probabilitati |
| `pandas` | Procesare date CSV |
| `matplotlib` | Vizualizare grafuri |
| `pyyaml` | Citire fisier de configuratie |

---

## Documentatie

Folderul `documentatii/` contine documentatie teoretica pentru fiecare centralitate:

| Fisier | Autor | Subiect |
|--------|-------|---------|
| `Degree_Centrality_Saracsan_Sorana.pdf` | Saracsan Sorana | Degree Centrality |
| `Closeness_Centrality_Butyka_Ana.pdf` | Butyka Ana | Closeness Centrality |
| `Katz_Centrality_Plugar_Mihai.docx` | Plugar Mihai | Katz Centrality |
| `Betweenness_Centrality_Ous_Andreea.pdf` | Ous Andreea | Betweenness Centrality |
| `Eigenvector_Stefan_Natalia-Maria.pdf` | Stefan Natalia-Maria | Eigenvector Centrality |
| `Freeman_Centralization_Muresan_Andreea.pdf` | Muresan Andreea | Freeman Centralization |
| `PageRank_Zanfir_Denis.pdf` | Zanfir Denis | PageRank Centrality |
| `Classical_vs_stochastic_centralities.docx` | — | Comparatie clasic vs. stochastic |
| `Interpretare_corelatii.docx` | — | Interpretarea rezultatelor de corelatie |

---

> Proiect realizat pentru cursul de **Tehnici de Vizualizare a Datelor**  
> Coordonator: Bura Cotiso Andrei
