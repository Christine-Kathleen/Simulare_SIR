from collections import defaultdict


def CitireMuchii(nume_fisier):
    muchii=set() # folosim un set pentru eliminarea duplicatelor automat

    with open(nume_fisier,"r") as f: #deschidem fisierul in read mode
        for linie in f:
            linie=linie.strip()

            if not linie:
                continue #sarim peste liniile goale

            try:
                a,b=linie.split(",") #se impart liniile folosind ","
                a=int(a.strip())
                b=int(b.strip())
            except ValueError:
                 continue
                 #sarim peste liniile invalide(acesta va trece peste liniile care contin caractere invalide sau valorile sunt diferite de int)


            if a==b:
                continue #eliminare self loop

            muchie=tuple(sorted((a,b)))
            muchii.add(muchie)

        return sorted(muchii)

def construireListaAdiacenta(muchii): #determinare componente conexe(dfs iterativ)
    graf=defaultdict(list)

    for a,b in muchii:
        graf[a].append(b)
        graf[b].append(a)
    return graf

def determinareComponenteConexe(graf):
    viz = set()
    # set care reține nodurile deja vizitate

    componente = []
    # listă ce va conține toate componentele conexe

    for nod in graf:
        # parcurgem fiecare nod din graf

        if nod not in viz:
            # dacă nodul nu a fost vizitat, înseamnă că începe o nouă componentă

            stack = [nod]
            # inițializăm stiva pentru DFS

            comp = []
            # listă pentru nodurile componentei curente

            viz.add(nod)
            # marcăm nodul ca vizitat

            while stack:
                # cât timp există noduri în stivă

                x = stack.pop()
                # scoatem ultimul nod adăugat (DFS = LIFO)

                comp.append(x)
                # adăugăm nodul în componenta curentă

                for vecin in graf[x]:
                    # parcurgem vecinii nodului curent

                    if vecin not in viz:
                        # dacă vecinul nu a fost vizitat

                        viz.add(vecin)
                        # îl marcăm vizitat

                        stack.append(vecin)
                        # îl adăugăm în stivă pentru explorare

            componente.append(comp)
            # după ce am terminat explorarea, salvăm componenta

    return componente

def determinareComponentaConexaMaxima(componente):
    if not componente:
        return
    return max(componente,key=len)

def FiltrareMuchi(muchii,comp_max):
    comp_set=set(comp_max)
    return [(a,b)for a,b in muchii if a in comp_set and b in comp_set]

def StatsGraf(muchii):
    noduri=set() #set pentru noduri unice

    for a,b in muchii:
        noduri.add(a)
        noduri.add(b)

    print("Numarul de noduri:", len(noduri))
    print("Numarul total de muchii:",len(muchii))

def export_graf(muchii,fisier_out):
    with open(fisier_out,"w") as f:
        for a,b in muchii:
            f.write(f"{a},{b}\n")

def ProcesareGraf(fisier_in, fisier_out):
    muchii=CitireMuchii(fisier_in)
    graf=construireListaAdiacenta(muchii)
    componente=determinareComponenteConexe(graf)
    comp_max=determinareComponentaConexaMaxima(componente)
    muchii_curate=FiltrareMuchi(muchii, comp_max)
    StatsGraf(muchii_curate)
    export_graf(muchii_curate,fisier_out)
    return muchii_curate

if __name__=="__main__":
    muchii_finale=ProcesareGraf("MuchiiRandom.csv","GrafCuratat.csv")
    print ("\n Muchii finale:")
    for m in muchii_finale:
        print(m)
