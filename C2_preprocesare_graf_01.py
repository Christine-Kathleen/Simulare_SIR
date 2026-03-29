from collections import defaultdict

def citireMuchii(nume_fisier): 
    muchii = [] #lista de muchii 

    with open(nume_fisier, "r") as f: #context manager pt inchiderea automata a fisierului 
        for linie in f:
            linie = linie.strip() #se elimina "\n", "\t" si spatii libere de la inceput si sfarsit de string 

            if not linie: #daca linia nu exista, se proceseaza urmatoarea (un string vid se evalueaza ca "False")
                continue

            parti = linie.split(",")

            if len(parti) != 2: #nu sunt date extremitatile sau sunt mai multe
                continue

            try: #daca conversia nu reuseste, linia se omite
                a = int(parti[0])
                b = int(parti[1])
            except ValueError:
                continue

            if a == b:
                continue

            if a > b:
                a, b = b, a

            muchii.append(tuple((a, b)))

    return sorted(set(muchii)) #sortare cresc dupa prima extremitate

class EliminareCiclu:
    def __init__(self):
        self.parent = {}

    #nodul reprezentatnt pt fiecare subarbore e chiar nodul insusi, la inceput => se pleaca de la o padure formata din n subarbori
    def adaugareNod(self, x):
        if x not in self.parent: #se ver daca nodul exista deja in dictionar (ca si cheie)
            self.parent[x] = x

    def cautareParinte(self, x): 
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def verificareCiclu(self, x, y):
        rootX = self.cautareParinte(x)
        rootY = self.cautareParinte(y)

        if rootX == rootY: #daca extremitatile muchiei au acelasi parinte, sunt ignorate => se formeaza ciclu 
            return False

        self.parent[rootY] = rootX

        return True

def eliminareCicluri(muchii): 
    ec = EliminareCiclu()
    rezultat = []

    for a, b in muchii:
        ec.adaugareNod(a)
        ec.adaugareNod(b)

        if ec.verificareCiclu(a, b): #adaugam doar daca nu creeaza ciclu, altfel, muchia se ignora 
            rezultat.append((a, b))

    return rezultat

def construireListaAdiacenta(muchii): #lista de adiacenta
    #daca se acceseaza "lista[a]" si cheia "a" nu exista, se va crea automat "lista[a] = []", apoi se face "append"
    lista = defaultdict(list)  

    for a, b in muchii:
        lista[a].append(b)
        lista[b].append(a)

    return lista

def DFS(start, graf, viz, comp, c):
    viz[start] = True
    comp[start] = c #nodul de start e marcat cu o val ce semnifica nr comp in care se afla 

    for vecin in graf.get(start, []):
        if not viz[vecin]:
            DFS(vecin, graf, viz, comp, c)

def determinareComponenteConexe(graf, n):
    viz = [False] * (n + 1)
    comp = [0] * (n + 1) 
    ncc = 0 #nr componente conexe

    for i in range(0, n + 1):
        if not viz[i]:
            ncc += 1
            DFS(i, graf, viz, comp, ncc)

    return comp, ncc

def determinareComponentaConexaMaxima(muchii, n):
    graf = construireListaAdiacenta(muchii)
    comp, ncc = determinareComponenteConexe(graf, n)

    if ncc == 0:
        return []

    dimensiune = [0] * (ncc + 1) #nr cate noduri are fiecare comp (vect de dim)

    for i in range(0, n + 1):
        if comp[i] != 0:
            dimensiune[comp[i]] += 1

    comp_max = 1

    for i in range(2, ncc + 1): #se alege comp cu nr max de noduri ce o compun 
        if dimensiune[i] > dimensiune[comp_max]:
            comp_max = i

    rezultat = []

    for a, b in muchii: #din comp max se retin muchiile
        if comp[a] == comp_max and comp[b] == comp_max:
            rezultat.append((a, b))

    lista_adiacenta = defaultdict(list)
    
    for a, b in rezultat:
        lista_adiacenta[a].append(b)
        lista_adiacenta[b].append(a)

    return dict(lista_adiacenta)

def determinareNMaxim(muchii):
    noduri = set()
    
    for a, b in muchii:
        noduri.add(a)
        noduri.add(b)

    return max(noduri) if noduri else 0

def procesareGraf(fisier):
    muchii = citireMuchii(fisier)
    muchii = eliminareCicluri(muchii)
    muchii = determinareComponentaConexaMaxima(muchii, determinareNMaxim(muchii))

    return muchii

def scriereMuchii(fisier_output, lista_adiacenta):
    with open(fisier_output, "w") as f:
        for a in lista_adiacenta:
            for b in lista_adiacenta[a]:
                if a < b:
                    f.write(f"{a},{b}\n")

if __name__ == "__main__":
    scriereMuchii("new_output_graph5.csv", procesareGraf("graph5.csv"))
