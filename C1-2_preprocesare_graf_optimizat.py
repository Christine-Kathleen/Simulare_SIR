from collections import defaultdict

class EliminareCiclu: #"https://www.geeksforgeeks.org/dsa/union-by-rank-and-path-compression-in-union-find-algorithm/"
    def __init__(self):
        self.parent = {}
        self.rank = {} #estimare a inaltimii arborelui 
        self.size = {}

    def adaugareNod(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1

    def cautareParinte(self, x): #path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]] #se actualizeaza si parintele nodului dat
            x = self.parent[x]

        return x

    def verificareCiclu(self, x, y):
        rootX = self.cautareParinte(x)
        rootY = self.cautareParinte(y)

        if rootX == rootY:
            return False

        if self.rank[rootX] < self.rank[rootY]: #pt ca rad arborelui mai mic sa fie sub cea a celui mai mare => interschimbare
            rootX, rootY = rootY, rootX

        self.parent[rootY] = rootX
        self.size[rootX] += self.size[rootY]

        if self.rank[rootX] == self.rank[rootY]:
            self.rank[rootX] += 1

        return True

def procesareGraf(fisier):
    ec = EliminareCiclu()
    muchii_unice = set()

    with open(fisier, "r") as f:
        for linie in f:
            linie = linie.strip()

            if not linie:
                continue

            parti = linie.split(",")

            if len(parti) != 2: 
                continue

            try: 
                a = int(parti[0])
                b = int(parti[1])
            except ValueError:
                continue

            if a == b:
                continue

            if a > b:
                a, b = b, a

            muchii_unice.add((a, b))

    muchii_fara_cicluri = []

    for a, b in sorted(muchii_unice): #nu se actualizeaza aici parintii pt toate nodurile, ci doar in partea (*) de mai jos 
        ec.adaugareNod(a)
        ec.adaugareNod(b)

        if ec.verificareCiclu(a, b):
            muchii_fara_cicluri.append((a, b))

    if not muchii_fara_cicluri:
        return {}

    max_root = None
    max_size = 0

    #se itereaza prin parintii nodurilor (prin chei) si (*) se actualizeaza parintele pt fiecare nod; se cauta componenta conexa 
    #cu nr max de noduri si i se retine nodul reprezentant (rad)
    for node in ec.parent: 
        root = ec.cautareParinte(node)

        if ec.size[root] > max_size:
            max_size = ec.size[root]
            max_root = root

    componenta_max = []

    for a, b in muchii_fara_cicluri:
        if ec.cautareParinte(a) == max_root:
            componenta_max.append((a, b))

    lista_adiacenta = defaultdict(list)

    for a, b in componenta_max:
        lista_adiacenta[a].append(b)
        lista_adiacenta[b].append(a)

    return dict(lista_adiacenta)

def scriereMuchii(fisier_output, lista_adiacenta):
    with open(fisier_output, "w") as f:
        for a in lista_adiacenta:
            for b in lista_adiacenta[a]:
                if a < b:
                    f.write(f"{a},{b}\n")

if __name__ == "__main__":
    scriereMuchii("output_muchii_100_orientat.csv", procesareGraf("muchii_100_orientat.csv"))
