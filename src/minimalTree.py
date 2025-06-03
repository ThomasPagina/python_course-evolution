import itertools
import networkx as nx
from graphviz import Digraph

# 1. Beispiel-Sätze definieren
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A swift auburn fox leaps over a sleepy canine.",
    "The quick brown fox hopped over the lazy dog.",
    "A fast brown fox jumps above the lazy hound.",
    "The agile brown fox vaults over the lethargic dog.",
    "A quick brown fox jumps over the lazy dog!",
    "The quick brown fox jumps over the lazy cat.",
    "A swift auburn fox hops over a sleepy dog.",
    "The quick brown fox jumped over the lazy dog.",
    "A fast brown fox hopped above the lazy dog."
]

# 2. Levenshtein-Distanz-Funktion

def levenshtein(a: str, b: str) -> int:
    """
    Berechnet die Levenshtein-Distanz zwischen zwei Strings.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Einfügen
                dp[i][j - 1] + 1,      # Löschen
                dp[i - 1][j - 1] + cost # Ersetzen
            )
    return dp[m][n]

# 3. Distanzen aller Paare berechnen
pairs = list(itertools.combinations(range(len(sentences)), 2))
distances = {(i, j): levenshtein(sentences[i], sentences[j]) for i, j in pairs}

# 4. Graph aufbauen und Gewichte setzen
G = nx.Graph()
for i, s in enumerate(sentences):
    G.add_node(i, label=s)
for (i, j), w in distances.items():
    G.add_edge(i, j, weight=w)

# 5. Minimalen Spannbaum (MST) berechnen
T = nx.minimum_spanning_tree(G, weight='weight')

# 6. Wurzel auswählen (Minimale Summe der Distanzen)
sum_dist = {i: sum(distances.get(tuple(sorted((i, j))), 0) for j in G.neighbors(i)) for i in G.nodes}
root = min(sum_dist, key=sum_dist.get)

# 7. Kanten Richtung root -> Blätter ausrichten
dir_edges = list(nx.bfs_edges(T, root))

# 8. MST mit Graphviz visualisieren
dot = Digraph(name='Minimaler Veränderungsbaum', format='png')
for i in T.nodes():
    # Label: Index und Text (ggf. abschneiden)
    label = f"{i}: {sentences[i]}"
    dot.node(str(i), label)
for u, v in dir_edges:
    w = T[u][v]['weight']
    dot.edge(str(u), str(v), label=str(w))

# Ausgabe
output_path = dot.render(filename='sentence_mst', cleanup=True)
print(f"Graphviz-Datei erstellt: {output_path}")
print(dot.source)
