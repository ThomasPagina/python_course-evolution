"""
evolutionary_tree_guided_swap_root.py

Dieses Python-Programm lädt eine JSON-Datei, die einen Mutation-Tree beschreibt, und
erstellt unter Verwendung einer Evolutionären Strategie (ES) einen hypothetischen Baum
nur aus den lebendigen (alive=true) Knoten. Dabei werden Mutations-Constraints beachtet:
Ein Kind darf sich von seinem Elternknoten nur durch inkrementelle Buchstabenänderungen
(im Alphabet vorwärts) unterscheiden.

Alle Prozesse garantieren, dass stets genau ein Root-Knoten existiert. 
Wenn ein Knoten zum Schwesterknoten des aktuellen Root-Knotens wird, 
entsteht automatisch ein neuer hypothetischer Root, der beide als Kinder hat.

Die Mutation verwendet eine geführte Strategie, die fehlerhafte Einordnungen von
den Blättern (Leaves) her korrigiert, anschließend Overflow auflöst und schließlich
redundante hypothetische Knoten entfernt. Wenn beim Korrigieren eines ungültigen
Edges ein direkter Swap von Mutter- und Tochterknoten dazu führt, dass die Relation
ganzheitlich wieder valide wird, wird dieser Swap durchgeführt:

1. Suchen einer invaliden Kante (Leaf-first).
2. Versuchen eines direkten Parent-Child-Swaps:
   - Wenn der getauschte Baum anschließend **keine** weiteren Invaliditäten enthält,
     wird der Swap übernommen.
3. Gelingt der Swap nicht oder bleibt nach dem Swap eine Ungültigkeit, erfolgt ein
   "Reattach": Das bisherige Blatt wird entweder an einen Schwesterknoten gehängt
   oder, falls das Schwesterknoten-Verfahren Root betrifft, wird ein neuer hypothetischer
   Root gebildet.
4. Existieren keine invaliden Kanten, wird ein Overflow (Knoten mit >2 Kindern) aufgelöst,
   indem ein hypothetischer Knoten zwischen zwei eng verwandten Kindern (minimale
   Levenshtein-Distanz) eingefügt wird.
5. Sind weder Invalidität noch Overflow vorhanden, wird versucht, einen redundanten
   hypothetischen Knoten zu entfernen (Bypass).

Der finale beste Baum wird mittels Graphviz visualisiert; hypothetische Knoten
werden im Graphviz-Label durch einen vorangestellten Asterisk (*) markiert.

Author: Dein Name
Date: 2025-06-02
"""

import json
import random
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import graphviz  # pip install graphviz erforderlich


# ---------------------------
#  Konfigurations-Dataclass
# ---------------------------

@dataclass
class Config:
    """
    Konfigurationsparameter für den evolutionären Algorithmus.
    """
    input_path: str = "mutation_tree.json"
    population_size: int = 50
    generations: int = 100
    remove_fraction: float = 0.2
    seed: Optional[int] = None
    output_graph_basename: str = "best_tree"


# ---------------------------
#  Datenstrukturen
# ---------------------------

class Node:
    """
    Repräsentiert einen Knoten im Baum.
    Kann ein originaler (lebendiger) Knoten aus der JSON-Datei sein
    oder ein hypothetischer Knoten, der während der Evolution hinzugefügt wird.
    """
    def __init__(self, node_id: int, sequence: str, is_hypothetical: bool = False) -> None:
        self.id: int = node_id
        self.sequence: str = sequence
        self.is_hypothetical: bool = is_hypothetical

    def __repr__(self) -> str:
        kind = "HYP" if self.is_hypothetical else "REAL"
        return f"<Node {self.id} ({kind}): {self.sequence}>"


class Tree:
    """
    Repräsentiert einen Baum aus Node-Objekten.
    Eltern-Kind-Beziehungen werden über Dictionaries gespeichert.
    """
    def __init__(
        self,
        nodes: Dict[int, Node],
        parent_map: Dict[int, Optional[int]],
        children_map: Dict[int, List[int]]
    ) -> None:
        """
        :param nodes: Dict von node_id -> Node-Instanz (inkl. hypothetischer Knoten)
        :param parent_map: Dict von node_id -> parent_id oder None, wenn Wurzel
        :param children_map: Dict von node_id -> Liste der children_ids
        """
        self.nodes: Dict[int, Node] = nodes
        self.parent_map: Dict[int, Optional[int]] = parent_map
        self.children_map: Dict[int, List[int]] = children_map

    def copy(self) -> "Tree":
        """
        Erzeugt eine tiefe Kopie des aktuellen Baumes,
        sodass Änderungen an der Kopie die Ursprungsversion nicht beeinflussen.
        """
        new_nodes = {
            nid: Node(nid, node.sequence, node.is_hypothetical)
            for nid, node in self.nodes.items()
        }
        new_parent = dict(self.parent_map)
        new_children = {nid: list(children) for nid, children in self.children_map.items()}
        return Tree(new_nodes, new_parent, new_children)

    def get_root(self) -> Optional[int]:
        """
        Gibt die node_id der Wurzel zurück (parent_map[node_id] is None).
        Wenn mehrere Wurzeln existieren, wird die erste zurückgegeben.
        """
        roots = [nid for nid, pid in self.parent_map.items() if pid is None]
        return roots[0] if roots else None

    def all_node_ids(self) -> List[int]:
        """Gibt alle node_ids in diesem Baum zurück."""
        return list(self.nodes.keys())

    def is_leaf(self, node_id: int) -> bool:
        """Prüft, ob ein Knoten ein Blatt (Leaf) ist."""
        return len(self.children_map.get(node_id, [])) == 0

    def as_edge_list(self) -> List[Tuple[int, int]]:
        """
        Gibt alle Kanten (parent_id, child_id) als Liste von Tupeln zurück.
        """
        edges: List[Tuple[int, int]] = []
        for child, parent in self.parent_map.items():
            if parent is not None:
                edges.append((parent, child))
        return edges

    def has_any_invalid_edge(self) -> bool:
        """
        Prüft, ob es irgendeine Kante (p, c) gibt, bei der die Mutations-Constraint verletzt wird.
        """
        for parent_id, child_id in self.as_edge_list():
            seq_p = self.nodes[parent_id].sequence
            seq_c = self.nodes[child_id].sequence
            if not is_valid_mutation(seq_p, seq_c):
                return True
        return False

    def is_valid_structure(self) -> bool:
        """
        Prüft, ob der Baum eine gültige Struktur ohne Zyklen bildet (zusammenhängend).
        Hier wird nur auf Zyklusfreiheit geachtet, nicht auf Mutations-Constraints.
        """
        visited: Set[int] = set()
        root = self.get_root()
        if root is None:
            return False

        stack = [root]
        while stack:
            current = stack.pop()
            if current in visited:
                return False  # Zyklus
            visited.add(current)
            for child in self.children_map.get(current, []):
                stack.append(child)

        return visited == set(self.nodes.keys())


# ---------------------------
#  Hilfsfunktionen
# ---------------------------

def load_json_nodes(filepath: str) -> List[Dict]:
    """
    Liest die JSON-Datei ein und gibt die Liste der Knoten-Dictionaries zurück.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("nodes", [])


def filter_alive_nodes(raw_nodes: List[Dict]) -> Dict[int, str]:
    """
    Filtert aus den rohen JSON-Knoten diejenigen heraus, die alive=true sind.
    Gibt ein Dict zurück: node_id -> sequence.
    """
    alive: Dict[int, str] = {}
    for entry in raw_nodes:
        if entry.get("alive", False):
            nid = entry["id"]
            seq = entry["sequence"]
            alive[nid] = seq
    return alive


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Berechnet die Levenshtein-Distanz zwischen zwei Strings s1 und s2.
    Laufzeit: O(len(s1)*len(s2))
    """
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[len1][len2]


def is_valid_mutation(parent_seq: str, child_seq: str) -> bool:
    """
    Prüft, ob child_seq durch inkrementelle Buchstabenänderungen
    aus parent_seq hervorgegangen sein kann (Buchstabe im Alphabet nur vorwärts).
    Für alle Positionen i muss gelten: ord(parent_seq[i]) <= ord(child_seq[i]).
    """
    for c_p, c_c in zip(parent_seq, child_seq):
        if ord(c_c) < ord(c_p):
            return False
    return True


# ---------------------------
#  Initiale Baum-Erzeugung
# ---------------------------

def generate_random_tree(alive_sequences: Dict[int, str]) -> Tree:
    """
    Erzeugt zufällig einen Baum aus den lebendigen Knoten (ohne hypothetische Knoten),
    wobei jeder Knoten höchstens zwei Kinder hat und es genau eine Wurzel gibt.
    """
    real_ids = list(alive_sequences.keys())
    random.shuffle(real_ids)

    nodes: Dict[int, Node] = {}
    parent_map: Dict[int, Optional[int]] = {}
    children_map: Dict[int, List[int]] = {}

    for rid in real_ids:
        nodes[rid] = Node(rid, alive_sequences[rid], is_hypothetical=False)
        children_map[rid] = []

    root = real_ids[0]
    parent_map[root] = None
    eligible_parents: Set[int] = {root}

    for child_id in real_ids[1:]:
        parent_id = random.choice(list(eligible_parents))
        parent_map[child_id] = parent_id
        children_map[parent_id].append(child_id)
        children_map[child_id] = []
        if len(children_map[parent_id]) >= 2:
            eligible_parents.remove(parent_id)
        eligible_parents.add(child_id)

    return Tree(nodes, parent_map, children_map)


# ---------------------------
#  Fitness-Funktion
# ---------------------------

def compute_tree_fitness(tree: Tree, penalty_invalid: float = 1000.0) -> float:
    """
    Berechnet die Fitness eines Baumes. Je kleiner, desto besser.
    - Für jede Kante (p, c):
      * Wenn is_valid_mutation(seq_p, seq_c) == False, füge penalty_invalid hinzu.
      * Sonst summiere die Levenshtein-Distanz.
    """
    total = 0.0
    for parent_id, child_id in tree.as_edge_list():
        parent_seq = tree.nodes[parent_id].sequence
        child_seq = tree.nodes[child_id].sequence
        if not is_valid_mutation(parent_seq, child_seq):
            total += penalty_invalid
        else:
            total += levenshtein_distance(parent_seq, child_seq)
    return total


# ---------------------------
#  Helper: Ungültige Kanten suchen (Leaf-first)
# ---------------------------

def find_invalid_edge_leaf_first(tree: Tree) -> Optional[Tuple[int, int]]:
    """
    Zuerst werden Blätter (Leaves) betrachtet:
    Für jedes Blatt (node_id ohne Kinder) prüfen wir, ob die Kante (p, leaf) invalid ist.
    Wenn gefunden, zurückgeben (parent_id, leaf_id). 
    Wenn in allen Blättern kein Fehler vorliegt, prüfen wir interne Knoten (in beliebiger Reihenfolge)
    und geben die erste invalid Kante zurück.
    """
    # 1) Blätter prüfen
    for node_id in tree.all_node_ids():
        if tree.is_leaf(node_id):
            parent_id = tree.parent_map.get(node_id)
            if parent_id is None:
                continue
            seq_p = tree.nodes[parent_id].sequence
            seq_c = tree.nodes[node_id].sequence
            if not is_valid_mutation(seq_p, seq_c):
                return parent_id, node_id

    # 2) Alle anderen Kanten prüfen
    for parent_id, child_id in tree.as_edge_list():
        if tree.is_leaf(child_id):
            continue  # Blatt-Kanten bereits geprüft
        seq_p = tree.nodes[parent_id].sequence
        seq_c = tree.nodes[child_id].sequence
        if not is_valid_mutation(seq_p, seq_c):
            return parent_id, child_id

    return None


# ---------------------------
#  Reattach mit Root-Handling
# ---------------------------

def reattach_leaf(tree: Tree, parent_id: int, child_id: int, next_hypo_id: int) -> Tuple[Tree, int]:
    """
    Versucht, das Blatt child_id (c) von parent_id (p) zu lösen:
    1. c als Kind eines Schwesterknotens von p anhängen, falls möglich.
    2. Falls p ein direkter Kind des Root-Knotens war (p hatte parent=None),
       wird ein neuer hypothetischer Root angelegt, der beide als Kinder hat.
    3. Andernfalls c als Schwester von p ansetzen (parent[c] = gp).
    """
    new_tree = tree.copy()
    p = parent_id
    c = child_id
    gp = new_tree.parent_map.get(p)  # Großelternteil

    # 1) Versuche, c an einen Schwesterknoten von p anzuhängen
    if gp is not None:
        siblings = [s for s in new_tree.children_map.get(gp, []) if s != p]
        for s in siblings:
            seq_s = new_tree.nodes[s].sequence
            seq_c = new_tree.nodes[c].sequence
            if is_valid_mutation(seq_s, seq_c):
                new_tree.children_map[p].remove(c)
                new_tree.parent_map[c] = s
                new_tree.children_map[s].append(c)
                return new_tree, next_hypo_id

    # 2) Wenn p direkt unter Root war (gp ist None), neuen hypothetischen Root erzeugen
    if gp is None:
        old_root = p
        seq_old = new_tree.nodes[old_root].sequence
        seq_c = new_tree.nodes[c].sequence

        # Erzeuge hypothetischen Root h
        h_id = next_hypo_id
        next_hypo_id -= 1
        hypo_seq = "".join(chr(min(ord(ch1), ord(ch2))) for ch1, ch2 in zip(seq_old, seq_c))
        new_tree.nodes[h_id] = Node(h_id, hypo_seq, is_hypothetical=True)
        new_tree.children_map[h_id] = []

        # Entferne c von p
        new_tree.children_map[p].remove(c)

        # Setze h als neuen Root (parent=None)
        new_tree.parent_map[h_id] = None

        # Altem Root und c als Kinder von h zuweisen
        new_tree.parent_map[old_root] = h_id
        new_tree.parent_map[c] = h_id
        new_tree.children_map[h_id].extend([old_root, c])

        # Ersetze old_root in children_map (früher parent_map=None) = h
        # Kein weiterer Schritt nötig, da old_root und c nun Kinder von h sind

        return new_tree, next_hypo_id

    # 3) Ansonsten c als Schwester von p ansetzen (parent[c] = gp)
    new_tree.children_map[p].remove(c)
    new_tree.parent_map[c] = gp
    new_tree.children_map[gp].append(c)
    return new_tree, next_hypo_id


# ---------------------------
#  Swap-Funktion (Parent <-> Child)
# ---------------------------

def swap_parent_child(tree: Tree, parent_id: int, child_id: int) -> Tree:
    """
    Tauscht gezielt die Rollen von parent_id und child_id im Baum,
    sodass child_id zum Parent von parent_id wird.
    Prüft nur Zyklusfreiheit; Mutations-Constraints müssen danach validiert werden.
    """
    new_tree = tree.copy()

    def is_descendant(start: int, target: int) -> bool:
        stack = [start]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            seen.add(cur)
            for ch in new_tree.children_map.get(cur, []):
                if ch not in seen:
                    stack.append(ch)
        return False

    if is_descendant(child_id, parent_id):
        return new_tree  # Kein Tausch, Zyklus

    old_parent_of_p = new_tree.parent_map[parent_id]
    old_children_of_c = list(new_tree.children_map[child_id])

    # 1) Entferne parent_id aus children von old_parent_of_p
    if old_parent_of_p is not None:
        new_tree.children_map[old_parent_of_p].remove(parent_id)

    # 2) Setze child_id an Stelle von parent_id
    new_tree.parent_map[child_id] = old_parent_of_p
    if old_parent_of_p is not None:
        new_tree.children_map[old_parent_of_p].append(child_id)

    # 3) parent_id wird Kind von child_id
    new_tree.parent_map[parent_id] = child_id

    # 4) Hänge alte Kinder von child_id (außer parent_id) an parent_id
    for ch in old_children_of_c:
        if ch != parent_id:
            new_tree.parent_map[ch] = parent_id
            new_tree.children_map[parent_id].append(ch)

    # 5) Setze children_map[child_id] = [parent_id]
    new_tree.children_map[child_id] = [parent_id]

    return new_tree


# ---------------------------
#  Overflow und Hypothetik behandeln
# ---------------------------

def find_node_with_overflow(tree: Tree) -> Optional[int]:
    """
    Sucht im Baum einen Knoten, der mehr als zwei Kinder hat.
    Gibt die node_id zurück oder None, falls kein Overflow existiert.
    """
    for nid, children in tree.children_map.items():
        if len(children) > 2:
            return nid
    return None


def add_hypothetical_for_overflow(tree: Tree, next_hypo_id: int) -> Tuple[Tree, int]:
    """
    Fügt zwischen zwei Kindern eines überfüllten Knotens einen hypothetischen Knoten ein,
    der die beiden Kinder vereint, deren Sequenzen die geringste Levenshtein-Distanz aufweisen.
    Dadurch wird die Anzahl der direkten Kinder des Overflow-Knotens um 1 verringert.
    """
    new_tree = tree.copy()
    overflow_node = find_node_with_overflow(new_tree)
    if overflow_node is None:
        return new_tree, next_hypo_id

    children = new_tree.children_map[overflow_node]
    # Wähle zwei Kinder mit minimaler Distanz
    best_pair: Tuple[int, int] = (children[0], children[1])
    best_dist = levenshtein_distance(
        new_tree.nodes[children[0]].sequence,
        new_tree.nodes[children[1]].sequence
    )
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            cid1, cid2 = children[i], children[j]
            dist = levenshtein_distance(
                new_tree.nodes[cid1].sequence,
                new_tree.nodes[cid2].sequence
            )
            if dist < best_dist:
                best_dist = dist
                best_pair = (cid1, cid2)

    c1, c2 = best_pair
    seq_c1 = new_tree.nodes[c1].sequence
    seq_c2 = new_tree.nodes[c2].sequence
    hypo_seq = "".join(chr(min(ord(a), ord(b))) for a, b in zip(seq_c1, seq_c2))

    h_id = next_hypo_id
    next_hypo_id -= 1
    new_tree.nodes[h_id] = Node(h_id, hypo_seq, is_hypothetical=True)
    new_tree.children_map[h_id] = []

    new_tree.children_map[overflow_node] = [
        cid for cid in new_tree.children_map[overflow_node] if cid not in (c1, c2)
    ]
    new_tree.children_map[overflow_node].append(h_id)
    new_tree.parent_map[h_id] = overflow_node

    new_tree.children_map[h_id] = [c1, c2]
    new_tree.parent_map[c1] = h_id
    new_tree.parent_map[c2] = h_id

    return new_tree, next_hypo_id


def remove_redundant_hypothetical(tree: Tree) -> Tree:
    """
    Sucht einen hypothetischen Knoten, dessen Entfernen keine Mutations-Constraints verletzt.
    Falls gefunden, wird dieser Knoten entfernt (Bypass), und sein Kind wird direkt an seinen Elternteil gehängt.
    """
    new_tree = tree.copy()
    for nid, node in list(new_tree.nodes.items()):
        if not node.is_hypothetical:
            continue
        children = new_tree.children_map.get(nid, [])
        if len(children) != 1:
            continue
        child = children[0]
        parent = new_tree.parent_map[nid]
        if parent is not None:
            seq_parent = new_tree.nodes[parent].sequence
        else:
            seq_parent = None
        seq_child = new_tree.nodes[child].sequence

        if parent is None or is_valid_mutation(seq_parent, seq_child):
            if parent is not None:
                new_tree.children_map[parent].remove(nid)
                new_tree.children_map[parent].append(child)
                new_tree.parent_map[child] = parent
            else:
                new_tree.parent_map[child] = None
            del new_tree.nodes[nid]
            del new_tree.parent_map[nid]
            del new_tree.children_map[nid]
            return new_tree

    return new_tree


# ---------------------------
#  Geführte Mutation (Leaf-first mit Swap und Root-Handling)
# ---------------------------

def guided_mutation(tree: Tree, next_hypo_id: int) -> Tuple[Tree, int]:
    """
    Führt eine geführte Mutation aus in dieser Reihenfolge:
    1. Ungültige Kante finden (Leaf-first).
       a) Versuche einen direkten Swap (p, c).
          - Wenn nach Swap der Baum valide ist (keine invaliden Edges),
            dann übernehme den Swap.
       b) Andernfalls: reattach_leaf (eventuell mit neuem Root).
    2. Wenn keine invaliden Kanten, stelle Overflow fest und füge hypothetischen Knoten ein.
    3. Wenn weder Invalidität noch Overflow vorhanden, entferne redundante hypothetische Knoten.
    """
    invalid_edge = find_invalid_edge_leaf_first(tree)
    if invalid_edge is not None:
        p, c = invalid_edge
        # 1a) Direktes Tausch versuchen
        swapped = swap_parent_child(tree, p, c)
        if swapped is not tree and not swapped.has_any_invalid_edge():
            return swapped, next_hypo_id
        # 1b) Reattach (inklusive Root-Handling)
        return reattach_leaf(tree, p, c, next_hypo_id)

    # 2) Overflow behandeln
    overflow_node = find_node_with_overflow(tree)
    if overflow_node is not None:
        return add_hypothetical_for_overflow(tree, next_hypo_id)

    # 3) Hypothetischen Knoten entfernen
    new_tree = remove_redundant_hypothetical(tree)
    return new_tree, next_hypo_id


# ---------------------------
#  Evolutionäre Strategie (Hauptschleife)
# ---------------------------

def evolutionary_algorithm(
    alive_sequences: Dict[int, str],
    population_size: int = 50,
    generations: int = 100,
    remove_fraction: float = 0.2,
    seed: Optional[int] = None
) -> Tree:
    """
    Führt die Evolutionäre Strategie aus und gibt den besten Baum zurück.
    :param alive_sequences: Dict {node_id: sequence} für lebendige Knoten.
    :param population_size: Anzahl der Bäume in der Population.
    :param generations: Maximale Anzahl von Generationen.
    :param remove_fraction: Anteil (0..1) der schlechtesten Bäume, die pro Generation entfernt werden.
    :param seed: Optionaler Zufallsseed für Reproduzierbarkeit.
    :return: Der Baum mit der besten Fitness am Ende.
    """
    if seed is not None:
        random.seed(seed)

    next_hypo_id = -1  # Start-ID für hypothetische Knoten

    # 1) Start-Population erzeugen
    population: List[Tree] = []
    for _ in range(population_size):
        tree = generate_random_tree(alive_sequences)
        population.append(tree)

    # 2) Evolutionäre Schleife
    for gen in range(1, generations + 1):
        scored_population: List[Tuple[Tree, float]] = [
            (tree, compute_tree_fitness(tree)) for tree in population
        ]
        scored_population.sort(key=lambda x: x[1])

        best_fit = scored_population[0][1]
        print(f"Generation {gen:3d} | Beste Fitness: {best_fit:.2f}")

        keep_count = max(2, int(population_size * (1.0 - remove_fraction)))
        survivors = [tree for tree, _ in scored_population[:keep_count]]

        new_population: List[Tree] = survivors.copy()
        while len(new_population) < population_size:
            parent_tree = random.choice(survivors)
            child_tree, next_hypo_id = guided_mutation(parent_tree, next_hypo_id)
            new_population.append(child_tree)

        population = new_population

    final_scored = [(tree, compute_tree_fitness(tree)) for tree in population]
    final_scored.sort(key=lambda x: x[1])
    best_tree = final_scored[0][0]
    print(f"Evolution abgeschlossen. Beste finale Fitness: {final_scored[0][1]:.2f}")
    return best_tree


# ---------------------------
#  Graphviz-Visualisierung
# ---------------------------

def visualize_tree(tree: Tree, output_basename: str) -> None:
    """
    Visualisiert den Baum mit Graphviz. Hypothetische Knoten werden durch einen
    vorangestellten Asterisk (*) im Label markiert.
    Speichert als DOT- und PDF-Datei: {output_basename}.gv und {output_basename}.pdf
    """
    dot = graphviz.Digraph(comment="Rekonstruierter Baum", format="pdf")
    dot.attr("node", shape="ellipse")

    for nid, node in tree.nodes.items():
        prefix = "*" if node.is_hypothetical else ""
        label = f"{prefix}{nid}: {node.sequence}"
        dot.node(str(nid), label=label)

    for parent_id, child_id in tree.as_edge_list():
        dot.edge(str(parent_id), str(child_id))

    dot_filename = f"{output_basename}.gv"
    dot.render(dot_filename, cleanup=True)
    print(f"Graphviz-Dateien erzeugt: '{dot_filename}' und '{output_basename}.pdf'")


def print_tree(tree: Tree) -> None:
    """
    Gibt den Baum in Textform aus: node_id (Seq) → parent, children. Hypothetische Knoten markiert.
    """
    print("Baum-Struktur (node_id: sequence) → [children_ids]")
    for nid, node in tree.nodes.items():
        parent = tree.parent_map.get(nid)
        children = tree.children_map.get(nid, [])
        hypo_flag = "(HYP)" if node.is_hypothetical else ""
        print(f"  {nid}{hypo_flag}: {node.sequence}   parent={parent}   children={children}")


# ---------------------------
#  Main-Funktion
# ---------------------------

def main(config: Config) -> None:
    # 1) JSON einlesen
    raw_nodes = load_json_nodes(config.input_path)
    alive_sequences = filter_alive_nodes(raw_nodes)
    if not alive_sequences:
        print("Keine lebendigen Knoten in der Datei gefunden. Beende.")
        return

    print(f"Anzahl lebendiger Knoten: {len(alive_sequences)}")
    print("Starte evolutionären Algorithmus...")

    # 2) Evolutionäre Strategie ausführen
    best_tree = evolutionary_algorithm(
        alive_sequences=alive_sequences,
        population_size=config.population_size,
        generations=config.generations,
        remove_fraction=config.remove_fraction,
        seed=config.seed
    )

    # 3) Ergebnis textuell ausgeben
    print("\n=== Bester rekonstruierter Baum (Text) ===")
    print_tree(best_tree)

    # 4) Ergebnis mit Graphviz visualisieren
    visualize_tree(best_tree, config.output_graph_basename)


if __name__ == "__main__":
    cfg = Config(
        input_path="output//mutation_tree.json",
        population_size=50,
        generations=100,
        remove_fraction=0.2,
        seed=42,
        output_graph_basename="best_tree"
    )
    main(cfg)
