"""
tree_reconstruction_single.py

Dieses Python-Programm lädt eine JSON-Datei, die einen Mutation-Tree beschreibt, und
rekonstruiert durch gezielte, iterative Transformationen aus einem einzelnen zufälligen
Startbaum einen plausiblen Baum für die lebendigen Knoten (alive=true). 

Die Hauptidee:
1. Erzeuge einmalig einen zufälligen binären Startbaum aus allen lebendigen Knoten.
2. Führe in einer Schleife folgende Schritte durch, solange sich der Baum verändert:
   a) Suche eine ungültige Kante (Leaf-first). 
      • Versuche einen Parent-Child-Swap, der danach keine Invaliditätsverletzungen mehr hat.
      • Falls Swap nicht möglich oder unzureichend, führe einen „Reattach“ des fehlerhaften Blatts durch.
   b) Falls keine Invaliditäten mehr vorliegen, prüfe auf Overflow (Knoten mit >2 Kindern):
      • Wähle zwei Kinder mit minimaler Levenshtein-Distanz. 
      • **Bevor** du einen hypothetischen Knoten einfügst, prüfe, ob einer der beiden direkt von
        dem anderen abgeleitet werden kann (is_valid_mutation). In diesem Fall häng diesen einfach
        als Kind unter den anderen. Andernfalls füge einen hypothetischen Knoten ein.
   c) Falls weder Invalidität noch Overflow mehr existiert, versuche, einen redundanten
      hypothetischen Knoten zu entfernen (Bypass).
3. Sobald keine der drei Operationen (a–c) mehr eine Veränderung bewirkt, ist der Baum
   stabil und wird als Endergebnis ausgegeben.

Grafische Ausgabe: Am Ende wird der finale Baum mit Graphviz visualisiert. 
Hypothetische Knoten werden im Label durch einen vorangestellten Asterisk (*) markiert.

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
    Konfigurationsparameter für die Baumrekonstruktion.
    """
    input_path: str = "mutation_tree.json"
    seed: Optional[int] = None
    output_graph_basename: str = "reconstructed_tree"


# ---------------------------
#  Datenstrukturen
# ---------------------------

class Node:
    """
    Repräsentiert einen Knoten im Baum.
    Kann ein originaler (lebendiger) Knoten aus der JSON-Datei sein
    oder ein hypothetischer Knoten, der während der Restrukturierung hinzugefügt wird.
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
        :param nodes: Dict von node_id → Node-Instanz (inkl. hypothetischer Knoten)
        :param parent_map: Dict von node_id → parent_id oder None (wenn Wurzel)
        :param children_map: Dict von node_id → Liste der children_ids
        """
        self.nodes: Dict[int, Node] = nodes
        self.parent_map: Dict[int, Optional[int]] = parent_map
        self.children_map: Dict[int, List[int]] = children_map

    def copy(self) -> "Tree":
        """
        Erzeugt eine tiefe Kopie des Baumes (keine Seiteneffekte).
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
        """
        for nid, pid in self.parent_map.items():
            if pid is None:
                return nid
        return None

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
        (Nur Struktur, nicht Mutations-Constraints.)
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
    Liefert ein Dict: node_id → sequence.
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
    Laufzeit: O(len(s1) * len(s2))
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
                dp[i - 1][j] + 1,       # Löschung
                dp[i][j - 1] + 1,       # Einfügung
                dp[i - 1][j - 1] + cost # Ersetzung
            )
    return dp[len1][len2]


def is_valid_mutation(parent_seq: str, child_seq: str) -> bool:
    """
    Prüft, ob child_seq durch inkrementelle Buchstabenänderungen
    aus parent_seq hervorgegangen sein kann (Buchstabe nur vorwärts im Alphabet).
    """
    for c_p, c_c in zip(parent_seq, child_seq):
        if ord(c_c) < ord(c_p):
            return False
    return True


# ---------------------------
#  Initialer Zufallsbaum
# ---------------------------

def generate_random_tree(alive_sequences: Dict[int, str]) -> Tree:
    """
    Erzeugt zufällig einen Startbaum aus allen lebendigen Knoten,
    wobei jeder Knoten maximal zwei Kinder hat und es genau eine Wurzel gibt.
    """
    real_ids = list(alive_sequences.keys())
    random.shuffle(real_ids)

    nodes: Dict[int, Node] = {}
    parent_map: Dict[int, Optional[int]] = {}
    children_map: Dict[int, List[int]] = {}

    for rid in real_ids:
        nodes[rid] = Node(rid, alive_sequences[rid], is_hypothetical=False)
        children_map[rid] = []

    # Der erste Knoten wird Wurzel
    root = real_ids[0]
    parent_map[root] = None
    eligible_parents: Set[int] = {root}

    # Weise jeden weiteren Knoten zufällig einem Parent mit <2 Kindern zu
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
#  Suche ungültige Kante (Leaf-first)
# ---------------------------

def find_invalid_edge_leaf_first(tree: Tree) -> Optional[Tuple[int, int]]:
    """
    Zuerst Blätter prüfen: Wenn (p, leaf) invalid ist, zurückgeben.
    Sonst interne Knoten (erste gefundene invalid Kante).
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

    # 2) Andere Kanten prüfen
    for parent_id, child_id in tree.as_edge_list():
        if tree.is_leaf(child_id):
            continue
        seq_p = tree.nodes[parent_id].sequence
        seq_c = tree.nodes[child_id].sequence
        if not is_valid_mutation(seq_p, seq_c):
            return parent_id, child_id

    return None


# ---------------------------
#  Reattach mit Root-Erweiterung
# ---------------------------

def reattach_leaf(tree: Tree, parent_id: int, child_id: int, next_hypo_id: int) -> Tuple[Tree, int]:
    """
    Löst ein fehlerhaftes Blatt child_id von parent_id:
    1. Versucht, child_id an einen validen Schwesterknoten von parent_id anzuhängen.
    2. Falls parent_id direkt unter Root steht (gp=None), wird ein neuer hypothetischer 
       Root erzeugt, der parent_id und child_id vereint.
    3. Sonst: Hänge child_id als Schwester von parent_id unter dessen Großeltern.
    """
    new_tree = tree.copy()
    p = parent_id
    c = child_id
    gp = new_tree.parent_map.get(p)

    # 1) Versuche, c an einen validen Schwesterknoten von p anzuhängen
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

    # 2) Falls p direkt unter Root steht (gp=None), neuen hypothetischen Root anlegen
    if gp is None:
        old_root = p
        seq_old = new_tree.nodes[old_root].sequence
        seq_c = new_tree.nodes[c].sequence

        # Hypothetischen Root h erstellen
        h_id = next_hypo_id
        next_hypo_id -= 1
        hypo_seq = "".join(
            chr(min(ord(ch1), ord(ch2))) for ch1, ch2 in zip(seq_old, seq_c)
        )
        new_tree.nodes[h_id] = Node(h_id, hypo_seq, is_hypothetical=True)
        new_tree.children_map[h_id] = []

        # Entferne c von p
        new_tree.children_map[p].remove(c)

        # h wird neue Wurzel (parent=None)
        new_tree.parent_map[h_id] = None

        # Assigniere old_root und c als Kinder von h
        new_tree.parent_map[old_root] = h_id
        new_tree.parent_map[c] = h_id
        new_tree.children_map[h_id].extend([old_root, c])

        return new_tree, next_hypo_id

    # 3) Ansonsten c als Schwester von p unter gp ansetzen
    new_tree.children_map[p].remove(c)
    new_tree.parent_map[c] = gp
    new_tree.children_map[gp].append(c)
    return new_tree, next_hypo_id


# ---------------------------
#  Swap-Funktion (Parent ↔ Child)
# ---------------------------

def swap_parent_child(tree: Tree, parent_id: int, child_id: int) -> Tree:
    """
    Tauscht die Rollen von parent_id und child_id:
    child_id wird Parent von parent_id; Struktur wird nur abgebrochen, wenn ein Zyklus entstünde.
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

    # Zyklus vermeiden
    if is_descendant(child_id, parent_id):
        return new_tree

    old_parent = new_tree.parent_map[parent_id]
    old_children = list(new_tree.children_map.get(child_id, []))

    # 1) Entferne parent_id aus den Kindern von old_parent
    if old_parent is not None:
        new_tree.children_map[old_parent].remove(parent_id)

    # 2) Setze child_id an Stelle von parent_id
    new_tree.parent_map[child_id] = old_parent
    if old_parent is not None:
        new_tree.children_map[old_parent].append(child_id)

    # 3) parent_id wird Kind von child_id
    new_tree.parent_map[parent_id] = child_id

    # 4) Hänge alte Kinder von child_id (außer parent_id) an parent_id
    for ch in old_children:
        if ch != parent_id:
            new_tree.parent_map[ch] = parent_id
            new_tree.children_map[parent_id].append(ch)

    # 5) Setze Kinderliste von child_id neu auf [parent_id]
    new_tree.children_map[child_id] = [parent_id]

    return new_tree


# ---------------------------
#  Overflow und Hypothetik
# ---------------------------

def find_node_with_overflow(tree: Tree) -> Optional[int]:
    """
    Gibt die node_id eines Knotens mit >2 Kindern zurück (Overflow), oder None, falls keiner existiert.
    """
    for nid, children in tree.children_map.items():
        if len(children) > 2:
            return nid
    return None


def add_hypothetical_for_overflow(tree: Tree, next_hypo_id: int) -> Tuple[Tree, int]:
    """
    Fügt zwischen zwei der Kinder eines überfüllten Knotens einen hypothetischen Knoten ein,
    oder hängt einen Knoten direkt unter den anderen, falls das valide wäre.
    """
    new_tree = tree.copy()
    overflow_node = find_node_with_overflow(new_tree)
    if overflow_node is None:
        return new_tree, next_hypo_id

    children = new_tree.children_map[overflow_node]
    # Wähle das Paar mit geringster Levenshtein-Distanz
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

    # 1) Prüfen, ob c2 direktes Kind von c1 sein kann
    if is_valid_mutation(seq_c1, seq_c2):
        # Entferne c2 aus Kindern von overflow_node und hänge unter c1
        new_tree.children_map[overflow_node].remove(c2)
        new_tree.parent_map[c2] = c1
        new_tree.children_map[c1].append(c2)
        return new_tree, next_hypo_id

    # 2) Prüfen, ob c1 direktes Kind von c2 sein kann
    if is_valid_mutation(seq_c2, seq_c1):
        new_tree.children_map[overflow_node].remove(c1)
        new_tree.parent_map[c1] = c2
        new_tree.children_map[c2].append(c1)
        return new_tree, next_hypo_id

    # 3) Sonst: Erzeuge hypothetischen Knoten h
    hypo_seq = "".join(chr(min(ord(a), ord(b))) for a, b in zip(seq_c1, seq_c2))
    h_id = next_hypo_id
    next_hypo_id -= 1
    new_tree.nodes[h_id] = Node(h_id, hypo_seq, is_hypothetical=True)
    new_tree.children_map[h_id] = []

    # Entferne c1, c2 aus Kindern von overflow_node
    new_tree.children_map[overflow_node] = [
        cid for cid in new_tree.children_map[overflow_node] if cid not in (c1, c2)
    ]
    # Hänge h an overflow_node
    new_tree.children_map[overflow_node].append(h_id)
    new_tree.parent_map[h_id] = overflow_node

    # Hänge c1, c2 unter h
    new_tree.children_map[h_id] = [c1, c2]
    new_tree.parent_map[c1] = h_id
    new_tree.parent_map[c2] = h_id

    return new_tree, next_hypo_id


def remove_redundant_hypothetical(tree: Tree) -> Tree:
    """
    Entfernt, falls möglich, einen hypothetischen Knoten, dessen Bypass keine Invalidität erzeugt.
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
            # Bypass
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
#  Einzelschritt-Mutation (Leaf-first mit Swap und Root-Handling)
# ---------------------------

def single_step_mutation(tree: Tree, next_hypo_id: int) -> Tuple[Tree, int, bool]:
    """
    Führt genau einen Schritt der Restrukturierung aus:
    1. Suche eine ungültige Kante (Leaf-first).
       a) Versuche einen Parent-Child-Swap; akzeptiere, falls nach Swap keine Invalidität mehr besteht.
       b) Sonst: Reattach (inklusive neuem Root, falls nötig).
       → Wenn in Schritt 1 eine Änderung erfolgte, kehre mit (neuer_tree, next_hypo_id, True) zurück.
    2. Wenn keine invaliden Kanten mehr existieren, prüfe auf Overflow:
       → Wenn Overflow existiert, füge hypothetischen Knoten ein oder hänge direkt unter den anderen Knoten.
         und kehre mit (neuer_tree, next_hypo_id, True) zurück.
    3. Sonst versuche, einen redundanten hypothetischen Knoten zu entfernen:
       → Wenn entfernt wurde, kehre mit (neuer_tree, next_hypo_id, True) zurück.
    4. Wenn keine der Operationen mehr anwendbar ist, kehre mit (tree, next_hypo_id, False) zurück.
    """
    # 1) Ungültige Kante finden
    invalid_edge = find_invalid_edge_leaf_first(tree)
    if invalid_edge is not None:
        p, c = invalid_edge
        # 1a) Direktes Swap testen
        swapped = swap_parent_child(tree, p, c)
        if swapped is not tree and not swapped.has_any_invalid_edge():
            return swapped, next_hypo_id, True
        # 1b) Reattach (inklusive Root-Erweiterung)
        new_tree, new_next = reattach_leaf(tree, p, c, next_hypo_id)
        return new_tree, new_next, True

    # 2) Overflow prüfen
    overflow_node = find_node_with_overflow(tree)
    if overflow_node is not None:
        new_tree, new_next = add_hypothetical_for_overflow(tree, next_hypo_id)
        return new_tree, new_next, True

    # 3) Redundanten hypothetischen Knoten entfernen
    new_tree = remove_redundant_hypothetical(tree)
    if len(new_tree.nodes) < len(tree.nodes):
        return new_tree, next_hypo_id, True

    # 4) Keine Änderung mehr möglich
    return tree, next_hypo_id, False


# ---------------------------
#  Einzelbaum-Iterativer Algorithmus
# ---------------------------

def reconstruct_single_tree(
    alive_sequences: Dict[int, str],
    seed: Optional[int] = None
) -> Tree:
    """
    Startet mit einem einzelnen zufälligen Baum und wendet wiederholt single_step_mutation
    an, bis keine Veränderung mehr möglich ist. Gibt den finale stabile Baum zurück.
    """
    if seed is not None:
        random.seed(seed)

    # Nächstverfügbare ID für hypothetische Knoten (negativ)
    next_hypo_id = -1

    # 1) Initialer Zufallsbaum
    current_tree = generate_random_tree(alive_sequences)

    # 2) Iteriere, bis stabil
    while True:
        new_tree, next_hypo_id, changed = single_step_mutation(current_tree, next_hypo_id)
        if not changed:
            break
        current_tree = new_tree

    return current_tree


# ---------------------------
#  Graphviz-Visualisierung
# ---------------------------

def visualize_tree(tree: Tree, output_basename: str) -> None:
    """
    Visualisiert den Baum mit Graphviz. Hypothetische Knoten werden durch
    einen vorangestellten Asterisk (*) im Label markiert.
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
    Gibt den Baum in Textform aus: node_id (Sequence) → parent, children.
    Hypothetische Knoten werden markiert.
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
        print("Keine lebendigen Knoten gefunden. Abbruch.")
        return

    print(f"Anzahl lebendiger Knoten: {len(alive_sequences)}")
    print("Starte iterative Baumrekonstruktion...")

    # 2) Baum rekonstruieren
    best_tree = reconstruct_single_tree(
        alive_sequences=alive_sequences,
        seed=config.seed
    )

    # 3) Ergebnis textuell ausgeben
    print("\n=== Rekonstruierter Baum (Text) ===")
    print_tree(best_tree)

    # 4) Ergebnis mit Graphviz visualisieren
    visualize_tree(best_tree, config.output_graph_basename)


if __name__ == "__main__":
    cfg = Config(
        input_path="output//mutation_tree.json",
        seed=42,
        output_graph_basename="reconstructed_tree"
    )
    main(cfg)
