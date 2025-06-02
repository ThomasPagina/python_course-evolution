#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modul: mutations_tree.py

Dieses Modul simuliert über mehrere Generationen hinweg die Kopie eines Ausgangsstrings,
fügt mit einer festgelegten Fehlerrate Mutationen hinzu und speichert die
Abstammungslinien in einem Tree. Außerdem kann der Tree über Graphviz visualisiert
und in eine JSON-Datei gespeichert bzw. daraus geladen werden.

Beispielhafter Ablauf:
1. Erzeugen eines Start-Strings bestehend aus 'a'-Zeichen.
2. Für eine bestimmte Anzahl von Generationen:
   a. Jeder Eltern-String erzeugt eine festgelegte Anzahl von Kopien.
   b. In jeder Kopie werden mit gegebener Fehlerrate zufällige Buchstaben
      durch den nächsten Buchstaben im Alphabet ersetzt.
   c. Die neu entstandenen Kopien werden in der nächsten Generation wieder zu Eltern.
3. Der gesamte Stammbaum der Strings wird in einem Tree gespeichert.
4. Der Baum kann als JSON abgespeichert und später wieder geladen werden.
5. Der Baum lässt sich mit Graphviz visualisieren (als PDF, PNG, SVG, o.ä.).
"""

import json
import random
import string
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import graphviz  # pip install graphviz


@dataclass
class Config:
    """
    Konfigurationsparameter für die Simulation.
    """
    initial_length: int
    generations: int
    copies_per_parent: int
    error_rate: float  # z.B. 0.1 für 10% Fehlerrate pro Kopie


@dataclass
class TreeNode:
    """
    Repräsentiert einen Knoten im Abstammungsbaum.
    Jeder Knoten besitzt eine eindeutige ID, den erzeugten DNA-ähnlichen String
    und eine Liste von IDs seiner Kinder.
    """
    id: int
    sequence: str
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)


class MutationTree:
    """
    Die Klasse MutationTree verwaltet den gesamten Baum der kopierten Strings.
    Sie bietet Methoden zum Hinzufügen von Knoten, Speichern/Laden und Visualisieren.
    """

    def __init__(self) -> None:
        # Dict: node_id -> TreeNode
        self._nodes: Dict[int, TreeNode] = {}
        self._next_id: int = 0

    def add_root(self, sequence: str) -> int:
        """
        Fügt den Startknoten (Wurzel) mit einer gegebenen Sequenz hinzu.
        Gibt die ID des erstellten Knotens zurück.
        """
        node = TreeNode(id=self._next_id, sequence=sequence, parent_id=None)
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node.id

    def add_child(self, parent_id: int, sequence: str) -> int:
        """
        Erzeugt einen neuen Knoten als Kind des Knotens mit parent_id.
        Gibt die ID des neu erstellten Kindes zurück.
        """
        if parent_id not in self._nodes:
            raise ValueError(f"Parent-ID {parent_id} existiert nicht.")
        node = TreeNode(id=self._next_id, sequence=sequence, parent_id=parent_id)
        self._nodes[self._next_id] = node
        self._nodes[parent_id].children_ids.append(self._next_id)
        self._next_id += 1
        return node.id

    def get_node(self, node_id: int) -> TreeNode:
        """
        Liefert den TreeNode für die gegebene ID.
        """
        try:
            return self._nodes[node_id]
        except KeyError as e:
            raise ValueError(f"Node-ID {node_id} existiert nicht.") from e

    def to_dict(self) -> Dict:
        """
        Wandelt den gesamten Baum in ein serialisierbares Dict um.
        """
        return {
            "nodes": [
                {
                    "id": node.id,
                    "sequence": node.sequence,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                }
                for node in self._nodes.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MutationTree":
        """
        Erstellt einen MutationTree aus einem Dict (z.B. aus JSON).
        """
        tree = cls()
        # Temporär alle Knoten anlegen, um parent-child-Verlinkungen rekonstruieren zu können
        max_id = -1
        for node_data in data["nodes"]:
            node = TreeNode(
                id=node_data["id"],
                sequence=node_data["sequence"],
                parent_id=node_data["parent_id"],
                children_ids=list(node_data["children_ids"]),
            )
            tree._nodes[node.id] = node
            if node.id > max_id:
                max_id = node.id
        tree._next_id = max_id + 1
        return tree

    def save_to_json(self, filepath: Path) -> None:
        """
        Speichert den Baum als JSON-Datei.
        """
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "MutationTree":
        """
        Lädt den Baum aus einer JSON-Datei und gibt den geladenen MutationTree zurück.
        """
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def visualize(self, output_path: Path, fmt: str = "pdf") -> None:
        """
        Visualisiert den Baum mithilfe von Graphviz. Standardformat: PDF.
        Beispiel-Ausgabe: output_path mit Endung .pdf, .png, .svg etc.

        Jeder Knoten zeigt seine ID und Sequenz im Label.
        Kanten werden von Eltern zu Kind gezeichnet.
        """
        dot = graphviz.Digraph(comment="Mutationsbaum", format=fmt)

        # Knoten hinzufügen
        for node in self._nodes.values():
            label = f"{node.id}: {node.sequence}"
            dot.node(str(node.id), label)

        # Kanten hinzufügen
        for node in self._nodes.values():
            if node.parent_id is not None:
                dot.edge(str(node.parent_id), str(node.id))

        dot.render(str(output_path), view=False, cleanup=True)


def generate_initial_sequence(length: int) -> str:
    """
    Erzeugt einen String der Länge 'length', bestehend ausschließlich aus 'a'-Zeichen.
    """
    return "a" * length


def mutate_sequence(sequence: str, error_rate: float) -> str:
    """
    Generiert aus 'sequence' eine mutierte Kopie:
    - Mit Wahrscheinlichkeit 'error_rate' wird jeder Buchstabe mutiert.
    - Die Mutation ersetzt einen Buchstaben durch den nächsten im Alphabet.
      Z.B. 'a' -> 'b', 'b' -> 'c', ..., 'z' bleibt 'z'.

    Ist 'error_rate' 0.1, so wird jeder Buchstabe mit 10% Wahrscheinlichkeit
    durch den nächsten Buchstaben ersetzt.
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")

    sequence_list = list(sequence)
    for idx, char in enumerate(sequence_list):
        if random.random() < error_rate:
            # Aktuellen Buchstaben um eins im Alphabet verschieben
            if char.lower() == "z":
                # Bleibt 'z' liegen
                next_char = "z"
            else:
                # Nächstes Zeichen im Alphabet berechnen
                next_char = chr(ord(char.lower()) + 1)
            # Erhaltung der Groß-/Kleinschreibung (falls relevant)
            if char.isupper():
                next_char = next_char.upper()
            sequence_list[idx] = next_char

    return "".join(sequence_list)


def simulate(config: Config) -> MutationTree:
    """
    Führt die Simulation für die gegebene Konfiguration durch.
    - Zuerst wird der Start-String erzeugt.
    - Für jede Generation werden von allen aktuellen Eltern jeweils 'copies_per_parent' Kopien
      erzeugt und eventuelle Mutationen eingefügt.
    - Die neu entstandenen Kopien werden zu Eltern der nächsten Generation.
    - Die gesamte Abstammung wird in einem MutationTree gespeichert.
    """
    if config.generations < 1:
        raise ValueError("Anzahl der Generationen muss mindestens 1 sein.")
    if config.copies_per_parent < 1:
        raise ValueError("copies_per_parent muss mindestens 1 sein.")

    tree = MutationTree()
    # Erzeuge den Initial-String und füge ihn als Wurzel ein
    root_sequence = generate_initial_sequence(config.initial_length)
    root_id = tree.add_root(root_sequence)

    # Aktuelle Elter-Liste (IDs) initialisieren
    current_parents = [root_id]

    for gen in range(1, config.generations + 1):
        next_generation: List[int] = []
        for parent_id in current_parents:
            parent_node = tree.get_node(parent_id)
            for _ in range(config.copies_per_parent):
                mutated_seq = mutate_sequence(parent_node.sequence, config.error_rate)
                child_id = tree.add_child(parent_id, mutated_seq)
                next_generation.append(child_id)
        current_parents = next_generation

    return tree


def main() -> None:
    """
    Beispielhafter Aufruf der Simulation mit Speichern, Laden und Visualisieren.
    """
    # Beispiel-Konfiguration: String-Länge 10, 3 Generationen, 2 Kopien pro Eltern, 10% Fehlerrate
    config = Config(
        initial_length=10,
        generations=3,
        copies_per_parent=2,
        error_rate=0.1,
    )

    # Simulation durchführen
    tree = simulate(config)

    # Ergebnisse speichern
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "mutation_tree.json"
    tree.save_to_json(json_path)
    print(f"Baum als JSON gespeichert unter: {json_path}")

    # Baum visualisieren (als PDF)
    pdf_path = output_dir / "mutation_tree"
    tree.visualize(pdf_path, fmt="pdf")
    print(f"Graphviz-PDF erzeugt unter: {pdf_path}.pdf")

    # Beispiel: Baum erneut laden und visualisieren (optional)
    loaded_tree = MutationTree.load_from_json(json_path)
    svg_path = output_dir / "mutation_tree_loaded"
    loaded_tree.visualize(svg_path, fmt="svg")
    print(f"Geladener Baum als SVG erzeugt unter: {svg_path}.svg")


if __name__ == "__main__":
    # Seed für Reproduzierbarkeit setzen (optional)
    random.seed(42)
    main()
