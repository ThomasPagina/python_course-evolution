#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modul: mutations_tree.py

Dieses Modul simuliert über mehrere Generationen hinweg die Kopie eines Ausgangsstrings,
fügt mit einer festgelegten Fehlerrate Mutationen hinzu, eliminiert nach jeder Generation
zufällig einen Anteil der Gesamtpopulation und speichert die Abstammungslinien in einem Tree.
Alle lebenden Knoten (inkl. Root) können in jeder Generation Nachkommen erzeugen. Verstorbene
Individuen bleiben im Baum bestehen, werden dort als "verstorben" markiert und erzeugen keine
weiteren Nachkommen. Der Baum lässt sich als JSON speichern/​laden und mit Graphviz visualisieren.
"""

import json
import random
import string
from dataclasses import dataclass, field
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
    birth_rate: float        # Erwartete Anzahl von Nachkommen pro lebendem Knoten (z.B. 1.3)
    error_rate: float        # Wahrscheinlichkeit für Kopierfehler pro Buchstabe (z.B. 0.1 für 10%)
    elimination_rate: float  # Anteil der Gesamtpopulation, der nach jeder Generation stirbt (z.B. 0.2)


@dataclass
class TreeNode:
    """
    Repräsentiert einen Knoten im Abstammungsbaum.
    Jeder Knoten besitzt eine eindeutige ID, die Sequenz, die ID des Elternknotens,
    eine Liste seiner Kinder-IDs und einen Alive-Status.
    """
    id: int
    sequence: str
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    alive: bool = True  # True = lebend, False = gestorben


class MutationTree:
    """
    Verwaltet den gesamten Baum der kopierten Strings, inklusive Alive-Status jedes Knotens.
    Bietet Methoden zum Hinzufügen von Knoten, Speichern/Laden und Visualisieren.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, TreeNode] = {}
        self._next_id: int = 0

    def add_root(self, sequence: str) -> int:
        """
        Fügt den Wurzelknoten mit der gegebenen Sequenz hinzu.
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
                    "alive": node.alive,
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
        max_id = -1
        for node_data in data["nodes"]:
            node = TreeNode(
                id=node_data["id"],
                sequence=node_data["sequence"],
                parent_id=node_data["parent_id"],
                children_ids=list(node_data["children_ids"]),
                alive=node_data.get("alive", True),
            )
            tree._nodes[node.id] = node
            if node.id > max_id:
                max_id = node.id
        tree._next_id = max_id + 1
        return tree

    def save_to_json(self, filepath: Path) -> None:
        """
        Speichert den Baum als JSON-Datei unter dem angegebenen Pfad.
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

        Lebende Knoten werden normal dargestellt, verstorbene Knoten werden
        im Label als "(verstorben)" markiert und rot eingefärbt.
        """
        dot = graphviz.Digraph(comment="Mutationsbaum", format=fmt)

        # Knoten hinzufügen
        for node in self._nodes.values():
            if node.alive:
                label = f"{node.id}: {node.sequence}"
                dot.node(str(node.id), label)
            else:
                label = f"{node.id}: {node.sequence} (verstorben)"
                dot.node(str(node.id), label, fontcolor="red")

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
    - Die Mutation ersetzt einen Buchstaben durch den nächsten im Alphabet:
      'a' -> 'b', 'b' -> 'c', …, 'z' bleibt 'z'.
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")

    sequence_list = list(sequence)
    for idx, char in enumerate(sequence_list):
        if random.random() < error_rate:
            if char.lower() == "z":
                next_char = "z"
            else:
                next_char = chr(ord(char.lower()) + 1)
            if char.isupper():
                next_char = next_char.upper()
            sequence_list[idx] = next_char

    return "".join(sequence_list)


def simulate(config: Config) -> MutationTree:
    """
    Führt die Simulation gemäß der Config durch:
    - Alle lebenden Knoten (inkl. des Root-Knotens) erzeugen in jeder Generation
      Nachkommen entsprechend der erwarteten birth_rate.
    - Die erwartete Anzahl von Kindern pro Elternknoten wird wie folgt umgesetzt:
      integer_part = int(birth_rate), extra_child_prob = birth_rate - integer_part.
      Jeder lebende Elternknoten erzeugt zunächst 'integer_part' Kopien und
      mit Wahrscheinlichkeit 'extra_child_prob' eine weitere Kopie.
    - Kopien werden mit Mutationen gemäß error_rate hergestellt.
    - Nach der Erzeugung aller Kinder einer Generation wird in der Gesamtpopulation
      (eltern + kinder) mit Wahrscheinlichkeit elimination_rate jeder lebende Knoten
      zum "gestorben" markiert (alive=False).
    - Gestorbene Knoten bleiben im Baum, produzieren aber keine weiteren Kinder.
    """
    if config.generations < 1:
        raise ValueError("Anzahl der Generationen muss mindestens 1 sein.")
    if config.birth_rate < 0.0:
        raise ValueError("birth_rate muss >= 0.0 sein.")
    if not 0.0 <= config.error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")
    if not 0.0 <= config.elimination_rate < 1.0:
        raise ValueError("elimination_rate muss zwischen 0.0 (inkl.) und 1.0 (exkl.) liegen.")

    tree = MutationTree()

    # 1. Initial-String erzeugen und als Wurzelknoten einfügen
    root_sequence = generate_initial_sequence(config.initial_length)
    root_id = tree.add_root(root_sequence)

    for gen in range(1, config.generations + 1):
        # 2. Alle zurzeit lebenden Knoten sammeln
        alive_parents = [node.id for node in tree._nodes.values() if node.alive]

        # 3. Für jeden lebenden Knoten Nachkommen erzeugen
        new_children: List[int] = []
        integer_part = int(config.birth_rate)
        extra_prob = config.birth_rate - integer_part

        for parent_id in alive_parents:
            parent_node = tree.get_node(parent_id)

            # Zunächst integer_part Kinder erzeugen
            for _ in range(integer_part):
                mutated_seq = mutate_sequence(parent_node.sequence, config.error_rate)
                child_id = tree.add_child(parent_id, mutated_seq)
                new_children.append(child_id)

            # Mit Wahrscheinlichkeit extra_prob ein weiteres Kind
            if random.random() < extra_prob:
                mutated_seq = mutate_sequence(parent_node.sequence, config.error_rate)
                child_id = tree.add_child(parent_id, mutated_seq)
                new_children.append(child_id)

        # 4. Elimination: Nach der Geburt stirbt ein Anteil der Gesamtpopulation
        for node in list(tree._nodes.values()):
            if node.alive and random.random() < config.elimination_rate:
                node.alive = False

        # Debug-Ausgabe (optional):
        # print(f"Generation {gen}: Geboren = {len(new_children)}, "
        #       f"Gesamt lebend nach Elimination = {sum(1 for n in tree._nodes.values() if n.alive)}")

    return tree


def main() -> None:
    """
    Beispielhafter Aufruf der Simulation mit Speichern, Laden und Visualisieren.
    """
    # Beispiel-Konfiguration:
    # String-Länge 10, 5 Generationen, birth_rate=1.3, 10% Fehlerrate, 15% Eliminationsrate
    config = Config(
        initial_length=10,
        generations=5,
        birth_rate=1.3,
        error_rate=0.1,
        elimination_rate=0.2,
    )

    # Seed für Reproduzierbarkeit (optional)
    random.seed(42)

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

    # Beispiel: Baum erneut laden und als SVG visualisieren
    loaded_tree = MutationTree.load_from_json(json_path)
    svg_path = output_dir / "mutation_tree_loaded"
    loaded_tree.visualize(svg_path, fmt="svg")
    print(f"Geladener Baum als SVG erzeugt unter: {svg_path}.svg")


if __name__ == "__main__":
    main()
