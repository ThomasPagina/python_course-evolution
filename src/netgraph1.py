#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modul: sexual_inheritance.py

Dieses Modul simuliert über mehrere Generationen hinweg eine sexuelle Vererbung von
Zeichenketten. Die Anfangspopulation besteht aus Strings gleicher Länge, jeweils
bestehend nur aus einem einzigen Buchstaben (z.B. "aaaa", "bbbb"). In jeder Generation
werden aus dieser Population Paare zufällig gewählt, und deren Allele (Zeichen)
werden für jeden Nachkommen kombiniert. Anschließend kann bei jedem Kindzeichen
mit einer bestimmten Fehlerrate eine Mutation stattfinden (ein Buchstabe wird im
Alphabet um einen ersetzt). Ein Kind erhält zwei Eltern-IDs und alle Kinder werden
im Graphen gespeichert. Am Ende entsteht ein gerichteter Netzwerkgraph (kein reiner Baum),
der mit Graphviz visualisiert werden kann.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import graphviz  # pip install graphviz


@dataclass
class Config:
    """
    Konfigurationsparameter für die Simulation.
    - initial_letters: Liste der Zeichen, aus denen die Anfangsstrings aufgebaut werden.
      Jeder Buchstabe erzeugt genau einen Startstring, der ausschließlich aus diesem
      Buchstaben besteht, mit Länge `sequence_length`.
    - sequence_length: Länge jedes Strings in allen Generationen.
    - generations: Anzahl der Generationen, die simuliert werden.
    - error_rate: Wahrscheinlichkeit für eine Mutation pro Zeichen im Kindstring.
    """
    initial_letters: List[str]
    sequence_length: int
    generations: int
    error_rate: float  # z.B. 0.05 für 5% Mutationswahrscheinlichkeit pro Zeichen


@dataclass
class Node:
    """
    Repräsentiert einen Knoten im Vererbungsnetzwerk.
    Jeder Knoten hat:
    - eine eindeutige ID (int)
    - die Sequenz (String)
    - eine Liste von parent_ids (List[int]), in der Regel genau zwei Eltern, außer bei Anfangs-Knoten
    - eine Liste von children_ids (List[int])
    """
    id: int
    sequence: str
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)


class PopulationGraph:
    """
    Verwaltet das gesamte Netzwerk aller erzeugten Knoten (Population über alle Generationen).
    Bietet Methoden zum Hinzufügen von Anfangsknoten und Kindknoten sowie
    Speichern/Laden und Visualisieren mit Graphviz.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._next_id: int = 0

    def add_initial_node(self, sequence: str) -> int:
        """
        Fügt einen Startknoten ohne Eltern hinzu (Anfangspopulation).
        Gibt die ID des neuen Knotens zurück.
        """
        node = Node(id=self._next_id, sequence=sequence, parent_ids=[])
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node.id

    def add_child(self, parent1: int, parent2: int, sequence: str) -> int:
        """
        Erzeugt einen Kindknoten mit zwei Eltern (parent1, parent2) und speichert dessen Sequenz.
        Gibt die ID des neu erstellten Kindes zurück.
        """
        if parent1 not in self._nodes or parent2 not in self._nodes:
            raise ValueError(f"Eltern-IDs {parent1} und/oder {parent2} existieren nicht.")
        node = Node(id=self._next_id, sequence=sequence, parent_ids=[parent1, parent2])
        self._nodes[self._next_id] = node
        # Verknüpfe Kind in beiden Elternknoten
        self._nodes[parent1].children_ids.append(self._next_id)
        self._nodes[parent2].children_ids.append(self._next_id)
        self._next_id += 1
        return node.id

    def get_node(self, node_id: int) -> Node:
        """
        Liefert den Node für die gegebene ID.
        """
        try:
            return self._nodes[node_id]
        except KeyError as e:
            raise ValueError(f"Node-ID {node_id} existiert nicht.") from e

    def all_nodes(self) -> List[Node]:
        """
        Liefert alle gespeicherten Knoten als Liste.
        """
        return list(self._nodes.values())

    def to_dict(self) -> Dict:
        """
        Wandelt das gesamte Netzwerk in ein serialisierbares Dict um.
        """
        return {
            "nodes": [
                {
                    "id": node.id,
                    "sequence": node.sequence,
                    "parent_ids": node.parent_ids,
                    "children_ids": node.children_ids,
                }
                for node in self._nodes.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PopulationGraph":
        """
        Erstellt einen PopulationGraph aus einem Dict (z.B. aus JSON).
        """
        graph = cls()
        max_id = -1
        for node_data in data["nodes"]:
            node = Node(
                id=node_data["id"],
                sequence=node_data["sequence"],
                parent_ids=list(node_data["parent_ids"]),
                children_ids=list(node_data["children_ids"]),
            )
            graph._nodes[node.id] = node
            if node.id > max_id:
                max_id = node.id
        graph._next_id = max_id + 1
        return graph

    def save_to_json(self, filepath: Path) -> None:
        """
        Speichert das Netzwerk als JSON-Datei.
        """
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "PopulationGraph":
        """
        Lädt das Netzwerk aus einer JSON-Datei und gibt den PopulationGraph zurück.
        """
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def visualize(self, output_path: Path, fmt: str = "pdf") -> None:
        """
        Visualisiert das Netzwerk mit Graphviz als gerichteten Graphen.
        Jeder Knoten wird mit seiner ID und Sequenz beschriftet. Kanten zeigen
        von jedem Elternknoten zu seinen Kindern.
        """
        dot = graphviz.Digraph(comment="Sexuelles Vererbungsnetzwerk", format=fmt)

        # Knoten hinzufügen
        for node in self._nodes.values():
            label = f"{node.id}: {node.sequence}"
            dot.node(str(node.id), label)

        # Kanten hinzufügen (von jedem Elternknoten zu seinen Kindern)
        for node in self._nodes.values():
            for child_id in node.children_ids:
                dot.edge(str(node.id), str(child_id))

        dot.render(str(output_path), view=False, cleanup=True)


def generate_initial_sequence(letter: str, length: int) -> str:
    """
    Erzeugt einen String der Länge 'length', bestehend ausschließlich aus 'letter'-Zeichen.
    """
    return letter * length


def mate_sequences(seq1: str, seq2: str) -> str:
    """
    Kombiniert zwei Elternsequenzen seq1 und seq2.
    Für jede Position i wird zufällig entschieden, welches Allel (Zeichen) übernommen wird.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Elternsequenzen müssen dieselbe Länge haben.")
    child_list = []
    for c1, c2 in zip(seq1, seq2):
        # Mit 50% Wahrscheinlichkeit c1, sonst c2
        child_list.append(c1 if random.random() < 0.5 else c2)
    return "".join(child_list)


def mutate_sequence(sequence: str, error_rate: float) -> str:
    """
    Führt eine Mutation auf einer gegebenen Sequenz aus:
    - Für jedes Zeichen hat man Fehlerwahrscheinlichkeit error_rate,
      und wenn Mutation eintritt, wird der Buchstabe um eins im Alphabet verschoben.
    - Es kann dabei passieren, dass kein Zeichen verändert wird (Mutation optional).
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")

    seq_list = list(sequence)
    for idx, char in enumerate(seq_list):
        if random.random() < error_rate:
            if char.lower() == "z":
                new_char = "z"
            else:
                new_char = chr(ord(char.lower()) + 1)
            if char.isupper():
                new_char = new_char.upper()
            seq_list[idx] = new_char

    return "".join(seq_list)


class Simulator:
    """
    Kapselt die Simulation mit sexueller Vererbung:
    - Initialisierung einer Anfangspopulation aus Uniform-Strings.
    - Für jede Generation: Paarweise Verpaarung, Mischen der Allele, Mutation.
    - Alle erzeugten Knoten werden im PopulationGraph gespeichert, der ein gerichtetes
      Netzwerk darstellt (Eltern → Kind).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.graph = PopulationGraph()
        # Aktuelle Population wird als Liste von Knoten-IDs geführt
        self.current_population: List[int] = []

    def simulate(self) -> PopulationGraph:
        """
        Führt die gesamte Simulation durch und gibt das vollständige PopulationGraph zurück.
        """
        self._validate_config()
        self._initialize_population()

        # In jeder Generation werden genau so viele Kinder erzeugt,
        # wie aktuell in der Population sind (Populationsgröße bleibt konstant).
        pop_size = len(self.current_population)
        for _ in range(self.config.generations):
            self._generate_next_generation(pop_size)
        return self.graph

    def _validate_config(self) -> None:
        """
        Stellt sicher, dass die Konfigurationsparameter gültig sind.
        """
        if not self.config.initial_letters:
            raise ValueError("initial_letters darf nicht leer sein.")
        if self.config.sequence_length < 1:
            raise ValueError("sequence_length muss mindestens 1 sein.")
        if self.config.generations < 1:
            raise ValueError("generations muss mindestens 1 sein.")
        if not 0.0 <= self.config.error_rate <= 1.0:
            raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")

    def _initialize_population(self) -> None:
        """
        Erzeugt die Anfangspopulation: Für jeden Buchstaben in initial_letters
        einen String, der nur aus diesem Buchstaben besteht.
        Speichert die entsprechenden Knoten-IDs in current_population.
        """
        for letter in self.config.initial_letters:
            seq = generate_initial_sequence(letter, self.config.sequence_length)
            node_id = self.graph.add_initial_node(seq)
            self.current_population.append(node_id)

    def _generate_next_generation(self, pop_size: int) -> None:
        """
        Erzeugt die nächste Generation in der Größe 'pop_size':
        - Wähle für jeden neuen Nachkommen zufällig zwei Eltern aus der aktuellen Population
          (ohne Zurücklegen).
        - Mische deren Sequenzen mit mate_sequences.
        - Wende mutate_sequence an.
        - Füge den neuen Knoten in das Netzwerk ein und baue die Eltern-Kind-Beziehungen auf.
        - Die neue Generation ersetzt die alte Population (Generationsersetzung).
        """
        new_population: List[int] = []
        for _ in range(pop_size):
            parent1, parent2 = random.sample(self.current_population, 2)
            seq1 = self.graph.get_node(parent1).sequence
            seq2 = self.graph.get_node(parent2).sequence

            # Sequenzen mischen
            child_seq = mate_sequences(seq1, seq2)
            # Mutation
            child_seq = mutate_sequence(child_seq, self.config.error_rate)

            # Kindknoten anlegen mit zwei Eltern
            child_id = self.graph.add_child(parent1, parent2, child_seq)
            new_population.append(child_id)

        # Aktualisiere die aktuelle Population
        self.current_population = new_population


def main() -> None:
    """
    Beispielhafter Aufruf der Simulation:
    - Anfangspopulation aus "aaaa", "cccc" (zwei Knoten).
    - Sequenzlänge = 4, 6 Generationen, 20% Fehlerrate pro Zeichen.
    """
    config = Config(
        initial_letters=["a", "c"],
        sequence_length=4,
        generations=6,
        error_rate=0.2,
    )

    random.seed(42)  # Für Reproduzierbarkeit (optional)

    simulator = Simulator(config)
    graph = simulator.simulate()

    # Ergebnisse speichern
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "population_graph.json"
    graph.save_to_json(json_path)
    print(f"Netzwerk als JSON gespeichert unter: {json_path}")

    # Netzwerk visualisieren (als PDF)
    pdf_path = output_dir / "population_graph"
    graph.visualize(pdf_path, fmt="pdf")
    print(f"Graphviz-PDF erzeugt unter: {pdf_path}.pdf")

    # Netzwerk als svg speichern
    svg_path = output_dir / "population_graph"
    graph.visualize(svg_path, fmt="svg")
    print(f"Graphviz-SVG erzeugt unter: {svg_path}.svg")


if __name__ == "__main__":
    main()
