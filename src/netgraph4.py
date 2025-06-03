#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modul: sexual_inheritance_refactored.py

Dieses Modul simuliert über mehrere Generationen hinweg eine sexuelle Vererbung von
Zeichenketten mit strenger Selektion und zweifacher Mutationsoption.
Die Methode `_generate_next_generation` wurde in mehrere übersichtliche Untermethoden
aufgespalten, um die Lesbarkeit und Wartbarkeit zu erhöhen.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import graphviz  # pip install graphviz


@dataclass
class Config:
    """
    Konfigurationsparameter für die Simulation:
      - initial_letters: List[str]
          Die beiden Buchstaben für die Gründer‐Strings (z.B. ["a","c"]).
      - sequence_length: int
          Länge jeder Sequenz (z.B. 4 → "aaaa").
      - generations: int
          Anzahl der Generationen.
      - error_rate: float
          Mutation‐Wahrscheinlichkeit pro Zeichen (z.B. 0.05).
      - pop_size: int
          Ziel‐Populationsgröße nach jeder Reproduktion (z.B. 10).
      - elimination_rate: float
          Anteil (zwischen 0.0 und 1.0), der pro Generation eliminiert werden muss (z.B. 0.3).
    """
    initial_letters: List[str]         # Muss genau zwei Buchstaben enthalten, z.B. ["a","c"]
    sequence_length: int               # Länge jedes Strings
    generations: int                   # Anzahl der Generationen
    error_rate: float                  # Mutation‐Wahrscheinlichkeit pro Zeichen
    pop_size: int                      # Ziel‐Populationsgröße pro Generation
    elimination_rate: float            # Mindestanteil, der eliminiert wird (0.0–1.0)


@dataclass
class Node:
    """
    Ein einzelner Knoten im Vererbungsnetzwerk.
      - id:   eindeutige Knoten‐ID (int)
      - sequence: der String (z.B. "abbc")
      - parent_ids: Liste von Eltern‐IDs (zwei für Children, leer für Founder)
      - children_ids: Liste aller Nachkommen‐IDs
    """
    id: int
    sequence: str
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)


class PopulationGraph:
    """
    Speichert alle Knoten (Population über alle Generationen) und erlaubt:
      - Hinzufügen von Gründer‐ und Kind‐Knoten
      - Serialisierung/Deserialisierung (JSON)
      - Visualisierung mittels Graphviz, wobei tote Knoten (IDs in dead_nodes) ein '†' erhalten.
    """
    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._next_id: int = 0
        self.dead_nodes: Set[int] = set()  # IDs aller eliminierten Knoten

    def add_initial_node(self, sequence: str) -> int:
        node = Node(id=self._next_id, sequence=sequence, parent_ids=[])
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node.id

    def add_child(self, parent1: int, parent2: int, sequence: str) -> int:
        if parent1 not in self._nodes or parent2 not in self._nodes:
            raise ValueError(f"Eltern-IDs {parent1} und/oder {parent2} existieren nicht.")
        node = Node(id=self._next_id, sequence=sequence, parent_ids=[parent1, parent2])
        self._nodes[self._next_id] = node
        # Eltern-Knoten um child-Verweis erweitern
        self._nodes[parent1].children_ids.append(self._next_id)
        self._nodes[parent2].children_ids.append(self._next_id)
        self._next_id += 1
        return node.id

    def get_node(self, node_id: int) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise ValueError(f"Node-ID {node_id} existiert nicht.")

    def all_nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def to_dict(self) -> Dict:
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
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "PopulationGraph":
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def visualize(self, output_path: Path, fmt: str = "pdf") -> None:
        dot = graphviz.Digraph(
            comment="Sexuelles Vererbungsnetzwerk (refactored)", format=fmt
        )
        for node in self._nodes.values():
            suffix = " †" if node.id in self.dead_nodes else ""
            label = f"{node.id}: {node.sequence}{suffix}"
            dot.node(str(node.id), label)
        for node in self._nodes.values():
            for child_id in node.children_ids:
                dot.edge(str(node.id), str(child_id))
        dot.render(str(output_path), view=False, cleanup=True)


def generate_initial_sequence(letter: str, length: int) -> str:
    return letter * length


def mate_sequences(seq1: str, seq2: str) -> str:
    if len(seq1) != len(seq2):
        raise ValueError("Elternsequenzen müssen dieselbe Länge haben.")
    child = []
    for c1, c2 in zip(seq1, seq2):
        child.append(c1 if random.random() < 0.5 else c2)
    return "".join(child)


def mutate_sequence(sequence: str, error_rate: float) -> str:
    """
    Mutation (zwei Optionen):
      a) Buchstaben‐Mutation: Mit Wahrscheinlichkeit error_rate pro Zeichen
         um +1 im Alphabet verschieben.
      b) Swap‐Mutation: Zwei zufällige Positionen austauschen.
      Gesamt‐Wahrscheinlichkeit für Variante a) bzw. b) jeweils 0.5.
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")

    if random.random() < 0.5:
        # Variante a: Buchstaben‐Mutation
        seq_list = list(sequence)
        for idx, ch in enumerate(seq_list):
            if random.random() < error_rate:
                base = ch.lower()
                if base == "z":
                    new_ch = "z"
                else:
                    new_ch = chr(ord(base) + 1)
                if ch.isupper():
                    new_ch = new_ch.upper()
                seq_list[idx] = new_ch
        return "".join(seq_list)
    else:
        # Variante b: Swap‐Mutation
        if len(sequence) < 2:
            return sequence
        i, j = random.sample(range(len(sequence)), 2)
        seq_list = list(sequence)
        seq_list[i], seq_list[j] = seq_list[j], seq_list[i]
        return "".join(seq_list)


class Simulator:
    """
    Führt die Simulation in mehreren Generationen durch:
      - Startpopulation aus `initial_letters` als reine Monosequenzen.
      - In jeder Generation:
           1. Kinder erzeugen, bis pop_size erreicht ist.
           2. Strenge Selektion (alle unbalanced entfernen, ggf. weitere, bis Rate erfüllt).
           3. Tote markieren in graph.dead_nodes, Survivors werden Eltern der nächsten Generation.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.graph = PopulationGraph()
        self.current_population: List[int] = []  # Eltern‐IDS

    def simulate(self) -> PopulationGraph:
        self._validate_config()
        self._initialize_population()
        for gen in range(1, self.config.generations + 1):
            self._generate_next_generation(gen)
        return self.graph

    def _validate_config(self) -> None:
        if len(self.config.initial_letters) != 2:
            raise ValueError("initial_letters muss genau zwei Buchstaben enthalten.")
        if self.config.sequence_length < 1:
            raise ValueError("sequence_length muss mindestens 1 sein.")
        if self.config.generations < 1:
            raise ValueError("generations muss mindestens 1 sein.")
        if not 0.0 <= self.config.error_rate <= 1.0:
            raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")
        if self.config.pop_size < len(self.config.initial_letters):
            raise ValueError("pop_size muss ≥ Anzahl der initial_letters sein.")
        if not 0.0 <= self.config.elimination_rate < 1.0:
            raise ValueError("elimination_rate muss in [0.0, 1.0) liegen.")

    def _initialize_population(self) -> None:
        for letter in self.config.initial_letters:
            seq = generate_initial_sequence(letter, self.config.sequence_length)
            nid = self.graph.add_initial_node(seq)
            self.current_population.append(nid)

    def _fitness(self, sequence: str) -> int:
        letter1, letter2 = self.config.initial_letters
        return abs(sequence.count(letter1) - sequence.count(letter2))

    def _generate_next_generation(self, generation_number: int) -> None:
        pop_size = self.config.pop_size
        elimination_count_min = int(self.config.elimination_rate * pop_size)

        if len(self.current_population) < 2:
            raise ValueError(
                f"Nicht genügend Eltern (brauchen mind. 2, haben {len(self.current_population)}) "
                f"in Generation {generation_number}."
            )

        # 1. Kinder erzeugen
        offspring_ids = self._create_offspring(pop_size)

        # 2. Fitness‐Gruppen bilden
        unbalanced, balanced = self._compute_fitness_groups(offspring_ids)

        # 3. Bestimme, welche IDs eliminiert werden
        removed_ids = self._select_removed(unbalanced, balanced, elimination_count_min)

        # 4. Markiere tote Knoten und aktualisiere Eltern‐Pool
        self._update_population(offspring_ids, removed_ids)

        # 5. Debug‐Ausgabe
        num_unbalanced = len(unbalanced)
        num_removed = len(removed_ids)
        survivors = len(offspring_ids) - num_removed
        print(
            f"Gen {generation_number:2d}: erzeugt {pop_size} Kinder, "
            f"unbalanced = {num_unbalanced}, entfernt = {num_removed}, Survivors = {survivors}"
        )

    def _create_offspring(self, pop_size: int) -> List[int]:
        """
        Erzeugt exactly `pop_size` Kinder:
          - Wähle zufällig 2 Eltern aus der aktuellen Population (ohne Zurücklegen).
          - Rekombiniere mit mate_sequences und mutiere mit mutate_sequence.
          - Füge jedes Kind dem Graph hinzu und sammele dessen ID.
        """
        offspring_ids: List[int] = []
        while len(offspring_ids) < pop_size:
            parent1, parent2 = random.sample(self.current_population, 2)
            seq1 = self.graph.get_node(parent1).sequence
            seq2 = self.graph.get_node(parent2).sequence

            child_seq = mate_sequences(seq1, seq2)
            child_seq = mutate_sequence(child_seq, self.config.error_rate)

            child_id = self.graph.add_child(parent1, parent2, child_seq)
            offspring_ids.append(child_id)

        return offspring_ids

    def _compute_fitness_groups(self, offspring_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Teilt die Liste `offspring_ids` basierend auf Fitness in:
          - unbalanced: fitness > 0
          - balanced:   fitness == 0
        """
        scored = [
            (oid, self._fitness(self.graph.get_node(oid).sequence))
            for oid in offspring_ids
        ]
        unbalanced = [oid for oid, score in scored if score > 0]
        balanced = [oid for oid, score in scored if score == 0]
        return unbalanced, balanced

    def _select_removed(
        self,
        unbalanced: List[int],
        balanced: List[int],
        elimination_count_min: int
    ) -> List[int]:
        """
        Entferne zuerst alle `unbalanced`. Falls deren Anzahl < elimination_count_min,
        entferne zusätzlich zufällig aus `balanced`, bis die Mindestanzahl erreicht ist.
        """
        removed_ids: List[int] = []
        removed_ids.extend(unbalanced)

        already_removed = len(removed_ids)
        if already_removed < elimination_count_min:
            remaining = elimination_count_min - already_removed
            if remaining >= len(balanced):
                removed_ids.extend(balanced)
            else:
                extra = random.sample(balanced, remaining)
                removed_ids.extend(extra)

        return removed_ids

    def _update_population(self, offspring_ids: List[int], removed_ids: List[int]) -> None:
        """
        Markiere alle `removed_ids` als tot und aktualisiere `self.current_population`
        mit den überlebenden IDs (offspring_ids minus removed_ids).
        """
        self.graph.dead_nodes.update(removed_ids)
        survivors = [oid for oid in offspring_ids if oid not in removed_ids]
        self.current_population = survivors


def main() -> None:
    """
    Beispielaufruf der Simulation:
      - Founder: "aaaa", "cccc"
      - Sequenzlänge = 4
      - 5 Generationen
      - 10 Individuen pro Generation
      - Eliminationsrate = 30% (mindestens 3 pro Runde)
      - Mutation: 20% pro Zeichen
    """
    config = Config(
        initial_letters=["a", "c"],
        sequence_length=4,
        generations=5,
        error_rate=0.20,
        pop_size=10,
        elimination_rate=0.30,
    )

    random.seed(42)  # Für Reproduzierbarkeit

    simulator = Simulator(config)
    graph = simulator.simulate()

    # Speichere Ergebnis
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "population_graph_refactored.json"
    graph.save_to_json(json_path)
    print(f"📂 Netzwerk als JSON gespeichert: {json_path}")

    # Graphviz‐PDF (tote Knoten werden mit '†' markiert)
    pdf_path = output_dir / "population_graph_refactored"
    graph.visualize(pdf_path, fmt="pdf")
    print(f"📄 Graphviz-PDF erzeugt: {pdf_path}.pdf")

    # Graphviz‐SVG
    svg_path = output_dir / "population_graph_refactored"
    graph.visualize(svg_path, fmt="svg")
    print(f"📄 Graphviz-SVG erzeugt: {svg_path}.svg")


if __name__ == "__main__":
    main()
