#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modul: sexual_inheritance_with_selection_and_dead_mark.py

Dieses Modul simuliert über mehrere Generationen hinweg eine sexuelle Vererbung von
Zeichenketten mit folgender Abwandlung:
  1. Die Anfangspopulation besteht aus einzelnen Startknoten („founder“), z.B. "aaaa" und "cccc".
  2. In jeder Generation werden solange Nachkommen erzeugt, bis die Populationsgröße `pop_size` erreicht ist.
  3. Anschließend wird ein bestimmter Prozentsatz der unfitesten Individuen eliminiert (Selection).
  4. Alle eliminierten Knoten werden in `dead_nodes` aufgenommen.
  5. Bei der Visualisierung wird hinter jede Sequenz, deren Knoten-ID in `dead_nodes` steht, ein "†" angefügt.
  6. Die Fitness basiert darauf, wie gut der Anteil von Buchstabe₁ vs. Buchstabe₂ ausgeglichen ist:
     Fitness = |#letter1 – #letter2|, je kleiner desto fitter.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import graphviz  # pip install graphviz


@dataclass
class Config:
    """
    Konfigurationsparameter für die Simulation:
      - initial_letters: List[str]
          Die beiden Buchstaben für die Gründer‐Strings (z.B. ["a","c"]).
          Es wird erwartet, dass genau 2 Buchstaben angegeben werden.
      - sequence_length: int
          Länge jeder Sequenz (z.B. 4 → "aaaa", "cccc", etc.).
      - generations: int
          Wie viele Generationen durchlaufen werden sollen.
      - error_rate: float
          Mutation‐Wahrscheinlichkeit pro Zeichen (z.B. 0.05 für 5%).
      - pop_size: int
          Ziel‐Populationsgröße nach jeder Reproduktion (z.B. 10).
      - elimination_rate: float
          Prozentsatz (zwischen 0.0 und 1.0), der aus der gerade erzeugten
          Population nach Erreichen von `pop_size` eliminiert wird.
          Z.B. 0.3 → 30% der unfitesten Individuen entfernen.
    """
    initial_letters: List[str]         # Muss genau zwei Buchstaben enthalten, z.B. ["a","c"]
    sequence_length: int               # Länge jedes Strings
    generations: int                   # Anzahl der Generationen
    error_rate: float                  # Mutation‐Wahrscheinlichkeit pro Zeichen
    pop_size: int                      # Ziel‐Populationsgröße pro Generation
    elimination_rate: float            # Anteil, der eliminiert wird (0.0–1.0)


@dataclass
class Node:
    """
    Ein einzelner Knoten im Vererbungsnetzwerk.
      - id: eindeutige Knoten‐ID (int)
      - sequence: der String (z.B. "abbc")
      - parent_ids: Liste von Eltern‐IDs (meist genau zwei, außer Founder: leer)
      - children_ids: alle IDs der Nachkommen
    """
    id: int
    sequence: str
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)


class PopulationGraph:
    """
    Speichert alle Knoten (Population über alle Generationen) und erlaubt:
      - Hinzufügen von Founder‐Knoten und Child‐Knoten
      - Serialisierung/Deserialisierung (JSON)
      - Visualisierung mit Graphviz, wobei tote Knoten (IDs in `dead_nodes`) ein '†'-Symbol erhalten.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._next_id: int = 0
        # Set aller Knoten-IDs, die während der Selektion eliminiert wurden
        self.dead_nodes: Set[int] = set()

    def add_initial_node(self, sequence: str) -> int:
        """
        Fügt einen Gründer‐Knoten ohne Eltern hinzu.
        Rückgabe: die automatisch vergebene ID.
        """
        node = Node(id=self._next_id, sequence=sequence, parent_ids=[])
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node.id

    def add_child(self, parent1: int, parent2: int, sequence: str) -> int:
        """
        Fügt einen Kind‐Knoten mit zwei Eltern hinzu. Verknüpft in den Eltern‐Knoten
        die children_ids. Gibt die neue Knoten‐ID zurück.
        """
        if parent1 not in self._nodes or parent2 not in self._nodes:
            raise ValueError(f"Eltern-IDs {parent1} und/oder {parent2} existieren nicht.")
        node = Node(id=self._next_id, sequence=sequence, parent_ids=[parent1, parent2])
        self._nodes[self._next_id] = node
        self._nodes[parent1].children_ids.append(self._next_id)
        self._nodes[parent2].children_ids.append(self._next_id)
        self._next_id += 1
        return node.id

    def get_node(self, node_id: int) -> Node:
        """
        Liefert das Node-Objekt für die angefragte ID.
        """
        try:
            return self._nodes[node_id]
        except KeyError:
            raise ValueError(f"Node-ID {node_id} existiert nicht.")

    def all_nodes(self) -> List[Node]:
        """
        Liefert alle Knoten als Liste.
        """
        return list(self._nodes.values())

    def to_dict(self) -> Dict:
        """
        Serialisiert alle Knoten in ein Dict, das als JSON gespeichert werden kann.
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
        Baut einen PopulationGraph aus einem zuvor serialisierten Dict (z.B. JSON).
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
        Speichert die komplette PopulationGraph in eine JSON‐Datei.
        """
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "PopulationGraph":
        """
        Lädt aus einer JSON‐Datei und gibt eine PopulationGraph zurück.
        """
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def visualize(self, output_path: Path, fmt: str = "pdf") -> None:
        """
        Visualisiert das Netzwerk via Graphviz. Jeder Knoten trägt ID und Sequenz.
        Tote Knoten (ID in `dead_nodes`) erhalten ein "†" hinter dem Label.
        Kanten verlaufen von Eltern → Kind.
        """
        dot = graphviz.Digraph(comment="Sexuelles Vererbungsnetzwerk mit Selektion und toten Knoten",
                               format=fmt)
        for node in self._nodes.values():
            # Wenn dieser Knoten eliminiert wurde, ein † anfügen:
            suffix = " †" if node.id in self.dead_nodes else ""
            label = f"{node.id}: {node.sequence}{suffix}"
            dot.node(str(node.id), label)

        for node in self._nodes.values():
            for child_id in node.children_ids:
                dot.edge(str(node.id), str(child_id))

        dot.render(str(output_path), view=False, cleanup=True)


def generate_initial_sequence(letter: str, length: int) -> str:
    """
    Erzeugt einen String der Länge `length`, ausschließlich aus `letter`.
    """
    return letter * length


def mate_sequences(seq1: str, seq2: str) -> str:
    """
    Kombiniert zwei Elternsequenzen. Für jede Position wird mit 50% Wahrscheinlichkeit
    das Allel von seq1 übernommen, sonst das von seq2.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Elternsequenzen müssen dieselbe Länge haben.")
    child = []
    for c1, c2 in zip(seq1, seq2):
        child.append(c1 if random.random() < 0.5 else c2)
    return "".join(child)


def mutate_sequence(sequence: str, error_rate: float) -> str:
    """
    Mutation: Für jedes Zeichen in `sequence` mit Wahrscheinlichkeit `error_rate`
    wird der Buchstabe im Alphabet um +1 verschoben (a→b, b→c, …, z bleibt z).
    Groß‐/Kleinschreibung bleibt erhalten.
    """
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")
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


class Simulator:
    """
    Führt die Simulation in mehreren Generationen durch:
      - Startpopulation aus `initial_letters` (z.B. ["a","c"]) als reine Monozyste (z.B. "aaaa").
      - In jeder Generation: solange Kinder erzeugen, bis `pop_size` erreicht ist.
      - Dann Elimination: Entferne die `elimination_rate * pop_size` schlechtesten Individuen.
      - Alle eliminierten Knoten werden in `graph.dead_nodes` eingetragen.
      - Die verbleibenden Knoten bilden die Eltern für die nächste Generation.
      - Fitness: |#letter1 - #letter2|, je kleiner desto fitter.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.graph = PopulationGraph()
        # Aktuelle Population (Eltern‐Pool) als Liste von Node‐IDs
        self.current_population: List[int] = []

    def simulate(self) -> PopulationGraph:
        """
        Führt die gesamte Simulation durch und gibt das PopulationGraph zurück.
        """
        self._validate_config()
        self._initialize_population()

        for gen in range(1, self.config.generations + 1):
            self._generate_next_generation(gen)
        return self.graph

    def _validate_config(self) -> None:
        """
        Validiert, ob alle Konfig‐Parameter im zulässigen Bereich sind.
        """
        if len(self.config.initial_letters) != 2:
            raise ValueError("initial_letters muss genau zwei Buchstaben enthalten, z.B. ['a','c'].")
        if self.config.sequence_length < 1:
            raise ValueError("sequence_length muss mindestens 1 sein.")
        if self.config.generations < 1:
            raise ValueError("generations muss mindestens 1 sein.")
        if not 0.0 <= self.config.error_rate <= 1.0:
            raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")
        if self.config.pop_size < len(self.config.initial_letters):
            raise ValueError("pop_size muss mindestens so groß sein wie die Anzahl der initial_letters.")
        if not 0.0 <= self.config.elimination_rate < 1.0:
            raise ValueError("elimination_rate muss in [0.0, 1.0) liegen.")

    def _initialize_population(self) -> None:
        """
        Erstellt für jeden 'initial_letter' einen Gründer‐String der Länge `sequence_length`.
        Speichert dessen Knoten‐ID in `current_population`.
        """
        for letter in self.config.initial_letters:
            seq = generate_initial_sequence(letter, self.config.sequence_length)
            nid = self.graph.add_initial_node(seq)
            self.current_population.append(nid)

    def _fitness(self, sequence: str) -> int:
        """
        Berechnet die Fitness eines Strings:
          Fitness = |Anzahl(letter1) - Anzahl(letter2)|. 
        Je kleiner der Wert, desto fitter das Individuum.
        """
        letter1, letter2 = self.config.initial_letters
        count1 = sequence.count(letter1)
        count2 = sequence.count(letter2)
        return abs(count1 - count2)

    def _generate_next_generation(self, generation_number: int) -> None:
        """
        Erzeugt eine neue Generation wie folgt:
          1. Solange Nachkommen generieren, bis child_population_size == pop_size.
          2. Elimination: Entferne die `elimination_rate * pop_size` schlechtesten Individuen.
             → Füge deren IDs zu `graph.dead_nodes` hinzu.
          3. Die verbleibenden Knoten werden die Eltern für die nächste Generation.
        """
        pop_size = self.config.pop_size
        eliminate_count = int(self.config.elimination_rate * pop_size)

        # Prüfe, ob genügend Eltern vorhanden sind, um zu paaren
        if len(self.current_population) < 2:
            raise ValueError(f"Nicht genügend Eltern (brauchen mind. 2, haben {len(self.current_population)}) "
                             f"in Generation {generation_number}.")

        # 1. Kinder generieren, bis wir pop_size erreichen
        offspring_ids: List[int] = []
        while len(offspring_ids) < pop_size:
            parent1, parent2 = random.sample(self.current_population, 2)
            seq1 = self.graph.get_node(parent1).sequence
            seq2 = self.graph.get_node(parent2).sequence

            child_seq = mate_sequences(seq1, seq2)
            child_seq = mutate_sequence(child_seq, self.config.error_rate)

            child_id = self.graph.add_child(parent1, parent2, child_seq)
            offspring_ids.append(child_id)

        # 2. Elimination nach Fitness
        #    → Wir berechnen Fitness für jedes Kind, sortieren aufsteigend (0 = beste Fitness), 
        #       das Ende der Liste sind die schlechtesten.
        scored_offspring = [
            (oid, self._fitness(self.graph.get_node(oid).sequence))
            for oid in offspring_ids
        ]
        # Sortierung nach Fitness‐Score: niedrigster Score zuerst (fitteste)
        scored_offspring.sort(key=lambda x: x[1])

        # survivors sind die pop_size - eliminate_count fittesten Individuen:
        survivors = [oid for oid, score in scored_offspring[: pop_size - eliminate_count]]

        # eliminated_ids sind diejenigen, die nicht in survivors sind:
        eliminated_ids = [oid for oid, _ in scored_offspring if oid not in survivors]

        #  → Markiere alle eliminierten IDs als „dead“
        self.graph.dead_nodes.update(eliminated_ids)

        # 3. Update der aktuellen Population (Elternpool für nächste Runde)
        self.current_population = survivors

        # (Optional) Debug‐Ausgabe pro Generation:
        best_score = scored_offspring[0][1]
        worst_score = scored_offspring[-1][1]
        print(
            f"Gen {generation_number:2d}: erzeugt {pop_size} Nachkommen, "
            f"eliminiert {eliminate_count}, "
            f"Survivors = {len(survivors)} (best: {best_score}, worst: {worst_score})"
        )


def main() -> None:
    """
    Beispielaufruf der Simulation:
      - Founder: "aaaa", "cccc"
      - Sequenzlänge = 4
      - 5 Generationen
      - 10 Individuen pro Generation
      - Eliminationsrate = 30% (d.h. 3 Individuen pro Generation werden aussortiert)
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

    json_path = output_dir / "population_graph.json"
    graph.save_to_json(json_path)
    print(f"📂 Netzwerk als JSON gespeichert: {json_path}")

    # Graphviz‐PDF (tote Knoten werden mit '†' markiert)
    pdf_path = output_dir / "population_graph_dead"
    graph.visualize(pdf_path, fmt="pdf")
    print(f"📄 Graphviz-PDF erzeugt: {pdf_path}.pdf")

    # Graphviz‐SVG
    svg_path = output_dir / "population_graph_dead"
    graph.visualize(svg_path, fmt="svg")
    print(f"📄 Graphviz-SVG erzeugt: {svg_path}.svg")


if __name__ == "__main__":
    main()
