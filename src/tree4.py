#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import graphviz

@dataclass
class Config:
    initial_length: int
    generations: int
    max_population: int
    error_rate: float
    elimination_rate: float

@dataclass
class TreeNode:
    id: int
    sequence: str
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    alive: bool = True

    def fitness(self) -> float:
        unique_letters = set(self.sequence)
        return len(unique_letters) / len(self.sequence)

class MutationTree:
    def __init__(self) -> None:
        self._nodes: Dict[int, TreeNode] = {}
        self._next_id: int = 0

    def add_root(self, sequence: str) -> int:
        node = TreeNode(id=self._next_id, sequence=sequence, parent_id=None)
        self._nodes[self._next_id] = node
        self._next_id += 1
        return node.id

    def add_child(self, parent_id: int, sequence: str) -> int:
        if parent_id not in self._nodes:
            raise ValueError(f"Parent-ID {parent_id} existiert nicht.")
        node = TreeNode(id=self._next_id, sequence=sequence, parent_id=parent_id)
        self._nodes[self._next_id] = node
        self._nodes[parent_id].children_ids.append(self._next_id)
        self._next_id += 1
        return node.id

    def get_node(self, node_id: int) -> TreeNode:
        try:
            return self._nodes[node_id]
        except KeyError as e:
            raise ValueError(f"Node-ID {node_id} existiert nicht.") from e

    def alive_nodes(self) -> List[int]:
        return [nid for nid, node in self._nodes.items() if node.alive]

    def total_population(self) -> int:
        return len(self._nodes)

    def to_dict(self) -> Dict:
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
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "MutationTree":
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def visualize(self, output_path: Path, fmt: str = "pdf") -> None:
        dot = graphviz.Digraph(comment="Mutationsbaum", format=fmt)
        for node in self._nodes.values():
            if node.alive:
                label = f"{node.id}: {node.sequence}"
                dot.node(str(node.id), label)
            else:
                label = f"{node.id}: {node.sequence} (†)"
                dot.node(str(node.id), label, fontcolor="red")
        for node in self._nodes.values():
            if node.parent_id is not None:
                dot.edge(str(node.parent_id), str(node.id))
        dot.render(str(output_path), view=False, cleanup=True)

def generate_initial_sequence(length: int) -> str:
    return "a" * length

def mutate_sequence(sequence: str, error_rate: float) -> str:
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate muss zwischen 0.0 und 1.0 liegen.")
    seq_list = list(sequence)
    for idx, char in enumerate(seq_list):
        if random.random() < error_rate:
            next_char = "z" if char.lower() == "z" else chr(ord(char.lower()) + 1)
            if char.isupper():
                next_char = next_char.upper()
            seq_list[idx] = next_char
    return "".join(seq_list)

class Simulator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.tree = MutationTree()

    def simulate(self) -> MutationTree:
        self._validate_config()
        self._initialize_root()
        for gen in range(1, self.config.generations + 1):
            self._generate_offspring_to_capacity()
            self._apply_elimination_exact()
        return self.tree

    def _validate_config(self) -> None:
        if self.config.generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        if self.config.max_population < 1:
            raise ValueError("max_population must be at least 1.")
        if not 0.0 <= self.config.error_rate <= 1.0:
            raise ValueError("error_rate must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.elimination_rate < 1.0:
            raise ValueError("elimination_rate must be between 0.0 (inclusive) and 1.0 (exclusive).")

    def _initialize_root(self) -> None:
        root_seq = generate_initial_sequence(self.config.initial_length)
        self.tree.add_root(root_seq)

    def _generate_offspring_to_capacity(self) -> None:
        alive_ids = self.tree.alive_nodes()
        current_alive = len(alive_ids)
        capacity = self.config.max_population
        if current_alive >= capacity:
            return
        num_to_generate = capacity - current_alive
        for _ in range(num_to_generate):
            parent_id = random.choice(alive_ids)
            parent_seq = self.tree.get_node(parent_id).sequence
            mutated_seq = mutate_sequence(parent_seq, self.config.error_rate)
            self.tree.add_child(parent_id, mutated_seq)

    def _apply_elimination_exact(self) -> None:
        alive_ids = self.tree.alive_nodes()
        alive_count = len(alive_ids)
        if alive_count == 0:
            return
        num_to_eliminate = int(self.config.elimination_rate * alive_count)
        if num_to_eliminate < 1:
            return
        alive_nodes = [self.tree.get_node(nid) for nid in alive_ids]
        alive_nodes.sort(key=lambda node: node.fitness())
        to_eliminate = alive_nodes[:num_to_eliminate]
        for node in to_eliminate:
            node.alive = False

def main() -> None:
    config = Config(
        initial_length=10,
        generations=5,
        max_population=15,
        error_rate=0.1,
        elimination_rate=0.4,
    )
    random.seed(42)
    simulator = Simulator(config)
    tree = simulator.simulate()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "mutation_tree.json"
    tree.save_to_json(json_path)
    print(f"Tree saved as JSON at: {json_path}")
    pdf_path = output_dir / "mutation_tree"
    tree.visualize(pdf_path, fmt="pdf")
    print(f"Graphviz PDF generated at: {pdf_path}.pdf")
    loaded_tree = MutationTree.load_from_json(json_path)
    svg_path = output_dir / "mutation_tree_loaded"
    loaded_tree.visualize(svg_path, fmt="svg")
    print(f"Loaded tree as SVG generated at: {svg_path}.svg")

if __name__ == "__main__":
    main()
