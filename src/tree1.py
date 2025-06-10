#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import string
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import graphviz

@dataclass
class Config:
    initial_length: int
    generations: int
    copies_per_parent: int
    error_rate: float

@dataclass
class TreeNode:
    id: int
    sequence: str
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

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

    def to_dict(self) -> Dict:
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
        tree = cls()
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
            label = f"{node.id}: {node.sequence}"
            dot.node(str(node.id), label)
        for node in self._nodes.values():
            if node.parent_id is not None:
                dot.edge(str(node.parent_id), str(node.id))
        dot.render(str(output_path), view=False, cleanup=True)

def generate_initial_sequence(length: int) -> str:
    return "a" * length

def mutate_sequence(sequence: str, error_rate: float) -> str:
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
    if config.generations < 1:
        raise ValueError("Anzahl der Generationen muss mindestens 1 sein.")
    if config.copies_per_parent < 1:
        raise ValueError("copies_per_parent muss mindestens 1 sein.")
    tree = MutationTree()
    root_sequence = generate_initial_sequence(config.initial_length)
    root_id = tree.add_root(root_sequence)
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
    config = Config(
        initial_length=10,
        generations=3,
        copies_per_parent=2,
        error_rate=0.1,
    )
    tree = simulate(config)
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
    random.seed(42)
    main()
