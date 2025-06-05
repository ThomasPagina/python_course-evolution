"""
reconstructTree.py

This Python program loads a JSON file describing a mutation tree and
constructs a hypothetical tree using an Evolutionary Strategy (ES)
from only the living (alive=true) nodes. Mutation constraints are enforced:
A child may only differ from its parent node by incremental letter changes
(forward in the alphabet).

All processes guarantee that exactly one root node exists at all times.
If a node becomes a sibling of the current root node,
a new hypothetical root is automatically created with both as children.

The mutation uses a guided strategy that corrects invalid placements starting
from the leaves, then resolves overflow, and finally removes redundant hypothetical nodes.
When correcting an invalid edge, if a direct swap of parent and child restores
the validity of the tree, the swap is performed:

1. Find an invalid edge (leaf-first).
2. Attempt a direct parent-child swap:
    - If the swapped tree contains **no** further invalidities,
      the swap is accepted.
3. If the swap fails or leaves an invalidity, perform a
    "reattach": The current leaf is either attached to a sibling node,
    or, if the sibling procedure involves the root, a new hypothetical
    root is created.
4. If there are no invalid edges, resolve overflow (nodes with >2 children)
    by inserting a hypothetical node between the two most closely related children
    (minimum Levenshtein distance).
5. If neither invalidity nor overflow is present, attempt to remove a redundant
    hypothetical node (bypass).

The final best tree is visualized using Graphviz; hypothetical nodes
are marked in the Graphviz label with a leading asterisk (*).
"""

import json
import random
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import graphviz  # pip install graphviz required


# ---------------------------
#  Configuration Dataclass
# ---------------------------

@dataclass
class Config:
     """
     Configuration parameters for the evolutionary algorithm.
     """
     input_path: str = "mutation_tree.json"
     population_size: int = 50
     generations: int = 100
     remove_fraction: float = 0.2
     seed: Optional[int] = None
     output_graph_basename: str = "best_tree"


# ---------------------------
#  Data Structures
# ---------------------------

class Node:
     """
     Represents a node in the tree.
     Can be an original (living) node from the JSON file
     or a hypothetical node added during evolution.
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
     Represents a tree of Node objects.
     Parent-child relationships are stored in dictionaries.
     """
     def __init__(
          self,
          nodes: Dict[int, Node],
          parent_map: Dict[int, Optional[int]],
          children_map: Dict[int, List[int]]
     ) -> None:
          """
          :param nodes: Dict of node_id -> Node instance (including hypothetical nodes)
          :param parent_map: Dict of node_id -> parent_id or None if root
          :param children_map: Dict of node_id -> list of children_ids
          """
          self.nodes: Dict[int, Node] = nodes
          self.parent_map: Dict[int, Optional[int]] = parent_map
          self.children_map: Dict[int, List[int]] = children_map

     def copy(self) -> "Tree":
          """
          Creates a deep copy of the current tree,
          so that changes to the copy do not affect the original.
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
          Returns the node_id of the root (parent_map[node_id] is None).
          If multiple roots exist, returns the first one.
          """
          roots = [nid for nid, pid in self.parent_map.items() if pid is None]
          return roots[0] if roots else None

     def all_node_ids(self) -> List[int]:
          """Returns all node_ids in this tree."""
          return list(self.nodes.keys())

     def is_leaf(self, node_id: int) -> bool:
          """Checks if a node is a leaf."""
          return len(self.children_map.get(node_id, [])) == 0

     def as_edge_list(self) -> List[Tuple[int, int]]:
          """
          Returns all edges (parent_id, child_id) as a list of tuples.
          """
          edges: List[Tuple[int, int]] = []
          for child, parent in self.parent_map.items():
                if parent is not None:
                     edges.append((parent, child))
          return edges

     def has_any_invalid_edge(self) -> bool:
          """
          Checks if there is any edge (p, c) where the mutation constraint is violated.
          """
          for parent_id, child_id in self.as_edge_list():
                seq_p = self.nodes[parent_id].sequence
                seq_c = self.nodes[child_id].sequence
                if not is_valid_mutation(seq_p, seq_c):
                     return True
          return False

     def is_valid_structure(self) -> bool:
          """
          Checks if the tree forms a valid structure without cycles (connected).
          Only checks for acyclicity, not mutation constraints.
          """
          visited: Set[int] = set()
          root = self.get_root()
          if root is None:
                return False

          stack = [root]
          while stack:
                current = stack.pop()
                if current in visited:
                     return False  # Cycle
                visited.add(current)
                for child in self.children_map.get(current, []):
                     stack.append(child)

          return visited == set(self.nodes.keys())


# ---------------------------
#  Helper Functions
# ---------------------------

def load_json_nodes(filepath: str) -> List[Dict]:
     """
     Reads the JSON file and returns the list of node dictionaries.
     """
     with open(filepath, 'r', encoding='utf-8') as f:
          data = json.load(f)
     return data.get("nodes", [])


def filter_alive_nodes(raw_nodes: List[Dict]) -> Dict[int, str]:
     """
     Filters from the raw JSON nodes those that are alive=true.
     Returns a dict: node_id -> sequence.
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
     Computes the Levenshtein distance between two strings s1 and s2.
     Runtime: O(len(s1)*len(s2))
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
     Checks if child_seq could have arisen from parent_seq by incremental letter changes
     (letters only move forward in the alphabet).
     For all positions i: ord(parent_seq[i]) <= ord(child_seq[i]).
     """
     for c_p, c_c in zip(parent_seq, child_seq):
          if ord(c_c) < ord(c_p):
                return False
     return True


# ---------------------------
#  Initial Tree Generation
# ---------------------------

def generate_random_tree(alive_sequences: Dict[int, str]) -> Tree:
     """
     Randomly generates a tree from the living nodes (without hypothetical nodes),
     where each node has at most two children and there is exactly one root.
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
#  Fitness Function
# ---------------------------

def compute_tree_fitness(tree: Tree, penalty_invalid: float = 1000.0) -> float:
     """
     Computes the fitness of a tree. Lower is better.
     - For each edge (p, c):
        * If is_valid_mutation(seq_p, seq_c) == False, add penalty_invalid.
        * Otherwise, sum the Levenshtein distance.
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
#  Helper: Find Invalid Edges (Leaf-first)
# ---------------------------

def find_invalid_edge_leaf_first(tree: Tree) -> Optional[Tuple[int, int]]:
     """
     First considers leaves:
     For each leaf (node_id with no children), check if the edge (p, leaf) is invalid.
     If found, return (parent_id, leaf_id).
     If no error is found in leaves, check internal nodes (in any order)
     and return the first invalid edge.
     """
     # 1) Check leaves
     for node_id in tree.all_node_ids():
          if tree.is_leaf(node_id):
                parent_id = tree.parent_map.get(node_id)
                if parent_id is None:
                     continue
                seq_p = tree.nodes[parent_id].sequence
                seq_c = tree.nodes[node_id].sequence
                if not is_valid_mutation(seq_p, seq_c):
                     return parent_id, node_id

     # 2) Check all other edges
     for parent_id, child_id in tree.as_edge_list():
          if tree.is_leaf(child_id):
                continue  # Leaf edges already checked
          seq_p = tree.nodes[parent_id].sequence
          seq_c = tree.nodes[child_id].sequence
          if not is_valid_mutation(seq_p, seq_c):
                return parent_id, child_id

     return None


# ---------------------------
#  Reattach with Root Handling
# ---------------------------

def reattach_leaf(tree: Tree, parent_id: int, child_id: int, next_hypo_id: int) -> Tuple[Tree, int]:
     """
     Attempts to detach the leaf child_id (c) from parent_id (p):
     1. Attach c as a child of a sibling node of p, if possible.
     2. If p was a direct child of the root node (p had parent=None),
         create a new hypothetical root with both as children.
     3. Otherwise, set c as a sibling of p (parent[c] = gp).
     """
     new_tree = tree.copy()
     p = parent_id
     c = child_id
     gp = new_tree.parent_map.get(p)  # grandparent

     # 1) Try to attach c to a sibling of p
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

     # 2) If p was directly under root (gp is None), create new hypothetical root
     if gp is None:
          old_root = p
          seq_old = new_tree.nodes[old_root].sequence
          seq_c = new_tree.nodes[c].sequence

          # Create hypothetical root h
          h_id = next_hypo_id
          next_hypo_id -= 1
          hypo_seq = "".join(chr(min(ord(ch1), ord(ch2))) for ch1, ch2 in zip(seq_old, seq_c))
          new_tree.nodes[h_id] = Node(h_id, hypo_seq, is_hypothetical=True)
          new_tree.children_map[h_id] = []

          # Remove c from p
          new_tree.children_map[p].remove(c)

          # Set h as new root (parent=None)
          new_tree.parent_map[h_id] = None

          # Assign old root and c as children of h
          new_tree.parent_map[old_root] = h_id
          new_tree.parent_map[c] = h_id
          new_tree.children_map[h_id].extend([old_root, c])

          return new_tree, next_hypo_id

     # 3) Otherwise, set c as sibling of p (parent[c] = gp)
     new_tree.children_map[p].remove(c)
     new_tree.parent_map[c] = gp
     new_tree.children_map[gp].append(c)
     return new_tree, next_hypo_id


# ---------------------------
#  Swap Function (Parent <-> Child)
# ---------------------------

def swap_parent_child(tree: Tree, parent_id: int, child_id: int) -> Tree:
     """
     Swaps the roles of parent_id and child_id in the tree,
     so that child_id becomes the parent of parent_id.
     Only checks for acyclicity; mutation constraints must be validated afterwards.
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
          return new_tree  # No swap, cycle

     old_parent_of_p = new_tree.parent_map[parent_id]
     old_children_of_c = list(new_tree.children_map[child_id])

     # 1) Remove parent_id from children of old_parent_of_p
     if old_parent_of_p is not None:
          new_tree.children_map[old_parent_of_p].remove(parent_id)

     # 2) Set child_id in place of parent_id
     new_tree.parent_map[child_id] = old_parent_of_p
     if old_parent_of_p is not None:
          new_tree.children_map[old_parent_of_p].append(child_id)

     # 3) parent_id becomes child of child_id
     new_tree.parent_map[parent_id] = child_id

     # 4) Attach old children of child_id (except parent_id) to parent_id
     for ch in old_children_of_c:
          if ch != parent_id:
                new_tree.parent_map[ch] = parent_id
                new_tree.children_map[parent_id].append(ch)

     # 5) Set children_map[child_id] = [parent_id]
     new_tree.children_map[child_id] = [parent_id]

     return new_tree


# ---------------------------
#  Handle Overflow and Hypotheticals
# ---------------------------

def find_node_with_overflow(tree: Tree) -> Optional[int]:
     """
     Searches the tree for a node with more than two children.
     Returns the node_id or None if no overflow exists.
     """
     for nid, children in tree.children_map.items():
          if len(children) > 2:
                return nid
     return None


def add_hypothetical_for_overflow(tree: Tree, next_hypo_id: int) -> Tuple[Tree, int]:
     """
     Inserts a hypothetical node between two children of an overflowing node,
     uniting the two children whose sequences have the smallest Levenshtein distance.
     This reduces the number of direct children of the overflow node by 1.
     """
     new_tree = tree.copy()
     overflow_node = find_node_with_overflow(new_tree)
     if overflow_node is None:
          return new_tree, next_hypo_id

     children = new_tree.children_map[overflow_node]
     # Choose two children with minimal distance
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
     Searches for a hypothetical node whose removal does not violate mutation constraints.
     If found, this node is removed (bypassed), and its child is attached directly to its parent.
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
#  Guided Mutation (Leaf-first with Swap and Root Handling)
# ---------------------------

def guided_mutation(tree: Tree, next_hypo_id: int) -> Tuple[Tree, int]:
     """
     Performs a guided mutation in this order:
     1. Find invalid edge (leaf-first).
         a) Try a direct swap (p, c).
             - If after the swap the tree is valid (no invalid edges),
                accept the swap.
         b) Otherwise: reattach_leaf (possibly with new root).
     2. If no invalid edges, check for overflow and insert hypothetical node.
     3. If neither invalidity nor overflow, remove redundant hypothetical nodes.
     """
     invalid_edge = find_invalid_edge_leaf_first(tree)
     if invalid_edge is not None:
          p, c = invalid_edge
          # 1a) Try direct swap
          swapped = swap_parent_child(tree, p, c)
          if swapped is not tree and not swapped.has_any_invalid_edge():
                return swapped, next_hypo_id
          # 1b) Reattach (including root handling)
          return reattach_leaf(tree, p, c, next_hypo_id)

     # 2) Handle overflow
     overflow_node = find_node_with_overflow(tree)
     if overflow_node is not None:
          return add_hypothetical_for_overflow(tree, next_hypo_id)

     # 3) Remove hypothetical node
     new_tree = remove_redundant_hypothetical(tree)
     return new_tree, next_hypo_id


# ---------------------------
#  Evolutionary Strategy (Main Loop)
# ---------------------------

def evolutionary_algorithm(
     alive_sequences: Dict[int, str],
     population_size: int = 50,
     generations: int = 100,
     remove_fraction: float = 0.2,
     seed: Optional[int] = None
) -> Tree:
     """
     Runs the evolutionary strategy and returns the best tree.
     :param alive_sequences: Dict {node_id: sequence} for living nodes.
     :param population_size: Number of trees in the population.
     :param generations: Maximum number of generations.
     :param remove_fraction: Fraction (0..1) of the worst trees removed per generation.
     :param seed: Optional random seed for reproducibility.
     :return: The tree with the best fitness at the end.
     """
     if seed is not None:
          random.seed(seed)

     next_hypo_id = -1  # Start ID for hypothetical nodes

     # 1) Create initial population
     population: List[Tree] = []
     for _ in range(population_size):
          tree = generate_random_tree(alive_sequences)
          population.append(tree)

     # 2) Evolutionary loop
     for gen in range(1, generations + 1):
          scored_population: List[Tuple[Tree, float]] = [
                (tree, compute_tree_fitness(tree)) for tree in population
          ]
          scored_population.sort(key=lambda x: x[1])

          best_fit = scored_population[0][1]
          print(f"Generation {gen:3d} | Best fitness: {best_fit:.2f}")

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
     print(f"Evolution completed. Best final fitness: {final_scored[0][1]:.2f}")
     return best_tree


# ---------------------------
#  Graphviz Visualization
# ---------------------------

def visualize_tree(tree: Tree, output_basename: str) -> None:
     """
     Visualizes the tree using Graphviz. Hypothetical nodes are marked
     with a leading asterisk (*) in the label.
     Saves as DOT and PDF files: {output_basename}.gv and {output_basename}.pdf
     """
     dot = graphviz.Digraph(comment="Reconstructed Tree", format="pdf")
     dot.attr("node", shape="ellipse")

     for nid, node in tree.nodes.items():
          prefix = "*" if node.is_hypothetical else ""
          label = f"{prefix}{nid}: {node.sequence}"
          dot.node(str(nid), label=label)

     for parent_id, child_id in tree.as_edge_list():
          dot.edge(str(parent_id), str(child_id))

     dot_filename = f"{output_basename}.gv"
     dot.render(dot_filename, cleanup=True)
     print(f"Graphviz files generated: '{dot_filename}' and '{output_basename}.pdf'")


def print_tree(tree: Tree) -> None:
     """
     Prints the tree in text form: node_id (Seq) → parent, children. Hypothetical nodes marked.
     """
     print("Tree structure (node_id: sequence) → [children_ids]")
     for nid, node in tree.nodes.items():
          parent = tree.parent_map.get(nid)
          children = tree.children_map.get(nid, [])
          hypo_flag = "(HYP)" if node.is_hypothetical else ""
          print(f"  {nid}{hypo_flag}: {node.sequence}   parent={parent}   children={children}")


# ---------------------------
#  Main Function
# ---------------------------

def main(config: Config) -> None:
     # 1) Read JSON
     raw_nodes = load_json_nodes(config.input_path)
     alive_sequences = filter_alive_nodes(raw_nodes)
     if not alive_sequences:
          print("No living nodes found in the file. Exiting.")
          return

     print(f"Number of living nodes: {len(alive_sequences)}")
     print("Starting evolutionary algorithm...")

     # 2) Run evolutionary strategy
     best_tree = evolutionary_algorithm(
          alive_sequences=alive_sequences,
          population_size=config.population_size,
          generations=config.generations,
          remove_fraction=config.remove_fraction,
          seed=config.seed
     )

     # 3) Output result as text
     print("\n=== Best reconstructed tree (text) ===")
     print_tree(best_tree)

     # 4) Visualize result with Graphviz
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
