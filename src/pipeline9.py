from typing import List
from abc import ABC, abstractmethod
import random
from graphviz import Digraph

class PipelineStep(ABC):
    @abstractmethod
    def process(self, s: str) -> str:
        pass
    def __repr__(self) -> str:
        return self.__class__.__name__

class ToLower(PipelineStep):
    def process(self, s: str) -> str:
        return s.lower()

class ToCapitalize(PipelineStep):
    def process(self, s: str) -> str:
        return s.capitalize()

class ReverseString(PipelineStep):
    def process(self, s: str) -> str:
        return s[::-1]

class RemoveLastChar(PipelineStep):
    def process(self, s: str) -> str:
        return s[:-1] if s else s

class SwapFirstLast(PipelineStep):
    def process(self, s: str) -> str:
        if len(s) < 2:
            return s
        return s[-1] + s[1:-1] + s[0]

class DoubleLastChar(PipelineStep):
    def process(self, s: str) -> str:
        if not s:
            return s
        return s + s[-1]

class LastBecomesFirst(PipelineStep):
    def process(self, s: str) -> str:
        if len(s) < 2:
            return s
        return s[-1] + s[:-1]

class MakeAppender(PipelineStep):
    def __init__(self, letter: str):
        self.letter = letter

    def process(self, s: str) -> str:
        return s + self.letter
    def __repr__(self) -> str:
        return f"MakeAppender({self.letter})"

class Pipeline(PipelineStep):
    def __init__(self, steps: List[PipelineStep]):
        self.original_steps = steps.copy()
        self.steps = steps.copy()

    def run_independent(self, s: str) -> str:
        result = None
        for step in self.steps:
            result = step.process(s)
            print(result)
        return result

    def run_chained(self, s: str) -> str:
        current = s
        for step in self.steps:
            current = step.process(current)
            print(current)
        return current

    def train_chain(self, source: str, target: str) -> List[PipelineStep]:
        attempts = 0
        while True:
            attempts += 1
            permuted = self.original_steps.copy()
            random.shuffle(permuted)
            result = source
            for step in permuted:
                result = step.process(result)
            if result == target:
                self.steps = permuted
                print(f"Gefundene Schrittfolge nach {attempts} Versuchen")
                return permuted
            print(f"Training: Im {attempts} Versuch lautet das Ergebnis: {result}.")

    def visualize(self, filename: str = None, format: str = 'png') -> Digraph:
        graph = Digraph(format=format)
        graph.attr('node', shape='box')

        def _add_steps(subg: Digraph, steps: List[PipelineStep], parent: str):
            with subg.subgraph(name=f'cluster_{parent}') as c:
                c.attr(label=parent)
                c.attr('node', shape='box')
                prev = None
                for idx, st in enumerate(steps):
                    node_id = f"{parent}_{idx}"
                    if isinstance(st, Pipeline):
                        _add_steps(c, st.steps, node_id)
                    else:
                        c.node(node_id, str(st))
                    if prev:
                        c.edge(prev, node_id)
                    prev = node_id

        _add_steps(graph, self.steps, 'Pipeline')

        if filename:
            graph.render(filename, cleanup=True)
        return graph

    def process(self, s: str) -> str:
        return self.run_chained(s)
    def __repr__(self) -> str:
        return " -> ".join([step.__class__.__name__ for step in self.steps])

if __name__ == "__main__":
    word = "BDR"
    target="Erdbeere"

    reverser = ReverseString()
    e_appender = MakeAppender('e')
    steps_Double_e = [
       e_appender, e_appender
    ]
    pipeline_Double_e = Pipeline(steps_Double_e)

    steps: List[PipelineStep] = [
          DoubleLastChar(),
          LastBecomesFirst(),
          e_appender, e_appender,
          pipeline_Double_e,
          reverser, reverser, reverser,
          ToCapitalize()
    ]

    pipeline = Pipeline(steps)

    print("\n=== Pipeline: verkettet ===")
    pipeline.run_chained(word)

    print("\n=== Pipeline: trainiert ===")
    pipeline.train_chain(word,target)
    pipeline.run_chained(word)
    pipeline.visualize("pipeline", format='png').view()
