"""Shared test helpers (imported by tests via sys.path set in conftest)."""

DESIGN = '''
import cadquery as cq
from vegeta.dedalus import Design, Parameter

class Plate(Design):
    parameters = [Parameter("width", 30.0, "mm", min=5), Parameter("thickness", 2.0, "mm", min=0.5)]
    def build(self, p):
        return cq.Workplane().box(p["width"], 10, p["thickness"])
'''

DESIGN_WITH_HOLE = DESIGN.replace(
    'Parameter("thickness", 2.0, "mm", min=0.5)]',
    'Parameter("thickness", 2.0, "mm", min=0.5),\n                  Parameter("hole", 4.0, "mm", min=0.5)]'
).replace('return cq.Workplane().box(p["width"], 10, p["thickness"])',
          'return cq.Workplane().box(p["width"], 10, p["thickness"]).faces(">Z").workplane().hole(p["hole"])')


class FakeProposer:
    """Scripted proposer: returns the queued proposal dicts in order; records what it was asked."""

    name = "fake"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def describe(self):
        return {"provider": "fake", "model": "fake-1"}

    def propose(self, context, instruction, history):
        self.calls.append({"context": context, "instruction": instruction, "history": list(history)})
        data = self.answers.pop(0)
        return data, {"input_tokens": 1, "output_tokens": 1, "model": "fake-1"}


def answer(kind, source=None, parameters=None, summary="s", rationale="r"):
    return {"kind": kind, "summary": summary, "rationale": rationale, "source": source,
            "parameters": parameters or {}, "expected_effects": ["e"], "risks": []}


