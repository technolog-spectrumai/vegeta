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




# -- prompt-to-3D helpers -------------------------------------------------------------------------------------------
TWO_PART = '''
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Model(Design):
    parameters = [Parameter("gap", 0.0, "mm", min=0)]

    def build(self, p):
        base = cq.Workplane("XY").box(100, 60, 10, centered=(True, True, False))
        post = cq.Workplane("XY").box(20, 20, 50, centered=(True, True, False))
        assy = cq.Assembly(name="thing")
        assy.add(base, name="base", color=cq.Color(0.8, 0.1, 0.1))
        assy.add(post, name="post", loc=cq.Location((30, 0, 10 + p["gap"])), color=cq.Color(0.1, 0.2, 0.9))
        return assy
'''


def box_part(name="box", size=(10.0, 20.0, 30.0), offset=(0.0, 0.0, 0.0), color=(0.5, 0.5, 0.5, 1.0), brep=True):
    """A CadQuery-like part: a box with seam vertices repeated per face (as ``shape.tessellate`` gives them)."""
    import numpy as np
    import trimesh

    from vegeta.fidia.mesh import Part

    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.array(size) / 2 * [0, 0, 1] + np.array(offset))
    v = m.vertices[m.faces].reshape(-1, 3)
    t = np.arange(len(v)).reshape(-1, 3)
    b = {"valid": True, "n_solids": 1, "volume_mm3": float(np.prod(size))} if brep else {}
    return Part(name=name, color=tuple(color), vertices=v, triangles=t, brep=b, source_name=name)
