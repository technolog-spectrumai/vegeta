"""The assembly base class and the two helpers every assembly uses to expose and pass on a component's parameters."""
from __future__ import annotations

from typing import Sequence

import cadquery as cq
from vegeta.dedalus import Design, Parameter


def component_parameters(design: type[Design], drop: Sequence[str] = ("part",)) -> list[Parameter]:
    """``design``'s parameters, less ``drop``: an assembly offers them as its own and hands them on with :func:`pick`."""
    return [q for q in design.parameters if q.name not in drop]


def pick(p: dict, design: type[Design], drop: Sequence[str] = ("part",), **fixed) -> dict:
    """The values of ``p`` that are ``design``'s parameters (less ``drop``), plus ``fixed``."""
    return {**{q.name: p[q.name] for q in component_parameters(design, drop)}, **fixed}


def turned(shape: cq.Shape, axis: str) -> cq.Shape:
    """A shape built along +Z (a propeller, a motor) turned to point along ``axis``: '+x', '-x', '+y', '-y', '+z', '-z'."""
    turns = {"+z": None, "-z": ((1, 0, 0), 180), "+x": ((0, 1, 0), 90), "-x": ((0, 1, 0), -90),
             "+y": ((1, 0, 0), -90), "-y": ((1, 0, 0), 90)}
    if axis not in turns:
        raise ValueError(f"axis must be one of {sorted(turns)}")
    if turns[axis] is None:
        return shape
    about, angle = turns[axis]
    return shape.rotate((0, 0, 0), about, angle)


class Assembly(Design):
    """Components arranged into a machine. Subclasses implement ``parts(p)``: the placed shapes by name."""

    def parts(self, p: dict) -> dict[str, cq.Shape]:  # pragma: no cover - abstract
        raise NotImplementedError

    def build(self, p):
        return cq.Compound.makeCompound(list(self.parts(p).values()))

    def _build_callable(self):
        return type(self).parts            # the arrangement is in parts(): its source is the design's identity

    def generate_parts(self, **overrides) -> dict[str, cq.Shape]:
        """The placed parts by name, for a notebook to colour, mesh or export one by one."""
        return self.parts(self.resolve(**overrides))

    def assembly(self, **overrides) -> cq.Assembly:
        """A CadQuery assembly with one named child per part (STEP export with names, ``show``)."""
        out = cq.Assembly(name=self.name)
        for name, shape in self.generate_parts(**overrides).items():
            out.add(shape, name=name)
        return out
