import cppimport
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import like this to make static analyzers behave
    from paper._core import Vec2, Segment, Layer, Paper
else:
    # Import like this to force cppimport to behave
    cppimport.force_rebuild(True)
    _core = cppimport.imp("paper._core")
    Vec2 = _core.Vec2
    Segment = _core.Segment
    Layer = _core.Layer
    Paper = _core.Paper
