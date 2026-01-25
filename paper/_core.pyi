"""
This is a dummy file so Intellisense thinks everything is
in Python and functions correctly
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import List


class Vec2:
    """
    2d vector with basic vector arithmetic and utilities
    """

    x: float
    y: float
    
    def __init__(self, x: float, y: float) -> None: ...
    
    # Arithmetic
    def __add__(self, other: Vec2) -> Vec2: ...
    def __sub__(self, other: Vec2) -> Vec2: ...
    def __mul__(self, other: float) -> Vec2: ...
    def __rmul__(self, other: float) -> Vec2: ...
    def __truediv__(self, other: float) -> Vec2: ...
    
    # Utilities
    def dot(self, other: Vec2) -> float: ...
    def cross(self, other: Vec2) -> float: ...
    def norm(self) -> float: ...
    def normalized(self) -> Vec2: ...
    
    def __repr__(self) -> str: ...

class Segment:
    """
    Line segment defined by endpoints
    """

    p1: Vec2
    p2: Vec2
    
    def __init__(self, p1: Vec2, p2: Vec2) -> None: ...
    def __repr__(self) -> str: ...

class Layer:
    """
    One layer of paper, MUST be a convex counterclockwise polygon
    """

    @property
    def vertices(self) -> List[Vec2]: ...
    
    def __init__(self, vertices: List[Vec2]) -> None: ...
    def __repr__(self) -> str: ...

class Paper:
    """
    A single (possibly folded) sheet of paper
    """

    @property
    def layers(self) -> List[Layer]: ...
    
    def __init__(self) -> None: ...

    def copy(self) -> Paper: ...
    
    def fold(self, s: Segment) -> bool:
        """
        Fold the paper along a segment.
        Returns True if successful, False if fold causes a rip.
        """
        ...
    
    def compute_bounds(self) -> List[float]:
        """
        Computes the global bounds of the Paper.
        Returns [min_x, max_x, min_y, max_y]
        """
        ...
    
    def compute_boundary_points(self, max_dist: float) -> npt.NDArray[np.float64]:
        """
        Computes an unordered array of shape (N, 2) containing N points
        along the exterior boundary of this Paper separated by at most
        max_dist. The array is also guaranteed to contain all boundary
        vertices.
        Points on the same edge are evenly spaced but points on
        different edges might not have the same spacing

        Args:
            max_dist: the max distance that points may be spaced
        """
        ...

    def rasterize(self, rows: int, cols: int, theta: float = 0) -> npt.NDArray[np.bool_]:
        """
        Rasterize the paper to a numpy array (uint8) rotated by theta radians
        with shape (rows, cols)
        """
        ...
    
    def __repr__(self) -> str: ...
