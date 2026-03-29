# `paper`

Python API for constructing, folding, and rasterizing 2D paper states.

## Classes

### `Vec2(x, y)`

Two-dimensional vector used by geometry APIs.

### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `x` | `float` | X coordinate. |
| `y` | `float` | Y coordinate. |

### Methods

>Supported operations:
>
>- Arithmetic: `+`, `-`, `*`, `/`
>- `dot(other) -> float`
>- `cross(other) -> float`
>- `norm() -> float`
>- `normalized() -> Vec2`

### `Segment(p1, p2)`

Line segment defined by two endpoints.

### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `p1` | `Vec2` | First endpoint. |
| `p2` | `Vec2` | Second endpoint. |

### `Layer(vertices)`

Single convex counterclockwise polygon layer.

### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `vertices` | `list[Vec2]` | Polygon vertices. |

### Attributes

| Name | Type | Description |
| --- | --- | --- |
| `vertices` | `list[Vec2]` | Layer vertices. |

### `Paper()`

Represents a sheet of paper, possibly with multiple folded layers.

### Attributes

| Name | Type | Description |
| --- | --- | --- |
| `layers` | `list[Layer]` | Current set of layers. |

### Methods

>Method summary:
>
>| Method | Description |
>| --- | --- |
>| `copy() -> Paper` | Returns a copy of the current paper state. |
>| `fold(s: Segment) -> bool` | Attempts one fold operation; returns `True` if valid and `False` otherwise. |
>| `compute_bounds() -> list[float]` | Returns `[min_x, max_x, min_y, max_y]`. |
>| `compute_boundary_points(max_dist: float) -> tuple[np.ndarray, np.ndarray]` | Samples boundary points and segment index offsets. |
>| `rasterize(rows: int, cols: int, theta: float = 0) -> np.ndarray` | Rasterizes paper occupancy into a boolean array with shape `(rows, cols)`. |

### Examples

```python
from paper import Paper, Segment, Vec2

p = Paper()
ok = p.fold(Segment(Vec2(0.0, 0.0), Vec2(1.0, 0.5)))
img = p.rasterize(64, 64)
print(ok, img.shape)
```

Expected output (example):

```text
True (64, 64)
```

## Notes

- This module is the stable Python import surface for simulation code.
- Internal C++ implementation details are intentionally not part of this reference.
