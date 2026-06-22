from __future__ import annotations

from paper import Paper, Segment, Vec2

BOUNDARY_POINT_SPACING = 0.2

PointKey = tuple[float, float]


def _point_key(x: float, y: float, digits: int = 6) -> PointKey:
    """Round a boundary point to a hashable key so coincident points collapse."""
    return (round(x, digits), round(y, digits))


def find_segments(paper: Paper) -> list[tuple[Segment, Paper]]:
    """
    Return every valid fold from a paper as a (segment, folded_paper) pair.
    """

    coords, idx = paper.compute_boundary_points(BOUNDARY_POINT_SPACING)
    side_count = len(idx) - 1
    if side_count < 2:
        return []

    keys = [_point_key(float(p[0]), float(p[1])) for p in coords]

    # Distinct point indices per boundary side, dropping coincident points.
    side_points: list[list[int]] = []
    for side in range(side_count):
        seen_on_side: set[PointKey] = set()
        unique: list[int] = []
        for i in range(int(idx[side]), int(idx[side + 1])):
            if keys[i] not in seen_on_side:
                seen_on_side.add(keys[i])
                unique.append(i)
        side_points.append(unique)

    base_layers = len(paper.layers)
    folds: list[tuple[Segment, Paper]] = []
    seen_creases: set[tuple[PointKey, PointKey]] = set()

    for side_1 in range(side_count):
        for side_2 in range(side_1 + 1, side_count):
            for i in side_points[side_1]:
                for j in side_points[side_2]:
                    if keys[i] == keys[j]:
                        continue
                    crease = (keys[i], keys[j]) if keys[i] <= keys[j] else (keys[j], keys[i])
                    if crease in seen_creases:
                        continue
                    seen_creases.add(crease)

                    a = Vec2(float(coords[i][0]), float(coords[i][1]))
                    b = Vec2(float(coords[j][0]), float(coords[j][1]))
                    _collect_fold(paper, a, b, base_layers, folds)
                    _collect_fold(paper, b, a, base_layers, folds)

    return folds


def _collect_fold(
    paper: Paper,
    p1: Vec2,
    p2: Vec2,
    base_layers: int,
    folds: list[tuple[Segment, Paper]],
) -> None:
    """
    Fold a copy of paper and keep it only if the fold added a layer.
    """
    try:
        segment = Segment(p1, p2)
    except ValueError:
        return
    folded = paper.copy()
    if folded.fold(segment) and len(folded.layers) > base_layers:
        folds.append((segment, folded))
