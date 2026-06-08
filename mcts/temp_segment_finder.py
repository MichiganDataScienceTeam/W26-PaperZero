from paper import Paper, Segment, Vec2
from typing import List, Tuple
import numpy as np

MAX_DIST = 0.2


def _point_key(point: np.ndarray, digits: int = 6) -> Tuple[float, float]:
    return (round(float(point[0]), digits), round(float(point[1]), digits))


def find_segments(paper: Paper) -> List[Segment]:
    coords, idx = paper.compute_boundary_points(MAX_DIST)
    side_count = len(idx) - 1
    if side_count < 2:
        return []

    base_layers = len(paper.layers)
    segments: List[Segment] = []
    seen_undirected: set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
    point_keys = [_point_key(point) for point in coords]

    side_points: List[List[int]] = []
    for side in range(side_count):
        start = int(idx[side])
        end = int(idx[side + 1])
        if start >= end:
            side_points.append([])
            continue

        unique: List[int] = []
        seen_side: set[Tuple[float, float]] = set()
        for point_idx in range(start, end):
            key = point_keys[point_idx]
            if key in seen_side:
                continue
            seen_side.add(key)
            unique.append(point_idx)
        side_points.append(unique)

    for side_1 in range(side_count):
        points_1 = side_points[side_1]
        if not points_1:
            continue

        for side_2 in range(side_1 + 1, side_count):
            points_2 = side_points[side_2]
            if not points_2:
                continue

            for i in points_1:
                p1 = coords[i]
                key_1 = point_keys[i]

                for j in points_2:
                    p2 = coords[j]
                    key_2 = point_keys[j]

                    if key_1 == key_2:
                        continue

                    undirected = (key_1, key_2) if key_1 <= key_2 else (key_2, key_1)
                    if undirected in seen_undirected:
                        continue

                    try:
                        seg_forward = Segment(Vec2(float(p1[0]), float(p1[1])), Vec2(float(p2[0]), float(p2[1])))
                        seg_reverse = Segment(Vec2(float(p2[0]), float(p2[1])), Vec2(float(p1[0]), float(p1[1])))
                    except Exception:
                        continue

                    folded_forward = paper.copy()
                    if not folded_forward.fold(seg_forward):
                        continue

                    if len(folded_forward.layers) == base_layers:
                        continue

                    seen_undirected.add(undirected)
                    segments.append(seg_forward)
                    segments.append(seg_reverse)

    return segments


if __name__ == "__main__":
    print(len(find_segments(Paper())))

    
