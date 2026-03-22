from paper import Paper, Vec2, Segment
from typing import List

MAX_DIST = 0.1

def find_segments(paper: Paper) -> List[Segment]:
    coords, _ = paper.compute_boundary_points(MAX_DIST)
    ans = []

    for boundary in coords:
        for boundary2 in coords:
            try:
                seg = Segment(Vec2(boundary[0], boundary[1]), Vec2(boundary2[0], boundary2[1]))
            except:
                continue
            c = paper.copy()
            if c.fold(seg):
                ans.append(seg)
            break
    return ans
    

if __name__ == "__main__":
    print(find_segments(Paper()))

    