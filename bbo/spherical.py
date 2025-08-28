import numpy as np


def point_in_spherical_triangle(p, a, b, c, xp=np):
    def edge_test(v0, v1):
        return xp.dot(xp.cross(v0, v1), p.T)

    t0 = edge_test(a, b)
    t1 = edge_test(b, c)
    t2 = edge_test(c, a)

    inside = (t0 >= 0) & (t1 >= 0) & (t2 >= 0)
    return inside


def spherical_polygon_mask(curve, points, v0=None, xp=np):
    if v0 is None:
        v0 = curve[0]
        start_idx = 1
    else:
        start_idx = 0

    normal = xp.mean(xp.cross(curve, xp.roll(curve, -1, axis=0)), axis=0)
    if xp.dot(normal, v0) < 0:
        curve = curve[::-1]

    if not xp.allclose(curve[0], curve[-1]):
        curve = xp.vstack([curve, curve[0]])

    mask = xp.zeros(len(points), dtype=bool)

    for i in range(start_idx, len(curve) - 1):
        a, b, c = v0, curve[i], curve[i + 1]
        mask |= point_in_spherical_triangle(points, a, b, c)

    return mask