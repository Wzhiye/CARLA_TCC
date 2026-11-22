"""2D geometry helpers (shapely based) used by perception and scenarios."""

import math

from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import nearest_points


def xy(location):
    """carla.Location (or anything with x/y) -> (x, y) tuple."""
    return (location.x, location.y)


def planar_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def actor_footprint(actor, fallback_radius=0.3, inflate=0.0):
    """Ground-plane footprint polygon of an actor's bounding box.

    Some props report an empty bounding box; in that case a small disc around
    the actor location is used instead.
    """
    transform = actor.get_transform()
    box = actor.bounding_box
    if box.extent.x > 0.01 and box.extent.y > 0.01:
        vertices = box.get_world_vertices(transform)
        polygon = MultiPoint([(v.x, v.y) for v in vertices]).convex_hull
    else:
        polygon = Point(transform.location.x, transform.location.y).buffer(fallback_radius)
    if inflate > 0:
        polygon = polygon.buffer(inflate)
    return polygon


def polyline(points):
    """Build a LineString from (x, y) points, dropping consecutive duplicates."""
    cleaned = []
    for point in points:
        if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-3:
            cleaned.append(point)
    if len(cleaned) < 2:
        return None
    return LineString(cleaned)


def corridor(line, half_width):
    """A strip of +/- half_width around a polyline with flat ends."""
    return line.buffer(half_width, cap_style="flat", join_style="round")


def distance_along(line, shape):
    """Arc length along ``line`` of the point of ``shape`` closest to the line."""
    on_line, _ = nearest_points(line, shape)
    return line.project(on_line)


def lateral_offset(location, waypoint):
    """Signed lateral offset of ``location`` from a waypoint (positive = right)."""
    transform = waypoint.transform
    right = transform.get_right_vector()
    dx = location.x - transform.location.x
    dy = location.y - transform.location.y
    return dx * right.x + dy * right.y


def longitudinal_offset(location, origin, forward):
    """Signed distance of ``location`` ahead of ``origin`` along ``forward``."""
    return (location.x - origin.x) * forward.x + (location.y - origin.y) * forward.y


def front_point(transform, front_extent):
    """World (x, y) of the front bumper centre."""
    forward = transform.get_forward_vector()
    return (transform.location.x + forward.x * front_extent,
            transform.location.y + forward.y * front_extent)


class PathFrame:
    """A reference polyline with (s, d) coordinates.

    ``s`` is the arc length along the line and ``d`` the signed lateral
    distance, positive to the right (CARLA's right vector convention: for a
    heading (tx, ty) the right side is (-ty, tx)).
    """

    def __init__(self, points):
        self.line = polyline(points)
        if self.line is None:
            raise ValueError("a PathFrame needs at least two distinct points")
        self.length = self.line.length

    def _tangent(self, s):
        a = self.line.interpolate(max(0.0, s - 0.5))
        b = self.line.interpolate(min(self.length, s + 0.5))
        tx, ty = b.x - a.x, b.y - a.y
        norm = math.hypot(tx, ty) or 1.0
        return tx / norm, ty / norm

    def project(self, x, y):
        """World (x, y) -> (s, d)."""
        s = self.line.project(Point(x, y))
        q = self.line.interpolate(s)
        tx, ty = self._tangent(s)
        d = (x - q.x) * -ty + (y - q.y) * tx
        return s, d

    def point_at(self, s, d=0.0):
        """(s, d) -> world (x, y). ``s`` is clamped to the line."""
        s = min(max(s, 0.0), self.length)
        q = self.line.interpolate(s)
        tx, ty = self._tangent(s)
        return q.x - ty * d, q.y + tx * d

    def strip(self, s_from, s_to, d_center, half_width, step=1.0):
        """Polygon covering [s_from, s_to] around the lateral position ``d_center``."""
        s_from, s_to = max(0.0, s_from), min(self.length, s_to)
        if s_to - s_from < 0.1:
            return None
        points = []
        s = s_from
        while s < s_to:
            points.append(self.point_at(s, d_center))
            s += step
        points.append(self.point_at(s_to, d_center))
        line = polyline(points)
        return corridor(line, half_width) if line is not None else None


def slope_limited_envelope(intervals, slope, s_values):
    """Smallest non-negative profile u(s) that meets every ``(s0, s1, need)`` interval.

    Inside an interval u must be >= need; away from it u may fall off with at
    most ``slope`` metres per metre. Used to plan smooth lateral shifts.
    """
    profile = []
    for s in s_values:
        value = 0.0
        for s0, s1, need in intervals:
            gap = 0.0 if s0 <= s <= s1 else min(abs(s - s0), abs(s - s1))
            value = max(value, need - slope * gap)
        profile.append(value)
    return profile
