"""Turn scenario actor specs into concrete transforms along a lane.

Lane coordinates: ``s`` is the distance along the lane from the zone start
(negative = upstream), ``lateral`` the offset from the lane centre
(positive = right, as CARLA's right vector).

Patterns:
    single      one prop
    lane_block  cones closing one lane: taper, a line along the open side, an end row
    vehicle     a stalled vehicle (hazard lights, hand brake)
    road_block  barriers across every lane of this direction, cones in front
"""

import math
from dataclasses import dataclass

import carla

from ra_demo.core import compat


@dataclass
class PropPlacement:
    blueprints: list          # candidate blueprint ids, first available wins
    transform: carla.Transform
    role: str
    kind: str = "prop"        # 'prop' | 'vehicle'
    hazard_lights: bool = False


def _blueprint_list(actor_spec, key="blueprint", fallback_key="fallback_blueprints"):
    value = actor_spec[key]
    candidates = list(value) if isinstance(value, (list, tuple)) else [value]
    candidates += list(actor_spec.get(fallback_key, []))
    return candidates


def lane_transform(anchor_wp, s, lateral, z_offset=0.0, yaw_offset=0.0):
    """World transform at lane coordinates (s, lateral) relative to ``anchor_wp``."""
    if s > 0.01:
        waypoint = anchor_wp.next(s)[0]
    elif s < -0.01:
        waypoint = anchor_wp.previous(-s)[0]
    else:
        waypoint = anchor_wp
    base = waypoint.transform
    right = base.get_right_vector()
    location = carla.Location(base.location.x + right.x * lateral,
                              base.location.y + right.y * lateral,
                              base.location.z + z_offset)
    rotation = carla.Rotation(pitch=0.0, yaw=base.rotation.yaw + yaw_offset, roll=0.0)
    return carla.Transform(location, rotation)


def same_direction_lanes(waypoint):
    """The driving lanes of this road that carry traffic in the same direction (incl. ``waypoint``)."""
    lanes = [waypoint]
    for side in (compat.left_lane, compat.right_lane):
        current = waypoint
        while True:
            other = side(current)
            if other is None or other.lane_type != carla.LaneType.Driving \
                    or not compat.same_direction(waypoint, other):
                break
            lanes.append(other)
            current = other
    return lanes


def lane_block(anchor_wp, actor_spec, zone_length):
    """Cones that close the lane: a taper, a line along the open side and an end row.

    With ``open_side: left`` the taper starts at the right edge of the lane and
    moves to its left edge, guiding traffic into the lane on the left;
    ``open_side: right`` mirrors it.
    """
    length = float(actor_spec.get("length_m", zone_length))
    spacing = float(actor_spec.get("spacing_m", 2.0))
    line_spacing = float(actor_spec.get("line_spacing_m", 3.0))
    taper = max(spacing, float(actor_spec.get("taper_m", 6.0)))
    inset = float(actor_spec.get("edge_inset_m", 0.35))
    open_side = actor_spec.get("open_side", "left")
    blueprints = _blueprint_list(actor_spec)

    edge = anchor_wp.lane_width / 2.0 - inset
    far = -edge if open_side == "left" else edge     # lane line next to the open lane
    near = -far                                       # the other edge
    placements = []

    steps = max(1, int(math.ceil(taper / spacing)))
    for i in range(steps + 1):
        s = taper * i / steps
        lateral = near + (far - near) * (i / steps)
        placements.append(PropPlacement(blueprints, lane_transform(anchor_wp, s, lateral), "taper"))

    s = taper + line_spacing
    while s < length - 0.5:
        placements.append(PropPlacement(blueprints, lane_transform(anchor_wp, s, far), "line"))
        s += line_spacing

    if actor_spec.get("close_end", True):
        count = max(2, int(round(anchor_wp.lane_width / 1.0)))
        for i in range(count):
            lateral = far + (near - far) * (i / (count - 1))
            placements.append(PropPlacement(blueprints, lane_transform(anchor_wp, length, lateral), "end"))
    return placements


def single(anchor_wp, actor_spec):
    """One prop at (s_m, lateral_m) with optional yaw/z offsets."""
    transform = lane_transform(
        anchor_wp,
        float(actor_spec.get("s_m", 0.0)),
        float(actor_spec.get("lateral_m", 0.0)),
        z_offset=float(actor_spec.get("z_offset_m", 0.0)),
        yaw_offset=float(actor_spec.get("yaw_offset_deg", 0.0)),
    )
    return [PropPlacement(_blueprint_list(actor_spec), transform, actor_spec.get("role", "single"))]


def vehicle(anchor_wp, actor_spec):
    """A stalled vehicle in the lane (``lateral_m`` > 0 moves it towards the right edge)."""
    transform = lane_transform(
        anchor_wp,
        float(actor_spec.get("s_m", 0.0)),
        float(actor_spec.get("lateral_m", 0.0)),
        z_offset=float(actor_spec.get("z_offset_m", 0.3)),
        yaw_offset=float(actor_spec.get("yaw_offset_deg", 0.0)),
    )
    return [PropPlacement(_blueprint_list(actor_spec), transform, actor_spec.get("role", "vehicle"),
                          kind="vehicle", hazard_lights=bool(actor_spec.get("hazard_lights", True)))]


def road_block(anchor_wp, actor_spec):
    """Barriers across every same-direction lane at ``s_m`` and a row of cones in front."""
    s = float(actor_spec.get("s_m", 0.0))
    spacing = float(actor_spec.get("barrier_spacing_m", 1.7))
    upstream = float(actor_spec.get("cones_upstream_m", 2.5))
    blueprints = _blueprint_list(actor_spec)
    cones = _blueprint_list(actor_spec, "cone_blueprint", "cone_fallbacks") \
        if "cone_blueprint" in actor_spec else []
    placements = []
    for lane in same_direction_lanes(anchor_wp):
        half = lane.lane_width / 2.0
        count = max(1, int(round(lane.lane_width / spacing)))
        for i in range(count):
            lateral = -half + lane.lane_width * (i + 0.5) / count
            placements.append(PropPlacement(blueprints, lane_transform(lane, s, lateral, yaw_offset=90.0),
                                            "barrier"))
        if cones and upstream > 0.0:
            for lateral in (-half * 0.6, 0.0, half * 0.6):
                placements.append(PropPlacement(cones, lane_transform(lane, s - upstream, lateral), "cone"))
    return placements


def expand(anchor_wp, actor_specs, zone_length):
    """All placements for a scenario's actor list."""
    placements = []
    for actor_spec in actor_specs:
        pattern = actor_spec.get("pattern", "single")
        if pattern == "lane_block":
            placements += lane_block(anchor_wp, actor_spec, zone_length)
        elif pattern == "vehicle":
            placements += vehicle(anchor_wp, actor_spec)
        elif pattern == "road_block":
            placements += road_block(anchor_wp, actor_spec)
        else:
            placements += single(anchor_wp, actor_spec)
    return placements


def closed_lanes(anchor_wp, actor_specs):
    """Lanes the scenario closes (used by the lane guard)."""
    if any(actor.get("pattern") == "road_block" for actor in actor_specs):
        return same_direction_lanes(anchor_wp)
    return [anchor_wp]
