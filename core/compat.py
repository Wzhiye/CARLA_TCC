"""Thin compatibility layer between CARLA 0.9.16 and later releases.

CARLA 1.0 renamed a few things (map names, lane accessors, destroy semantics).
Everything version-sensitive that the demo touches goes through this module so
that a later migration only needs changes here.
"""

import logging

import carla

LOG = logging.getLogger(__name__)


def client_version(client):
    """Return the client library version string."""
    try:
        return client.get_client_version()
    except RuntimeError:
        return "unknown"


def server_version(client):
    """Return the server version string, or 'unknown' if it cannot be read."""
    try:
        return client.get_server_version()
    except RuntimeError:
        return "unknown"


def is_carla_1x(version_string):
    """True for CARLA 1.x version strings."""
    return version_string.startswith("1.")


def short_map_name(name):
    """'Carla/Maps/Town10HD_Opt' -> 'Town10HD_Opt'."""
    return name.replace("\\", "/").split("/")[-1]


def same_map(current_name, wanted_name):
    """Compare map names while ignoring the path prefix and letter case."""
    return short_map_name(current_name).lower() == short_map_name(wanted_name).lower()


def left_lane(waypoint):
    """Lane to the left of ``waypoint`` (None if there is none).

    In CARLA 1.0 this accessor was renamed; the old name still works there with
    a DeprecationWarning, so a single call site keeps the migration simple.
    """
    return waypoint.get_left_lane()


def right_lane(waypoint):
    """Lane to the right of ``waypoint`` (None if there is none)."""
    return waypoint.get_right_lane()


def same_direction(wp_a, wp_b):
    """True when two waypoints belong to lanes with the same driving direction."""
    return wp_a.lane_id * wp_b.lane_id > 0


def describe_side_lane(reference, side):
    """Classify the lane next to ``reference``.

    Returns one of: 'none', 'driving_same', 'driving_opposite', or the lane type
    name in lower case (e.g. 'shoulder', 'parking', 'sidewalk').
    """
    other = left_lane(reference) if side == "left" else right_lane(reference)
    if other is None:
        return "none"
    if other.lane_type == carla.LaneType.Driving:
        return "driving_same" if same_direction(reference, other) else "driving_opposite"
    return str(other.lane_type).lower()
