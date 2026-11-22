"""Route-relative placement: find where on a robotaxi's own route to inject a scenario.

Placing obstacles on the target vehicle's planned route (instead of at fixed
map coordinates) guarantees the encounter, works on any map and can be
repeated as often as needed.
"""

from dataclasses import dataclass

import carla

from ra_demo.core import compat, geometry


@dataclass
class Anchor:
    waypoint: carla.Waypoint      # lane centre at the start of the zone
    route_distance: float         # metres from the vehicle to the zone start along the route
    zone_length: float
    lane_width: float
    left_lane: str                # describe_side_lane() result at the zone start
    right_lane: str

    @property
    def start(self):
        return self.waypoint.transform.location

    @property
    def forward(self):
        return self.waypoint.transform.get_forward_vector()


def route_profile(vehicle, plan):
    """[(waypoint, cumulative distance from the vehicle)] for the planned route."""
    profile = []
    previous = vehicle.get_location()
    total = 0.0
    for waypoint, _ in plan:
        location = waypoint.transform.location
        total += geometry.planar_distance(previous, location)
        profile.append((waypoint, total))
        previous = location
    return profile


def side_ok(actual, requirement):
    if requirement == "any":
        return True
    if requirement == "driving":
        return actual in ("driving_same", "driving_opposite")
    return actual == requirement


def _stretch_ok(profile, start_index, begin, end):
    """No junction and no lane change on the route between ``begin`` and ``end`` metres."""
    anchor = profile[start_index][0]
    for waypoint, distance in profile:
        if distance < begin:
            continue
        if distance > end:
            break
        if waypoint.is_junction:
            return False
        if waypoint.road_id != anchor.road_id or waypoint.lane_id != anchor.lane_id:
            return False
    return True


def _lane_continuous(anchor_wp, length):
    """The lane itself (not only the route) continues unchanged for ``length`` metres."""
    step = 2.0
    current = anchor_wp
    travelled = 0.0
    while travelled < length:
        following = current.next(step)
        if not following:
            return False
        current = following[0]
        travelled += step
        if current.is_junction or current.road_id != anchor_wp.road_id \
                or current.lane_id != anchor_wp.lane_id:
            return False
    return True


def _lane_occupied(vehicle, profile, end_distance, lane_width, snapshot):
    """True if another vehicle is on the route between the ego and ``end_distance``."""
    points = [geometry.xy(vehicle.get_location())]
    for waypoint, distance in profile:
        points.append(geometry.xy(waypoint.transform.location))
        if distance > end_distance:
            break
    line = geometry.polyline(points)
    if line is None:
        return False
    strip = geometry.corridor(line, lane_width / 2.0)
    for other in snapshot.vehicles:
        if other.id == vehicle.id:
            continue
        if strip.intersects(geometry.actor_footprint(other)):
            return True
    for walker in snapshot.walkers:
        if strip.intersects(geometry.actor_footprint(walker, fallback_radius=0.4)):
            return True
    return False


def find_route_anchor(vehicle, plan, spec, snapshot):
    """Search the route for a valid zone start. Returns (Anchor or None, reason)."""
    trigger = spec["trigger"]
    placement = spec["placement"]
    min_d = float(trigger["min_distance_m"])
    max_d = float(trigger["max_distance_m"])
    zone = float(placement["zone_length_m"])
    before = float(placement["clear_before_m"])
    after = float(placement["clear_after_m"])

    profile = route_profile(vehicle, plan)
    if not profile:
        return None, "vehicle has no route"
    total = profile[-1][1]
    if total < min_d + zone + after:
        return None, "route too short ({:.0f} m left)".format(total)

    reason = "no straight lane section on the route"
    for index, (waypoint, distance) in enumerate(profile):
        if distance < min_d:
            continue
        if distance > max_d or distance + zone + after > total:
            break
        if waypoint.lane_type != carla.LaneType.Driving:
            continue
        if not _stretch_ok(profile, index, distance - before, distance + zone + after):
            continue
        if not _lane_continuous(waypoint, zone + after):
            continue
        left = compat.describe_side_lane(waypoint, "left")
        right = compat.describe_side_lane(waypoint, "right")
        if not side_ok(left, placement["left_lane"]) or not side_ok(right, placement["right_lane"]):
            reason = "side-lane requirement not met (left={}, right={})".format(left, right)
            continue
        if placement.get("require_empty_lane", True) and _lane_occupied(
                vehicle, profile, distance + zone + 5.0, waypoint.lane_width, snapshot):
            reason = "another road user is between the robotaxi and the zone"
            # Moving further away will not help: the blocker is in front of the ego.
            break
        return Anchor(waypoint, distance, zone, waypoint.lane_width, left, right), "ok"
    return None, reason


def find_junction_exit_anchor(vehicle, plan, spec, snapshot):
    """Anchor just after the next junction the route goes through (road closure, S3).

    The junction must start between ``min_distance_m`` and ``max_distance_m``
    ahead, the route must go through it as required (``route_through``) and it
    must offer the turns the operator is meant to use (``require_turn``).
    """
    from ra_demo.av.maneuvers import junction_exits, yaw_difference   # avoid an import cycle

    trigger = spec["trigger"]
    placement = spec["placement"]
    min_d = float(trigger["min_distance_m"])
    max_d = float(trigger["max_distance_m"])
    after = float(placement["after_junction_m"])
    zone = float(placement["zone_length_m"])
    clear_after = float(placement["clear_after_m"])

    profile = route_profile(vehicle, plan)
    if not profile:
        return None, "vehicle has no route"
    reason = "no suitable junction {:.0f}-{:.0f} m ahead".format(min_d, max_d)
    index = 0
    while index < len(profile):
        waypoint, distance = profile[index]
        if distance > max_d:
            break
        if not waypoint.is_junction:
            index += 1
            continue
        # A junction starts here; find where the route leaves it.
        start = index
        while index < len(profile) and profile[index][0].is_junction:
            index += 1
        if index >= len(profile):
            return None, "route ends inside the junction"
        if distance < min_d or start == 0:
            continue
        approach = profile[start - 1][0]
        exit_wp = profile[index][0]
        if placement["route_through"] == "straight" and abs(yaw_difference(
                exit_wp.transform.rotation.yaw, approach.transform.rotation.yaw)) > 30.0:
            reason = "the route turns at the next junction"
            continue
        exits = junction_exits(waypoint.get_junction(), approach)
        wanted = placement["require_turn"]
        has_left, has_right = bool(exits["left"]), bool(exits["right"])
        if (wanted == "any" and not (has_left or has_right)) or (wanted == "left" and not has_left) \
                or (wanted == "right" and not has_right) or (wanted == "both" and not (has_left and has_right)):
            reason = "junction has no {} turn".format(wanted if wanted != "any" else "left/right")
            continue
        following = exit_wp.next(after) if after > 0.01 else [exit_wp]
        if not following or following[0].is_junction:
            reason = "no straight road after the junction"
            continue
        anchor_wp = following[0]
        if not _lane_continuous(anchor_wp, zone + clear_after):
            reason = "road after the junction too short"
            continue
        spot = anchor_wp.transform.location
        if any(other.get_location().distance(spot) < 8.0 for other in snapshot.vehicles):
            reason = "a vehicle is where the closure would go"
            continue
        left = compat.describe_side_lane(anchor_wp, "left")
        right = compat.describe_side_lane(anchor_wp, "right")
        route_distance = profile[index][1] + after
        return Anchor(anchor_wp, route_distance, zone, anchor_wp.lane_width, left, right), "ok"
    return None, reason
