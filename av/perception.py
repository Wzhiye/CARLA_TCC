"""Privileged (ground-truth) perception for the robotaxi.

CARLA's BasicAgent only checks other *vehicles* for obstacles, so cones,
barriers and pedestrians would be driven through. This module reads the
simulator's ground truth and reports the closest static obstacle (props and
stalled vehicles with hazard lights) and the closest pedestrian inside a
corridor that follows the planned route.

During a maneuver (e.g. a bypass) the maneuver can install a *path override*:
the corridor then follows the planned maneuver path instead of the lane, and
vehicles are checked in that corridor too.
"""

from dataclasses import dataclass

import carla
from shapely.geometry import Point
from shapely.ops import substring

from ra_demo.core import geometry


@dataclass
class Obstacle:
    actor_id: int
    type_id: str
    category: str          # 'static' | 'walker' | 'vehicle'
    distance: float        # metres along the route, measured from the ego front bumper
    location: carla.Location


@dataclass
class PerceptionResult:
    static: Obstacle = None
    walker: Obstacle = None
    vehicle: Obstacle = None   # only filled while a path override is active
    corridor: object = None    # shapely polygon, kept for debug drawing


class PrivilegedPerception:
    """Route-corridor obstacle check against static props, pedestrians and (in maneuvers) vehicles."""

    def __init__(self, vehicle, cfg):
        self._vehicle = vehicle
        self.lookahead = float(cfg.get("lookahead_m", 50.0))
        self.margin = float(cfg.get("lateral_margin_m", 0.35))
        self.walker_margin = float(cfg.get("walker_margin_m", 0.3))
        self.prop_radius = float(cfg.get("prop_fallback_radius_m", 0.3))
        self.ignored_ids = set()
        self.path_override = None   # shapely LineString in world coordinates
        self._footprints = {}       # static props never move: cache their footprints
        self.half_width = vehicle.bounding_box.extent.y
        self._half_width = self.half_width + self.margin
        self.front_extent = vehicle.bounding_box.extent.x

    def ignore(self, actor_id, active=True):
        """Temporarily ignore an obstacle (used by bypass maneuvers)."""
        if active:
            self.ignored_ids.add(actor_id)
        else:
            self.ignored_ids.discard(actor_id)

    def clear_overrides(self):
        self.ignored_ids.clear()
        self.path_override = None

    # ----------------------------------------------------------- corridors
    def _route_line(self, plan):
        transform = self._vehicle.get_transform()
        forward = transform.get_forward_vector()
        front = transform.location + carla.Location(x=forward.x * self.front_extent,
                                                    y=forward.y * self.front_extent)
        if self.path_override is not None:
            start = self.path_override.project(Point(front.x, front.y))
            end = min(self.path_override.length, start + self.lookahead)
            if end - start < 0.5:
                return None, front
            line = substring(self.path_override, start, end)
            # Start the corridor at the bumper even if the car is a little off the path.
            return geometry.polyline([geometry.xy(front)] + list(line.coords)[1:]), front

        points = [geometry.xy(front)]
        travelled = 0.0
        previous = front
        for waypoint, _ in plan:
            location = waypoint.transform.location
            # Skip waypoints that are already behind the front bumper.
            if geometry.longitudinal_offset(location, front, forward) < 0 and travelled == 0.0:
                continue
            travelled += geometry.planar_distance(previous, location)
            points.append(geometry.xy(location))
            previous = location
            if travelled >= self.lookahead:
                break
        return geometry.polyline(points), front

    # -------------------------------------------------------------- update
    def update(self, snapshot, plan):
        result = PerceptionResult()
        line, front = self._route_line(plan)
        if line is None:
            return result
        static_corridor = geometry.corridor(line, self._half_width)
        walker_corridor = geometry.corridor(line, self._half_width + self.walker_margin)
        result.corridor = static_corridor
        search_radius = self.lookahead + 10.0

        statics = list(snapshot.props) + list(getattr(snapshot, "stalled", {}).values())
        for actor in statics:
            if actor.id in self.ignored_ids:
                continue
            location = actor.get_location()
            if geometry.planar_distance(location, front) > search_radius:
                continue
            is_prop = actor.type_id.startswith("static.")
            footprint = self._footprints.get(actor.id) if is_prop else None
            if footprint is None:
                footprint = geometry.actor_footprint(actor, fallback_radius=self.prop_radius)
                if is_prop:
                    self._footprints[actor.id] = footprint
            if static_corridor.intersects(footprint):
                distance = geometry.distance_along(line, footprint)
                if result.static is None or distance < result.static.distance:
                    result.static = Obstacle(actor.id, actor.type_id, "static", distance, location)

        for walker in snapshot.walkers:
            location = walker.get_location()
            if geometry.planar_distance(location, front) > search_radius:
                continue
            footprint = geometry.actor_footprint(walker, fallback_radius=0.4)
            if walker_corridor.intersects(footprint):
                distance = geometry.distance_along(line, footprint)
                if result.walker is None or distance < result.walker.distance:
                    result.walker = Obstacle(walker.id, walker.type_id, "walker", distance, location)

        # Vehicles in the corridor. During maneuvers this is the main vehicle check; otherwise
        # it backs up BasicAgent's lane-id based check, which misses a car standing just
        # past a road boundary (lane ids change sign there) or inside a junction.
        stalled = getattr(snapshot, "stalled", {})
        for other in snapshot.vehicles:
            if other.id == self._vehicle.id or other.id in self.ignored_ids or other.id in stalled:
                continue
            location = other.get_location()
            if geometry.planar_distance(location, front) > search_radius:
                continue
            footprint = geometry.actor_footprint(other)
            if static_corridor.intersects(footprint):
                distance = geometry.distance_along(line, footprint)
                if result.vehicle is None or distance < result.vehicle.distance:
                    result.vehicle = Obstacle(other.id, other.type_id, "vehicle", distance, location)

        # Forget cached footprints of props that no longer exist.
        if len(self._footprints) > len(snapshot.props) + 50:
            alive = {prop.id for prop in snapshot.props}
            self._footprints = {k: v for k, v in self._footprints.items() if k in alive}
        return result
