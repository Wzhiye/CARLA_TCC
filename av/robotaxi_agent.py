"""Robotaxi driving agent built on CARLA's BasicAgent.

Differences from BasicAgent:
* uses the shared per-tick ActorSnapshot instead of querying the world itself;
* adds privileged perception for static props and pedestrians;
* plans a smooth stop in front of obstacles and at the end of the route
  instead of BasicAgent's binary emergency brake;
* exposes the current hazard so the robotaxi supervisor can decide when to
  ask for remote assistance;
* offers hooks for maneuvers: lateral offset, speed cap and a path override
  for perception.
"""

import math
from dataclasses import dataclass

from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import get_speed

from ra_demo.av.perception import PrivilegedPerception
from ra_demo.core import geometry

HAZARD_NONE = "NONE"
HAZARD_RED_LIGHT = "RED_LIGHT"
HAZARD_STATIC = "STATIC_OBSTACLE"
HAZARD_WALKER = "WALKER"
HAZARD_VEHICLE = "VEHICLE"

_BASE_MIN_DISTANCE = 3.0   # LocalPlanner default


def _continue_from(last, leg, near=15.0):
    """``leg`` without its first waypoints that lie behind ``last``.

    A global-planner leg may end with a lane change that jumps past the via
    point; the next leg then starts a few metres back, which would make the car
    turn around. Waypoints close to ``last`` and not ahead of it are dropped.
    """
    origin = last.transform.location
    forward = last.transform.get_forward_vector()
    index = 0
    while index < len(leg):
        location = leg[index][0].transform.location
        if geometry.planar_distance(location, origin) > near \
                or geometry.longitudinal_offset(location, origin, forward) > 1.0:
            break
        index += 1
    return leg[index:]


def _is_lane_change(first, second):
    """True if the route goes from ``first`` to ``second`` by changing lanes."""
    if first.road_id == second.road_id:
        return first.lane_id != second.lane_id
    # Road boundary: a lane change unless ``second`` is simply the next stretch of the lane.
    gap = max(0.5, first.transform.location.distance(second.transform.location))
    return not any((after.road_id, after.lane_id) == (second.road_id, second.lane_id)
                   for after in first.next(gap))


def _lane_samples(target, back=25.0, ahead=20.0, step=2.0, limit=150):
    """Waypoints along the target lane: back through every predecessor (also inside a
    junction) and ahead along the lane."""
    samples = [target]
    frontier = [(target, 0.0)]
    while frontier and len(samples) < limit:
        waypoint, distance = frontier.pop()
        if distance >= back:
            continue
        for before in waypoint.previous(step):
            samples.append(before)
            frontier.append((before, distance + step))
    current, distance = target, 0.0
    while distance < ahead:
        following = current.next(step)
        if not following:
            break
        current = following[0]
        samples.append(current)
        distance += step
    return samples


def _in_lane(transform, samples):
    """True if a vehicle (its transform) drives in the lane described by ``samples``."""
    location = transform.location
    for waypoint in samples:
        centre = waypoint.transform.location
        if abs(centre.x - location.x) > 3.0 or abs(centre.y - location.y) > 3.0:
            continue
        if geometry.planar_distance(centre, location) < waypoint.lane_width / 2.0 - 0.2:
            yaw = (transform.rotation.yaw - waypoint.transform.rotation.yaw + 180.0) % 360.0 - 180.0
            if abs(yaw) < 45.0:
                return True
    return False


@dataclass
class Hazard:
    kind: str = HAZARD_NONE
    actor_id: int = None
    type_id: str = ""
    distance: float = -1.0


class RobotaxiAgent(BasicAgent):
    """BasicAgent + privileged perception + smooth stopping + maneuver hooks."""

    def __init__(self, vehicle, cfg, fixed_delta, map_inst, grp_inst):
        driving = cfg.get("driving", {})
        self.cruise_speed = float(driving.get("target_speed_kmh", 30.0))
        self.use_speed_limits = bool(driving.get("follow_speed_limits", True))
        self.comfort_decel = float(driving.get("comfort_decel_mps2", 2.5))
        self.standoff = float(driving.get("obstacle_standoff_m", 6.0))
        # Wait further back behind a stalled vehicle: a solid obstacle needs a longer
        # run-up to steer around than a cone taper that already guides the car aside.
        self.vehicle_standoff = float(driving.get("stalled_vehicle_standoff_m", 10.0))
        self.walker_standoff = float(driving.get("walker_standoff_m", 5.0))
        self.stop_buffer = float(driving.get("route_end_buffer_m", 1.0))

        opt_dict = {
            "dt": fixed_delta,
            "max_brake": float(driving.get("max_brake", 0.6)),
            "base_vehicle_threshold": float(driving.get("base_vehicle_threshold_m", 6.0)),
            "base_tlight_threshold": float(driving.get("base_tlight_threshold_m", 5.0)),
            "detection_speed_ratio": float(driving.get("detection_speed_ratio", 1.0)),
            "sampling_resolution": float(driving.get("sampling_resolution_m", 2.0)),
        }
        super().__init__(vehicle, target_speed=self.cruise_speed, opt_dict=opt_dict,
                         map_inst=map_inst, grp_inst=grp_inst)
        self.perception = PrivilegedPerception(vehicle, cfg.get("perception", {}))
        self.base_lookahead = self.perception.lookahead
        self.hazard = Hazard()
        self.last_perception = None
        self.speed_mps = 0.0
        self.current_offset = 0.0
        self.speed_cap_kmh = None      # set by maneuvers and RA events
        self.stop_at_route_end = True  # maneuvers that end mid-road keep this on
        # Distances (from the front bumper, along the route) to the next junction's
        # entry and exit; None when there is no junction within the look-ahead.
        self.junction_entry = None
        self.junction_exit = None
        # True when the car holds at a junction entry because the obstacle lies
        # beyond the junction (a closed road).
        self.obstacle_beyond_junction = False

    # ------------------------------------------------------------- helpers
    def vehicle(self):
        return self._vehicle

    def world_map(self):
        return self._map

    def route_planner(self):
        return self._global_planner

    def route_length_ahead(self, limit=60.0):
        """Remaining route length (capped at ``limit``) from the vehicle position.

        The first stretch is measured along the lane (not straight-line) so that a
        car driving with a lateral offset still sees the remaining distance reach 0.
        """
        vehicle_location = self._vehicle.get_location()
        total = 0.0
        previous = None
        for waypoint, _ in self._local_planner.get_plan():
            location = waypoint.transform.location
            if previous is None:
                forward = waypoint.transform.get_forward_vector()
                total = max(0.0, geometry.longitudinal_offset(location, vehicle_location, forward))
            else:
                total += geometry.planar_distance(previous, location)
            previous = location
            if total >= limit:
                return limit
        return total

    def has_route(self):
        return not self._local_planner.done()

    def plan_route(self, end_location, via=()):
        """Global route from just ahead of the car through ``via`` to ``end_location``.

        Raises networkx.NetworkXNoPath when a leg cannot be routed.
        """
        target = self._local_planner.target_waypoint
        start = target.transform.location if target is not None else self._vehicle.get_location()
        points = [start] + list(via) + [end_location]
        route = []
        for begin, end in zip(points, points[1:]):
            leg = self.trace_route(self._map.get_waypoint(begin), self._map.get_waypoint(end))
            route.extend(_continue_from(route[-1][0], leg) if route else leg)
        return route

    def set_route(self, route):
        self._local_planner.set_global_plan(route, clean_queue=True)

    @staticmethod
    def route_length(route):
        total = 0.0
        for (first, _), (second, _) in zip(route, route[1:]):
            total += geometry.planar_distance(first.transform.location, second.transform.location)
        return total

    def plan(self):
        return list(self._local_planner.get_plan())

    def set_lateral_offset(self, offset, actual_offset=None):
        """Drive ``offset`` metres to the right (negative: left) of the lane centre.

        ``actual_offset`` is where the car is right now. Lane-centre waypoints are
        only consumed when the car comes within the planner's minimum distance of
        them, so that distance must exceed the car's real lateral offset; keeping
        it just above (instead of above the *commanded* offset) keeps the steering
        look-ahead short and the lateral response quick.
        """
        self.current_offset = offset
        self._offset = offset
        self._local_planner.set_offset(offset)
        reference = abs(offset) if actual_offset is None else abs(actual_offset)
        self._local_planner._base_min_distance = max(_BASE_MIN_DISTANCE, reference + 1.5)

    def reset_maneuver_hooks(self, keep_offset=False):
        if not keep_offset:
            self.set_lateral_offset(0.0)
        self.speed_cap_kmh = None
        self.perception.clear_overrides()

    def _scan_junction(self, limit):
        """Distances from the front bumper to the next junction entry/exit on the plan."""
        self.junction_entry = None
        self.junction_exit = None
        here = self._vehicle.get_location()
        if self._map.get_waypoint(here).is_junction:
            return
        front = self._vehicle.bounding_box.extent.x
        travelled = None
        previous = None
        for waypoint, _ in self._local_planner.get_plan():
            location = waypoint.transform.location
            if previous is None:
                forward = waypoint.transform.get_forward_vector()
                travelled = geometry.longitudinal_offset(location, here, forward)
            else:
                travelled += geometry.planar_distance(previous, location)
            previous = location
            if travelled - front > limit:
                return
            if waypoint.is_junction and self.junction_entry is None:
                self.junction_entry = max(0.0, travelled - front)
            elif not waypoint.is_junction and self.junction_entry is not None:
                self.junction_exit = travelled - front
                return

    def _junction_lookahead(self):
        """Perception range: the base look-ahead, stretched across a junction ahead.

        A closure right behind a long junction must be seen before the car
        enters the junction (it has to wait at the entry), so the corridor
        reaches 25 m past the exit of the next junction.
        """
        base = self.base_lookahead
        self._scan_junction(base)
        if self.junction_entry is None:
            return base
        if self.junction_exit is None:
            self._scan_junction(base + 80.0)
        if self.junction_exit is None:
            return base + 80.0
        return max(base, self.junction_exit + 25.0)

    def _lane_change_conflict(self, snapshot, horizon=40.0):
        """A vehicle in the lane the route changes into, too close to merge in front of.

        BasicAgent follows the route's lane changes without looking at the target
        lane (the global planner often puts one right after a junction). Returns
        (vehicle, longitudinal offset of that vehicle) for the nearest vehicle in
        the target lane that is moving between 8 m behind the car and 15 m past
        the lane change, or standing within 8 m of it; None if there is none.
        The car then waits until it can merge in behind it.
        """
        here = self._vehicle.get_location()
        target = None
        change_at = 0.0
        # Start from the lane the car is in: a new trip may begin with a lane change.
        previous_wp, previous, travelled = self._map.get_waypoint(here), here, 0.0
        for waypoint, _ in self._local_planner.get_plan():
            location = waypoint.transform.location
            travelled += geometry.planar_distance(previous, location)
            previous = location
            if travelled > horizon:
                return None
            if previous_wp is not None and not waypoint.is_junction and not previous_wp.is_junction \
                    and _is_lane_change(previous_wp, waypoint):
                target, change_at = waypoint, travelled
                break
            previous_wp = waypoint
        if target is None:
            return None
        samples = _lane_samples(target)
        forward = self._vehicle.get_transform().get_forward_vector()
        stalled = getattr(snapshot, "stalled", {})
        best = None
        for other in snapshot.vehicles:
            if other.id == self._vehicle.id or other.id in stalled:
                continue   # a broken-down car is a static obstacle for perception
            location = other.get_location()
            if abs(location.x - here.x) > horizon + 20.0 or abs(location.y - here.y) > horizon + 20.0:
                continue
            offset = geometry.longitudinal_offset(location, here, forward)
            if other.get_velocity().length() > 0.5:
                if not -8.0 <= offset <= change_at + 15.0:
                    continue
            elif not change_at - 8.0 <= offset <= change_at + 8.0:
                continue   # a standing car further on is followed normally once in its lane
            if not _in_lane(other.get_transform(), samples):
                continue
            if best is None or offset < best[1]:
                best = (other, offset)
        return best

    def front_location(self):
        transform = self._vehicle.get_transform()
        return geometry.front_point(transform, self._vehicle.bounding_box.extent.x)

    def _stop_speed_kmh(self, gap):
        """Speed (km/h) from which the car can still stop within ``gap`` metres."""
        if gap <= 0.0:
            return 0.0
        return math.sqrt(2.0 * self.comfort_decel * gap) * 3.6

    # ------------------------------------------------------------ run_step
    def run_step(self, snapshot=None):  # pylint: disable=arguments-differ
        """Compute one control step. ``snapshot`` is the shared ActorSnapshot."""
        if snapshot is None:
            return super().run_step()

        self.speed_mps = get_speed(self._vehicle) / 3.6
        hazard = Hazard()
        maneuvering = self.perception.path_override is not None

        # 1) Static props, pedestrians and (during maneuvers) vehicles in the corridor.
        self.perception.lookahead = self.base_lookahead if maneuvering else self._junction_lookahead()
        perception = self.perception.update(snapshot, self._local_planner.get_plan())
        self.last_perception = perception

        # 2) Vehicles: BasicAgent's lane logic normally, the maneuver corridor otherwise.
        vehicle_found, blocker, vehicle_distance = False, None, -1.0
        if maneuvering:
            if perception.vehicle is not None and perception.vehicle.distance < self.standoff + 2.0:
                vehicle_found = True
                blocker = snapshot.vehicles_by_id.get(perception.vehicle.actor_id)
                vehicle_distance = perception.vehicle.distance
        else:
            max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * self.speed_mps
            stalled = getattr(snapshot, "stalled", {})
            others = [v for v in snapshot.vehicles
                      if v.id != self._vehicle.id and v.id not in self.perception.ignored_ids
                      and v.id not in stalled]
            vehicle_found, blocker, vehicle_distance = self._vehicle_obstacle_detected(
                others, max_vehicle_distance)

        # 3) Red traffic lights.
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * self.speed_mps
        light_found, light = self._affected_by_traffic_light(snapshot.traffic_lights,
                                                             max_tlight_distance)

        # Longitudinal planning: the lowest speed that still allows a comfortable stop.
        cruise = self.cruise_speed
        if self.use_speed_limits:
            limit = self._vehicle.get_speed_limit()
            if limit and limit > 0:
                cruise = min(cruise, limit)
        if self.speed_cap_kmh is not None:
            cruise = min(cruise, self.speed_cap_kmh)
        allowed = cruise
        emergency = False

        self.obstacle_beyond_junction = False
        if perception.static is not None:
            standoff = self.vehicle_standoff if perception.static.type_id.startswith("vehicle.") \
                else self.standoff
            gap = perception.static.distance - standoff
            if not maneuvering:
                self._scan_junction(perception.static.distance)
                if self.junction_entry is not None and self.junction_entry < perception.static.distance:
                    room_after = perception.static.distance - (self.junction_exit or perception.static.distance)
                    # Never stop inside a junction: if the car would not fit between the
                    # junction exit and the obstacle, wait at the junction entry instead.
                    if room_after < standoff + 2.0 * self._vehicle.bounding_box.extent.x + 1.0:
                        gap = min(gap, self.junction_entry - 1.0)
                        self.obstacle_beyond_junction = True
            allowed = min(allowed, self._stop_speed_kmh(gap))
            emergency = emergency or gap <= 0.3
            hazard = Hazard(HAZARD_STATIC, perception.static.actor_id,
                            perception.static.type_id, perception.static.distance)

        if perception.walker is not None:
            gap = perception.walker.distance - self.walker_standoff
            allowed = min(allowed, self._stop_speed_kmh(gap))
            if gap <= 0.3:
                emergency = True
            if hazard.kind == HAZARD_NONE or perception.walker.distance < hazard.distance:
                hazard = Hazard(HAZARD_WALKER, perception.walker.actor_id,
                                perception.walker.type_id, perception.walker.distance)

        if maneuvering and perception.vehicle is not None:
            allowed = min(allowed, self._stop_speed_kmh(perception.vehicle.distance - self.standoff))

        if not maneuvering and perception.vehicle is not None:
            # Car-following backup: keep 4 m to any vehicle in the route corridor.
            ahead = perception.vehicle
            if ahead.distance < self._base_vehicle_threshold + self._speed_ratio * self.speed_mps + 10.0:
                gap = ahead.distance - 4.0
                allowed = min(allowed, self._stop_speed_kmh(gap))
                emergency = emergency or gap <= 0.3
                if hazard.kind == HAZARD_NONE or ahead.distance < hazard.distance:
                    hazard = Hazard(HAZARD_VEHICLE, ahead.actor_id, ahead.type_id, ahead.distance)

        if not maneuvering:
            conflict = self._lane_change_conflict(snapshot)
            if conflict is not None:
                # Let it pass (or keep clear of it) and merge in behind it: 10 m back.
                other, offset = conflict
                gap = offset - 10.0
                allowed = min(allowed, self._stop_speed_kmh(gap))
                emergency = emergency or gap <= 0.3
                if hazard.kind == HAZARD_NONE:
                    hazard = Hazard(HAZARD_VEHICLE, other.id, other.type_id, max(0.0, offset))

        if self.stop_at_route_end:
            remaining = self.route_length_ahead()
            if remaining < 60.0:
                allowed = min(allowed, self._stop_speed_kmh(remaining - self.stop_buffer))
                if remaining - self.stop_buffer <= 0.3:
                    emergency = True

        if vehicle_found and blocker is not None:
            emergency = True
            if hazard.kind == HAZARD_NONE or vehicle_distance < hazard.distance:
                hazard = Hazard(HAZARD_VEHICLE, blocker.id, blocker.type_id, vehicle_distance)

        if light_found:
            emergency = True
            # A red light explains the stop; it is never a reason to ask for help.
            hazard = Hazard(HAZARD_RED_LIGHT, light.id if light else None, "traffic_light", -1.0)

        self.hazard = hazard
        # Keep creeping at a minimum speed while far from the stop point so the PID
        # does not stall a few metres short of it.
        if 0.0 < allowed < 4.0 and not emergency:
            allowed = 4.0
        self._local_planner.set_speed(allowed)
        control = self._local_planner.run_step()
        if emergency:
            control = self.add_emergency_stop(control)
        return control
