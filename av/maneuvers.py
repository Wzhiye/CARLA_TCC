"""Remote-assistance maneuvers executed by a robotaxi after an operator command.

The operator only chooses *what* to do. Each maneuver checks on the vehicle
side whether it is possible (geometry, lanes, junctions) and safe (gap in
traffic) before and while it drives; it may reject a command with a reason.

Every maneuver has a side-effect-free ``check(snapshot)`` (used to tell the
operator up front which options are feasible, and why not) and a ``start()``
that commits to it.

    BypassManeuver   pass an obstacle on the left or right by shifting laterally
                     (into a same-direction lane, the opposite lane or a shoulder)
    TurnManeuver     re-route through the next junction to the left or right
    PullOverManeuver stop at the curb (or secure in place when there is no room)
    ResumeManeuver   leave the curb / return to the lane centre and continue
"""

import logging

import carla
import networkx
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

from ra_demo.av.robotaxi_agent import HAZARD_STATIC, HAZARD_VEHICLE, HAZARD_WALKER
from ra_demo.core import compat, geometry

LOG = logging.getLogger(__name__)

RUNNING = "RUNNING"
DONE = "DONE"
FAILED = "FAILED"

PHASE_WAIT_GAP = "WAIT_GAP"
PHASE_EXECUTE = "EXECUTE"

GONE = "the obstacle is no longer there"


class ManeuverRejected(Exception):
    """Raised by check()/start() when the vehicle refuses the command."""


# --------------------------------------------------------------------- helpers
def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def yaw_difference(to_yaw, from_yaw):
    """Signed heading change in degrees, in [-180, 180). Positive = right turn."""
    return (to_yaw - from_yaw + 180.0) % 360.0 - 180.0


def obstacle_cluster(snapshot, seed_ids, exclude_ids=(), link_m=6.0):
    """The blocking actor(s) plus every static obstacle chained to them within ``link_m``.

    Returns {actor_id: footprint polygon}. A cone line or a work zone is one
    cluster; a broken-down van with a warning sign behind it is another.
    """
    if isinstance(seed_ids, int):
        seed_ids = [seed_ids]
    chainable = dict(snapshot.props_by_id) if hasattr(snapshot, "props_by_id") \
        else {prop.id: prop for prop in snapshot.props}
    chainable.update(getattr(snapshot, "stalled", {}))
    elements = {}
    for seed_id in seed_ids:
        if seed_id in exclude_ids:
            continue
        seed = chainable.get(seed_id) or snapshot.vehicles_by_id.get(seed_id)
        if seed is not None:
            elements[seed_id] = (seed, geometry.actor_footprint(seed))
    frontier = list(elements)
    while frontier:
        current_actor, current_poly = elements[frontier.pop()]
        current_location = current_actor.get_location()
        for actor_id, actor in chainable.items():
            if actor_id in elements or actor_id in exclude_ids:
                continue
            if actor.get_location().distance(current_location) > link_m + 12.0:
                continue
            footprint = geometry.actor_footprint(actor)
            if footprint.distance(current_poly) <= link_m:
                elements[actor_id] = (actor, footprint)
                frontier.append(actor_id)
    return {actor_id: poly for actor_id, (_, poly) in elements.items()}


def lane_walk(waypoint, distance, forward=True, step=4.0):
    """(x, y) points along a lane: following its direction (``forward``) or against it."""
    points = [geometry.xy(waypoint.transform.location)]
    current = waypoint
    travelled = 0.0
    while travelled < distance:
        following = current.next(step) if forward else current.previous(step)
        if not following:
            break
        current = following[0]
        points.append(geometry.xy(current.transform.location))
        travelled += step
    return points


def route_lane_waypoint(agent):
    """Waypoint of the *route's* lane next to the car.

    After an aborted bypass the car may stand in the neighbouring lane; decisions
    must still be made relative to the lane the route follows.
    """
    vehicle_location = agent.vehicle().get_location()
    plan = agent.plan()
    if plan:
        first = plan[0][0]
        forward = first.transform.get_forward_vector()
        ahead = geometry.longitudinal_offset(first.transform.location, vehicle_location, forward)
        if ahead > 0.5:
            behind = first.previous(ahead)
            if behind:
                return behind[0]
        return first
    return agent.world_map().get_waypoint(vehicle_location)


def reference_points(agent, behind_m=15.0):
    """Lane-centre points from ``behind_m`` behind the car along its plan ahead."""
    vehicle = agent.vehicle()
    location = vehicle.get_location()
    waypoint = route_lane_waypoint(agent)
    points = []
    distance = behind_m
    while distance > 0.5:
        previous = waypoint.previous(distance)
        if previous:
            points.append(geometry.xy(previous[0].transform.location))
        distance -= 3.0
    points.append(geometry.xy(waypoint.transform.location))
    forward = vehicle.get_transform().get_forward_vector()
    for plan_wp, _ in agent.plan():
        plan_location = plan_wp.transform.location
        if geometry.longitudinal_offset(plan_location, location, forward) <= 0.5:
            continue
        points.append(geometry.xy(plan_location))
    return points


def next_junction_distance(agent, limit=150.0):
    """(distance from the car to the first junction waypoint on the plan, index) or (None, None)."""
    previous = agent.vehicle().get_location()
    travelled = 0.0
    for index, (waypoint, _) in enumerate(agent.plan()):
        location = waypoint.transform.location
        travelled += geometry.planar_distance(previous, location)
        previous = location
        if travelled > limit:
            break
        if waypoint.is_junction:
            return travelled, index
    return None, None


def junction_exits(junction, approach, min_angle=35.0):
    """{'left'|'right'|'straight': [(distance to our lane, exit waypoint), ...]} reachable from ``approach``.

    ``junction.get_waypoints()`` returns waypoints of the junction's internal
    connecting lanes (their road ids differ from the incoming road), so entries
    are matched to the approach by position and heading, not by road id.
    """
    end = approach.transform.location
    approach_yaw = approach.transform.rotation.yaw
    exits = {"left": [], "right": [], "straight": []}
    for entry, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
        distance = entry.transform.location.distance(end)
        if distance > 12.0:
            continue
        if abs(yaw_difference(entry.transform.rotation.yaw, approach_yaw)) > 35.0:
            continue
        diff = yaw_difference(exit_wp.transform.rotation.yaw, entry.transform.rotation.yaw)
        if diff < -min_angle:
            exits["left"].append((distance, exit_wp))
        elif diff > min_angle:
            exits["right"].append((distance, exit_wp))
        else:
            exits["straight"].append((distance, exit_wp))
    for values in exits.values():
        values.sort(key=lambda item: item[0])
    return exits


def blocking_hazard(agent):
    hazard = agent.hazard
    if hazard.kind in (HAZARD_STATIC, HAZARD_VEHICLE, HAZARD_WALKER) and hazard.distance >= 0:
        return hazard
    return None


class ManeuverContext:
    """What a maneuver may use from its robotaxi."""

    def __init__(self, agent, cfg, destination, fleet_ids):
        self.agent = agent
        self.vehicle = agent.vehicle()
        self.map = agent.world_map()
        self.cfg = cfg
        self.destination = destination
        self.fleet_ids = set(fleet_ids)


# ------------------------------------------------------------------ base class
class Maneuver:
    action = ""

    def __init__(self, ctx):
        self.ctx = ctx
        self.agent = ctx.agent
        self.cfg = ctx.cfg
        self.status = RUNNING
        self.phase = PHASE_EXECUTE
        self.note = ""
        self.started_at = None
        self.phase_since = None
        self._stuck_since = None
        self.preview = float(self.cfg.get("preview_m", 5.0))
        self.half_width = self.agent.perception.half_width
        self.length = 2.0 * self.agent.perception.front_extent

    def check(self, snapshot):
        """Raise ManeuverRejected if the maneuver is not possible now. No side effects."""
        raise NotImplementedError

    def start(self, snapshot, sim_time):
        raise NotImplementedError

    def step(self, snapshot, sim_time):
        raise NotImplementedError

    def cancel(self, keep_offset=False):
        self.agent.reset_maneuver_hooks(keep_offset=keep_offset)

    def debug_points(self):
        return []

    def describe(self):
        text = self.action
        if self.phase == PHASE_WAIT_GAP:
            text += " (waiting for a gap)"
        if self.note:
            text += ": " + self.note
        return text

    # ---------------------------------------------------------- utilities
    def _begin(self, sim_time, phase):
        self.started_at = sim_time
        self._set_phase(phase, sim_time)

    def _set_phase(self, phase, sim_time):
        self.phase = phase
        self.phase_since = sim_time

    @staticmethod
    def _hold():
        return carla.VehicleControl(throttle=0.0, brake=1.0)

    def _check_stuck(self, sim_time, timeout):
        """FAILED if the car has been standing still behind a hazard for ``timeout`` s."""
        if self.agent.speed_mps < 0.1 and blocking_hazard(self.agent) is not None:
            if self._stuck_since is None:
                self._stuck_since = sim_time
            elif sim_time - self._stuck_since > timeout:
                hazard = self.agent.hazard
                self.status = FAILED
                self.note = "blocked by {} at {:.1f} m".format(hazard.type_id or hazard.kind,
                                                               hazard.distance)
        else:
            self._stuck_since = None

    def _vehicle_regions_clear(self, snapshot, regions, moving_only_regions=(), skip_ids=(),
                               approaching=None):
        """(clear, reason) for vehicles/walkers in the given shapely regions.

        ``moving_only_regions`` only count vehicles faster than 1 m/s; with
        ``approaching`` (a unit (x, y) direction) they must also move along it.
        """
        own_id = self.ctx.vehicle.id
        for other in snapshot.vehicles:
            if other.id == own_id or other.id in skip_ids:
                continue
            footprint = geometry.actor_footprint(other)
            for region in regions:
                if region is not None and region.intersects(footprint):
                    return False, "{} in the way".format(other.type_id.split(".")[-1])
            velocity = other.get_velocity()
            if velocity.length() <= 1.0:
                continue
            if approaching is not None and velocity.x * approaching[0] + velocity.y * approaching[1] < 0.5:
                continue
            for region in moving_only_regions:
                if region is not None and region.intersects(footprint):
                    return False, "{} approaching".format(other.type_id.split(".")[-1])
        for walker in snapshot.walkers:
            footprint = geometry.actor_footprint(walker, fallback_radius=0.4)
            for region in regions:
                if region is not None and region.intersects(footprint):
                    return False, "pedestrian in the way"
        return True, ""


# ------------------------------------------------------------------- bypass
class BypassManeuver(Maneuver):
    """Pass the blocking obstacle cluster on one side with a smooth lateral shift.

    The lateral profile u(s) (metres towards the bypass side, as a function of
    the front-bumper distance s) is the smallest slope-limited curve that keeps
    the whole car ``lateral_margin_m`` away from every element of the cluster
    while any part of the car is alongside it.
    """

    def __init__(self, ctx, side, seed_ids=None, resume=False):
        super().__init__(ctx)
        self.side = side
        self.sign = -1.0 if side == "left" else 1.0
        self.action = "BYPASS_LEFT" if side == "left" else "BYPASS_RIGHT"
        self.seed_ids = list(seed_ids or [])
        self.resume = resume
        self.margin = float(self.cfg.get("lateral_margin_m", 0.7))
        self.slope = float(self.cfg.get("max_lateral_slope", 0.6))
        self.speed = float(self.cfg.get("bypass_speed_kmh", 10.0))
        # The car's lateral position trails the commanded offset by a few metres of
        # travel (the PID aims at a point ahead and needs time to yaw), so every
        # constraint must already be met this much earlier...
        self.lag = float(self.cfg.get("tracking_lag_m", 4.0))
        # ...and the rear of the car cuts inside when it turns back into its lane.
        self.rear_margin = float(self.cfg.get("rear_margin_m", 1.5))
        self.gap_timeout = float(self.cfg.get("gap_timeout_s", 30.0))
        self.oncoming_clear = float(self.cfg.get("oncoming_clear_m", 60.0))
        self.oncoming_speed = float(self.cfg.get("oncoming_speed_mps", 7.0))
        self.behind_clear = float(self.cfg.get("behind_clear_m", 20.0))
        self.stuck_timeout = float(self.cfg.get("stuck_timeout_s", 20.0))
        self.frame = None
        self.s0 = 0.0
        self.s_values = []
        self.profile = []
        self.s_end = 0.0
        self.u_max = 0.0
        self.s_last = 0.0
        self.path_line = None
        self.cluster_ids = []
        self.side_kind = ""
        self._side_wp = None
        self._side_center = 0.0
        self._side_half = 0.0
        self._committed = False

    # ------------------------------------------------------------ planning
    def plan_profile(self, frame, s0, cluster, start_shift=0.0, max_shift=None, side_label=None):
        """Compute the lateral profile. Raises ManeuverRejected if impossible."""
        intervals = []
        for polygon in cluster.values():
            sd = [frame.project(x, y) for x, y in polygon.exterior.coords]
            s_min = min(s for s, _ in sd) - s0
            s_max = max(s for s, _ in sd) - s0
            reach = max(self.sign * d for _, d in sd)   # how far it sticks out towards our side
            need = self.half_width + self.margin + reach
            if need > 0.0:
                intervals.append((s_min - 0.5 - self.lag, s_max + self.length + self.rear_margin, need))
        if not intervals:
            raise ManeuverRejected("the obstacle does not block the {} side".format(self.side))
        u_max = max(need for _, _, need in intervals)
        if max_shift is not None and u_max > max_shift:
            if round(u_max, 1) > round(max_shift, 1):
                raise ManeuverRejected("not enough room on the {}: needs a {:.1f} m shift, the {} allows {:.1f} m"
                                       .format(self.side, u_max, side_label or "space", max_shift))
            raise ManeuverRejected("not enough room on the {}: needs a {:.1f} m shift, slightly more than the {} "
                                   "allows".format(self.side, u_max, side_label or "space"))
        s_last = max(s1 for _, s1, _ in intervals)
        s_end = s_last + u_max / self.slope + 1.0
        step = 0.5
        s_values = [i * step for i in range(int((s_end + 6.0) / step) + 1)]
        envelope = geometry.slope_limited_envelope(intervals, self.slope, s_values)
        tolerance = 0.6 if self.resume else 0.35
        deficit = envelope[0] - (max(start_shift, 0.0) + tolerance)
        if deficit > 0.0:
            raise ManeuverRejected("obstacle too close to steer {}: needs about {:.0f} m more room".format(
                self.side, deficit / self.slope + 0.5))
        profile = [max(e, start_shift - self.slope * s) for s, e in zip(s_values, envelope)]
        return s_values, profile, s_end, u_max, s_last

    def check(self, snapshot):
        agent = self.agent
        seeds = [i for i in self.seed_ids if snapshot.actor(i) is not None] if self.seed_ids else None
        if self.seed_ids and not seeds:
            raise ManeuverRejected(GONE)
        if seeds is None:
            hazard = blocking_hazard(agent)
            if hazard is None:
                raise ManeuverRejected("nothing to bypass ahead")
            if hazard.kind == HAZARD_WALKER:
                raise ManeuverRejected("a pedestrian is in the way; wait")
            if hazard.actor_id in self.ctx.fleet_ids:
                raise ManeuverRejected("queued behind another robotaxi")
            seeds = [hazard.actor_id]

        waypoint = route_lane_waypoint(agent)
        side_kind = compat.describe_side_lane(waypoint, self.side)
        usable = ["driving_same", "driving_opposite"]
        if self.side == "right":
            usable += ["shoulder", "parking"]
        if side_kind not in usable:
            raise ManeuverRejected("no drivable lane on the {} ({})".format(self.side, side_kind))
        side_wp = compat.left_lane(waypoint) if self.side == "left" else compat.right_lane(waypoint)

        cluster = obstacle_cluster(snapshot, seeds, self.ctx.fleet_ids)
        if not cluster:
            raise ManeuverRejected(GONE)
        frame = geometry.PathFrame(reference_points(agent))
        fx, fy = agent.front_location()
        s0, _ = frame.project(fx, fy)
        start_shift = self.sign * agent.current_offset
        limit = waypoint.lane_width / 2.0 + side_wp.lane_width / 2.0 + 0.3
        label = "{} lane".format(side_kind.replace("_", " ").replace("driving ", ""))
        s_values, profile, s_end, u_max, s_last = self.plan_profile(
            frame, s0, cluster, start_shift, max_shift=limit, side_label=label)
        if s0 + s_end + 2.0 > frame.length:
            raise ManeuverRejected("route too short after the obstacle")
        junction_at = self._junction_on_path(s_end)
        if junction_at is not None:
            first = min(frame.project(*geometry.xy(snapshot.actor(i).get_location()))[0]
                        for i in cluster if snapshot.actor(i) is not None) - s0
            if junction_at < first:
                raise ManeuverRejected("the obstacle is beyond the next junction; turn instead")
            raise ManeuverRejected("the bypass would run into a junction {:.0f} m ahead".format(junction_at))

        self.frame, self.s0 = frame, s0
        self.s_values, self.profile, self.s_end = s_values, profile, s_end
        self.u_max, self.s_last = u_max, s_last
        self.cluster_ids = list(cluster)
        self.side_kind = side_kind
        self._side_wp = side_wp
        self._side_center = self.sign * (waypoint.lane_width / 2.0 + side_wp.lane_width / 2.0)
        self._side_half = side_wp.lane_width / 2.0
        return "shift {:.1f} m to the {} for {:.0f} m ({})".format(
            u_max, self.side, s_last, side_kind.replace("_", " "))

    def _junction_on_path(self, s_end):
        """Distance from the front bumper to a junction within the bypass, else None."""
        agent = self.agent
        here = agent.vehicle().get_location()
        travelled = None
        previous = None
        for waypoint, _ in agent.plan():
            location = waypoint.transform.location
            if previous is None:
                forward = waypoint.transform.get_forward_vector()
                travelled = geometry.longitudinal_offset(location, here, forward)
            else:
                travelled += geometry.planar_distance(previous, location)
            previous = location
            ahead = travelled - agent.perception.front_extent
            if ahead > s_end:
                return None
            if waypoint.is_junction and ahead > -2.0:
                return max(0.0, ahead)
        return None

    def start(self, snapshot, sim_time):
        note = self.check(snapshot)
        points = [self.frame.point_at(self.s0 + s, self.sign * u)
                  for s, u in zip(self.s_values[::2], self.profile[::2])]
        tail = self.s0 + self.s_values[-1]
        for extra in range(2, 22, 2):
            points.append(self.frame.point_at(tail + extra, 0.0))
        self.path_line = geometry.polyline(points)
        self._committed = abs(self.agent.current_offset) > 1.0
        self._begin(sim_time, PHASE_WAIT_GAP)
        return note

    # ----------------------------------------------------------- traffic gap
    def _gap_clear(self, snapshot):
        """Is it safe to (keep) move(ing) into the side lane?"""
        path_region = geometry.corridor(self.path_line, self.half_width + 0.6)
        regions = [path_region]
        moving = []
        approaching = None
        if self.side_kind == "driving_opposite":
            # Oncoming traffic needs enough distance for the whole time we occupy its lane.
            occupy_s = (self.s_end + 10.0) / max(1.0, self.speed / 3.6) + 2.0
            needed = max(self.oncoming_clear, self.oncoming_speed * occupy_s)
            # Ahead of us in our direction = against the opposite lane's direction.
            line = geometry.polyline(lane_walk(self._side_wp, needed, forward=False))
            if line is not None:
                moving.append(geometry.corridor(line, self._side_half))
            forward = self.ctx.vehicle.get_transform().get_forward_vector()
            approaching = (-forward.x, -forward.y)
        elif self.side_kind == "driving_same":
            line = geometry.polyline(lane_walk(self._side_wp, self.behind_clear, forward=False))
            if line is not None:
                moving.append(geometry.corridor(line, self._side_half))
        return self._vehicle_regions_clear(snapshot, regions, moving, skip_ids=self.cluster_ids,
                                           approaching=approaching)

    # ------------------------------------------------------------------ step
    def _shift_at(self, s_rel):
        if s_rel <= 0.0:
            return self.profile[0]
        index = s_rel / 0.5
        low = int(index)
        if low + 1 >= len(self.profile):
            return 0.0
        frac = index - low
        return self.profile[low] * (1.0 - frac) + self.profile[low + 1] * frac

    def step(self, snapshot, sim_time):
        agent = self.agent
        here = self.ctx.vehicle.get_location()
        _, actual = self.frame.project(here.x, here.y)
        if abs(actual) > 1.0:
            # Out of our lane: from here on, finishing the pass is the safest option.
            self._committed = True

        if not self._committed:
            clear, reason = self._gap_clear(snapshot)
            if not clear:
                if self.phase != PHASE_WAIT_GAP:
                    self._set_phase(PHASE_WAIT_GAP, sim_time)
                self.note = reason
                if sim_time - self.phase_since > self.gap_timeout:
                    self.status = FAILED
                    self.note = "no gap for {:.0f}s ({})".format(self.gap_timeout, reason)
                return self._hold()
            if self.phase == PHASE_WAIT_GAP:
                self._set_phase(PHASE_EXECUTE, sim_time)

        fx, fy = agent.front_location()
        s, _ = self.frame.project(fx, fy)
        s_rel = s - self.s0
        agent.set_lateral_offset(self.sign * self._shift_at(s_rel + self.preview), actual)
        agent.speed_cap_kmh = self.speed
        agent.perception.path_override = self.path_line
        for actor_id in self.cluster_ids:
            agent.perception.ignore(actor_id)
        control = agent.run_step(snapshot)
        self.note = "{:.0f}%".format(100.0 * min(1.0, max(0.0, s_rel / self.s_end)))

        if s_rel >= self.s_end + 1.0:
            agent.reset_maneuver_hooks()
            self.status = DONE
            self.note = "passed"
        else:
            self._check_stuck(sim_time, self.stuck_timeout)
        return control

    def debug_points(self):
        return list(self.path_line.coords) if self.path_line is not None else []


# --------------------------------------------------------------------- turn
class TurnManeuver(Maneuver):
    """Re-route through the next junction in the chosen direction."""

    def __init__(self, ctx, direction):
        super().__init__(ctx)
        self.direction = direction
        self.action = "TURN_LEFT" if direction == "left" else "TURN_RIGHT"
        self.search = float(self.cfg.get("turn_search_m", 150.0))
        self.min_angle = float(self.cfg.get("turn_min_angle_deg", 35.0))
        self._choice = None

    def check(self, snapshot):
        agent = self.agent
        current = self.ctx.map.get_waypoint(self.ctx.vehicle.get_location())
        if current.is_junction:
            raise ManeuverRejected("already inside a junction")
        junction_distance, index = next_junction_distance(agent, self.search)
        if junction_distance is None:
            raise ManeuverRejected("no junction within {:.0f} m on the route".format(self.search))
        hazard = blocking_hazard(agent)
        if hazard is not None and hazard.distance + agent.perception.front_extent < junction_distance:
            raise ManeuverRejected("the obstacle ({:.0f} m) is before the next junction ({:.0f} m)".format(
                hazard.distance, junction_distance))
        plan = agent.plan()
        approach = plan[index - 1][0] if index > 0 else current
        junction = plan[index][0].get_junction()
        exits = junction_exits(junction, approach, self.min_angle)[self.direction]
        if not exits:
            raise ManeuverRejected("no {} turn at the next junction".format(self.direction))
        exit_wp = exits[0][1]
        beyond = exit_wp.next(8.0)
        target = beyond[0] if beyond else exit_wp

        planner = agent.route_planner()
        try:
            to_turn = planner.trace_route(current.transform.location, target.transform.location)
        except networkx.NetworkXNoPath:
            raise ManeuverRejected("the {} turn is not reachable".format(self.direction))
        exit_location = exit_wp.transform.location
        via_exit = any(wp.transform.location.distance(exit_location) < 3.0 for wp, _ in to_turn)
        if not via_exit or len(to_turn) * 2.0 > junction_distance + 120.0:
            raise ManeuverRejected("the {} turn is not reachable from this lane".format(self.direction))

        # Everything that closes the road (barriers, cones, signs) and the lanes it sits on.
        closure = []
        if hazard is not None:
            cluster = obstacle_cluster(snapshot, hazard.actor_id, self.ctx.fleet_ids)
            closure = [snapshot.actor(actor_id).get_location() for actor_id in cluster
                       if snapshot.actor(actor_id) is not None]
        if _passes(to_turn, closure):
            raise ManeuverRejected("a {} turn here leads into the closed road".format(self.direction))
        onward = []
        ends_trip = False
        if self.ctx.destination is not None:
            blocked_edges = set()
            for location in closure:
                try:
                    blocked_edges.add(planner._localize(location))  # pylint: disable=protected-access
                except (KeyError, networkx.NetworkXException):
                    pass   # a cone off the road network
            try:
                onward = _trace_avoiding(planner, target.transform.location, self.ctx.destination,
                                         blocked_edges)
            except networkx.NetworkXNoPath:
                onward = None
            if onward is None or _passes(onward, closure):
                # The destination can only be reached through the closed road. Turn anyway
                # and end the trip a little way after the junction (drop-off moved there).
                onward = []
                ends_trip = True
                stop = _lane_point_after(exit_wp, 30.0)
                if stop is not None:
                    try:
                        extension = planner.trace_route(target.transform.location, stop.transform.location)
                    except networkx.NetworkXNoPath:
                        extension = []
                    if not _passes(extension, closure):
                        onward = extension
        self._choice = (junction_distance, junction, to_turn + onward[1:])
        note = "turning {} at the junction in {:.0f} m".format(self.direction, junction_distance)
        if ends_trip:
            note += "; the destination is only reachable through the closure, the trip ends after the turn"
        return note

    def start(self, snapshot, sim_time):
        note = self.check(snapshot)
        junction_distance, junction, plan = self._choice
        self.agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
        self._begin(sim_time, PHASE_EXECUTE)
        self.note = "junction {} in {:.0f} m".format(junction.id, junction_distance)
        return note

    def step(self, snapshot, sim_time):
        # The new route is in place; driving it is ordinary cruising.
        self.status = DONE
        return self.agent.run_step(snapshot)


def _lane_point_after(waypoint, distance, step=2.0):
    """The point ``distance`` metres down the lane, stopping early before a junction."""
    current, travelled = waypoint, 0.0
    while travelled < distance:
        following = current.next(step)
        if not following or following[0].is_junction:
            break
        current = following[0]
        travelled += step
    return current if travelled > 0.0 else None


def _passes(route, locations, radius=1.5):
    """True if any waypoint of ``route`` comes within ``radius`` of one of ``locations``."""
    if not locations:
        return False
    for waypoint, _ in route:
        point = waypoint.transform.location
        for location in locations:
            if abs(point.x - location.x) < radius and abs(point.y - location.y) < radius \
                    and geometry.planar_distance(point, location) < radius:
                return True
    return False


def _trace_avoiding(planner, origin, destination, blocked_edges, penalty=1.0e5):
    """trace_route with the graph edges of the blocked lanes made very expensive."""
    graph = planner._graph  # pylint: disable=protected-access
    edges = [edge for edge in (blocked_edges or ()) if edge is not None and graph.has_edge(*edge)]
    originals = {edge: graph.edges[edge]["length"] for edge in edges}
    for edge in edges:
        graph.edges[edge]["length"] = originals[edge] + penalty
    try:
        return planner.trace_route(origin, destination)
    finally:
        for edge, length in originals.items():
            graph.edges[edge]["length"] = length


# ---------------------------------------------------------------- pull over
class PullOverManeuver(Maneuver):
    """Move to the rightmost lane, ease towards the curb and stop."""

    action = "PULL_OVER"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.distance = float(self.cfg.get("pull_over_distance_m", 25.0))
        self.speed = float(self.cfg.get("pull_over_speed_kmh", 18.0))
        self.min_room = float(self.cfg.get("pull_over_min_room_m", 10.0))
        self.gap_timeout = float(self.cfg.get("gap_timeout_s", 30.0))
        self.behind_clear = float(self.cfg.get("behind_clear_m", 20.0))
        self.stuck_timeout = float(self.cfg.get("stuck_timeout_s", 20.0))
        self.in_place = False
        self.curb_offset = 0.0
        self.frame = None
        self.ramp_start = 0.0
        self.ramp_length = 8.0
        self.lane_changes = 0
        self._target_wp = None
        self._path = None
        self._room = float("inf")

    def _curb_offset(self, lane_wp):
        kind = compat.describe_side_lane(lane_wp, "right")
        offset = max(0.0, lane_wp.lane_width / 2.0 - self.half_width - 0.25)
        if kind in ("shoulder", "parking"):
            shoulder = compat.right_lane(lane_wp).lane_width
            if shoulder >= 1.0:
                offset += min(shoulder, 2.5)
        return offset, kind

    def _build_path(self, waypoint, distance):
        if self.lane_changes > 0:
            change = max(8.0, 3.0 * self.agent.speed_mps) * self.lane_changes
            path = BasicAgent._generate_lane_change_path(  # pylint: disable=protected-access
                waypoint, "right", distance_same_lane=2.0,
                distance_other_lane=max(5.0, distance - change), lane_change_distance=change,
                check=False, lane_changes=self.lane_changes, step_distance=2.0)
        else:
            path = [(waypoint, RoadOption.LANEFOLLOW)]
            travelled = 0.0
            while travelled < distance:
                following = path[-1][0].next(2.0)
                if not following:
                    return []
                path.append((following[0], RoadOption.LANEFOLLOW))
                travelled += 2.0
        if any(wp.is_junction for wp, _ in path):
            return []
        return path

    def check(self, snapshot):
        agent = self.agent
        waypoint = self.ctx.map.get_waypoint(self.ctx.vehicle.get_location())
        if waypoint.is_junction:
            raise ManeuverRejected("inside a junction; try again after it")
        target = route_lane_waypoint(agent)
        self.lane_changes = 0
        while True:
            right = compat.right_lane(target)
            if right is None or right.lane_type != carla.LaneType.Driving \
                    or not compat.same_direction(target, right):
                break
            target = right
            self.lane_changes += 1
        self._target_wp = target
        hazard = blocking_hazard(agent)
        self._room = hazard.distance - 2.0 if hazard is not None else float("inf")
        if self._room < self.min_room and self.lane_changes == 0:
            self.in_place = True
            return "securing the vehicle in place (obstacle {:.0f} m ahead)".format(hazard.distance)
        self.in_place = False
        start = route_lane_waypoint(agent)
        for distance in (min(self.distance, self._room), 18.0, 12.0):
            if distance < 8.0:
                continue
            self._path = self._build_path(start, distance)
            if self._path:
                return "stopping at the curb in {:.0f} m ({} lane change(s))".format(
                    distance, self.lane_changes)
        raise ManeuverRejected("no place to stop before the next junction")

    def start(self, snapshot, sim_time):
        note = self.check(snapshot)
        if self.in_place:
            self._begin(sim_time, PHASE_EXECUTE)
            self.note = "no room to move"
            return note
        start_wp = route_lane_waypoint(self.agent)
        self.curb_offset, curb_kind = self._curb_offset(self._target_wp)
        self.agent.set_global_plan(self._path, stop_waypoint_creation=True, clean_queue=True)
        self.frame = geometry.PathFrame([geometry.xy(wp.transform.location) for wp, _ in self._path])
        self.ramp_start = max(0.0, self.frame.length - 14.0)
        self._begin(sim_time, PHASE_WAIT_GAP if self.lane_changes else PHASE_EXECUTE)
        self._start_wp = start_wp
        return "{} (curb: {})".format(note, curb_kind)

    def step(self, snapshot, sim_time):
        if self.in_place:
            self.status = DONE
            return self._hold()
        if self.phase == PHASE_WAIT_GAP:
            right = compat.right_lane(self._start_wp)
            behind = None
            if right is not None:
                line = geometry.polyline(lane_walk(right, self.behind_clear, forward=False))
                behind = geometry.corridor(line, right.lane_width / 2.0) if line is not None else None
            clear, reason = self._vehicle_regions_clear(snapshot, [], [behind])
            if not clear:
                self.note = reason
                if sim_time - self.phase_since > self.gap_timeout:
                    self.status = FAILED
                    self.note = "no gap to change lanes"
                return self._hold()
            self.note = ""
            self._set_phase(PHASE_EXECUTE, sim_time)

        agent = self.agent
        fx, fy = agent.front_location()
        s, _ = self.frame.project(fx, fy)
        here = self.ctx.vehicle.get_location()
        _, actual = self.frame.project(here.x, here.y)
        agent.set_lateral_offset(self.curb_offset * smoothstep(
            (s + self.preview - self.ramp_start) / self.ramp_length), actual)
        agent.speed_cap_kmh = self.speed
        control = agent.run_step(snapshot)
        remaining = agent.route_length_ahead(10.0)
        self.note = "{:.0f} m to stop".format(remaining)
        if not agent.has_route() or (remaining < 1.5 and agent.speed_mps < 0.2):
            agent.speed_cap_kmh = None
            self.status = DONE
            self.note = "at the curb"
            return self._hold()
        self._check_stuck(sim_time, self.stuck_timeout)
        return control


# ------------------------------------------------------------------- resume
class ResumeManeuver(Maneuver):
    """Return from a lateral offset (curb or aborted bypass) to the lane centre and drive on."""

    action = "RESUME"

    def __init__(self, ctx, replan):
        super().__init__(ctx)
        self.replan = replan
        self.merge_length = float(self.cfg.get("merge_distance_m", 12.0))
        self.speed = float(self.cfg.get("merge_speed_kmh", 15.0))
        self.behind_clear = float(self.cfg.get("behind_clear_m", 20.0))
        self.gap_timeout = float(self.cfg.get("gap_timeout_s", 30.0))
        self.start_offset = 0.0
        self.frame = None
        self.s0 = 0.0
        self.no_destination = False

    def check(self, snapshot):
        if self.replan and self.ctx.destination is not None:
            planner = self.agent.route_planner()
            here = self.ctx.vehicle.get_location()
            try:
                planner.trace_route(here, self.ctx.destination)
            except networkx.NetworkXNoPath:
                raise ManeuverRejected("no route to the destination")
        return "continuing"

    def start(self, snapshot, sim_time):
        agent = self.agent
        self.check(snapshot)
        self.start_offset = agent.current_offset
        if self.replan:
            if self.ctx.destination is None:
                self.no_destination = True
            else:
                agent.set_destination(self.ctx.destination)
        self.frame = geometry.PathFrame(reference_points(agent))
        fx, fy = agent.front_location()
        self.s0, _ = self.frame.project(fx, fy)
        needs_gap = abs(self.start_offset) > 0.3
        self._begin(sim_time, PHASE_WAIT_GAP if needs_gap else PHASE_EXECUTE)
        return "merging back from {:.1f} m offset".format(self.start_offset) if needs_gap \
            else "continuing"

    def step(self, snapshot, sim_time):
        if self.no_destination:
            self.agent.reset_maneuver_hooks()
            self.status = DONE
            return self._hold()
        lane_half = route_lane_waypoint(self.agent).lane_width / 2.0
        if self.phase == PHASE_WAIT_GAP:
            behind = self.frame.strip(self.s0 - self.behind_clear - 5.0, self.s0 + 5.0, 0.0, lane_half)
            clear, reason = self._vehicle_regions_clear(snapshot, [], [behind])
            if not clear:
                self.note = reason
                if sim_time - self.phase_since > self.gap_timeout:
                    self.status = FAILED
                    self.note = "no gap to merge"
                return self._hold()
            self.note = ""
            self._set_phase(PHASE_EXECUTE, sim_time)

        agent = self.agent
        fx, fy = agent.front_location()
        s, _ = self.frame.project(fx, fy)
        here = self.ctx.vehicle.get_location()
        _, actual = self.frame.project(here.x, here.y)
        progress = (s - self.s0 + self.preview) / self.merge_length
        agent.set_lateral_offset(self.start_offset * (1.0 - smoothstep(progress)), actual)
        agent.speed_cap_kmh = self.speed
        control = agent.run_step(snapshot)
        if progress >= 1.0 and s - self.s0 >= 0.6 * self.merge_length:
            agent.reset_maneuver_hooks()
            self.status = DONE
        else:
            self._check_stuck(sim_time, 5.0)
        return control
