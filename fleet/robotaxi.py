"""One robotaxi: ride mission + driving agent + remote-assistance supervisor.

State machine:

    IDLE --plan trip--> CRUISING --arrived--> DWELLING --dwell over--> IDLE
                          |  ^
      stopped by obstacle |  | obstacle gone
                          v  |
                        BLOCKED --waited long enough--> AWAITING_RA
                                                           |
                       operator command accepted ----------+
                                                           v
                     CRUISING <--maneuver done-------- EXECUTING --failed/abort--> AWAITING_RA
                                                           |
                                        pull over done     v
                                                      PULLED_OVER --RESUME--> EXECUTING

Requests are vehicle-initiated and carry the options the vehicle considers
feasible. Commands are checked by the vehicle itself (ra_demo.av.maneuvers)
and may be rejected with a reason.
"""

import collections
import logging

import carla
import networkx

from ra_demo.av import maneuvers as mv
from ra_demo.av.robotaxi_agent import HAZARD_STATIC, HAZARD_VEHICLE, HAZARD_WALKER
from ra_demo.core import bus as ra
from ra_demo.core import compat

LOG = logging.getLogger(__name__)

STATE_IDLE = "IDLE"
STATE_CRUISING = "CRUISING"
STATE_DWELLING = "DWELLING"
STATE_BLOCKED = "BLOCKED"
STATE_AWAITING_RA = "AWAITING_RA"
STATE_EXECUTING = "EXECUTING"
STATE_PULLED_OVER = "PULLED_OVER"


def _lights(*flags):
    value = 0
    for flag in flags:
        value |= int(flag)
    return carla.VehicleLightState(value)


_LS = carla.VehicleLightState
HAZARD_LIGHTS = _lights(_LS.LeftBlinker, _LS.RightBlinker, _LS.Position)
NORMAL_LIGHTS = _lights(_LS.Position, _LS.LowBeam)
LEFT_SIGNAL = _lights(_LS.LeftBlinker, _LS.Position, _LS.LowBeam)
RIGHT_SIGNAL = _lights(_LS.RightBlinker, _LS.Position, _LS.LowBeam)

_REASONS = {
    HAZARD_STATIC: ra.REASON_LANE_BLOCKED,
    HAZARD_VEHICLE: ra.REASON_STOPPED_VEHICLE,
    HAZARD_WALKER: ra.REASON_PEDESTRIAN,
}

_MANEUVER_LIGHTS = {
    ra.ACTION_BYPASS_LEFT: LEFT_SIGNAL,
    ra.ACTION_BYPASS_RIGHT: RIGHT_SIGNAL,
    ra.ACTION_PULL_OVER: RIGHT_SIGNAL,
    ra.ACTION_RESUME: LEFT_SIGNAL,
}


def _late_lane_change(route, site, distance):
    """True if ``route`` changes into the site's lane less than ``distance`` m before the zone."""
    first = next((index for index, (waypoint, _) in enumerate(route) if site.covers(waypoint)), None)
    if first is None:
        return False
    travelled = 0.0
    for index in range(first, 0, -1):
        waypoint, previous = route[index][0], route[index - 1][0]
        travelled += waypoint.transform.location.distance(previous.transform.location)
        if travelled > distance:
            return False
        if previous.road_id == waypoint.road_id and previous.lane_id != waypoint.lane_id:
            return True
    return False


class Robotaxi:
    """Per-vehicle controller combining mission, agent and supervisor."""

    def __init__(self, name, display_name, vehicle, agent, mission, cfg, bus, map_inst,
                 obstacle_owner=None):
        self.name = name
        self.display_name = display_name
        self.vehicle = vehicle
        self.agent = agent
        self.mission = mission
        self.bus = bus
        self._map = map_inst
        # Callback: actor id -> scenario id (or None). Used to tag RA requests.
        self._obstacle_owner = obstacle_owner or (lambda actor_id: None)
        # Other robotaxis: id -> display name (filled in by the fleet manager).
        self.fleet_names = {}

        supervisor = cfg.get("supervisor", {})
        self._stopped_speed = float(supervisor.get("stopped_speed_mps", 0.15))
        self._waits = {
            HAZARD_STATIC: float(supervisor.get("static_block_wait_s", 3.0)),
            HAZARD_VEHICLE: float(supervisor.get("vehicle_block_wait_s", 25.0)),
            HAZARD_WALKER: float(supervisor.get("walker_block_wait_s", 15.0)),
        }
        self._stalled_wait = float(supervisor.get("stalled_vehicle_wait_s", 5.0))
        self._enter_blocked = float(supervisor.get("blocked_enter_s", 1.0))
        self._clear_confirm = float(supervisor.get("clear_confirm_s", 1.0))
        self._dwell = float(cfg.get("mission", {}).get("dwell_s", 10.0))
        self._maneuver_cfg = cfg.get("maneuvers", {})

        mission_cfg = cfg.get("mission", {})
        self._via_min_distance = float(mission_cfg.get("via_site_min_distance_m", 80.0))
        self._via_max_detour = float(mission_cfg.get("via_site_max_detour_m", 600.0))
        # Callables set by the fleet: incident sites a trip may be routed through, and
        # the locations of all sites (no pickup or drop-off right there).
        self.site_hint = None
        self.site_spots = None
        self.via_site = None

        self.state = STATE_IDLE
        self.state_since = 0.0
        self.status_note = ""
        self.request = None
        self.maneuver = None
        self.destination = None
        self._hold_position = False
        self._event_request = False        # request raised by an event (S4), not by an obstacle
        self._suspended_bypass = None      # (side, cluster ids) of an aborted bypass
        self._snapshot = None
        self._options = ([], {}, -1.0)     # (options, {action: reason}, sim time)
        self._blocked_since = None
        self._blocked_actor = None
        self._clear_since = None
        self._dwell_until = 0.0
        self._arrival_since = None
        self._retry_trip_at = 0.0

        # Statistics shown in the HUD and printed at exit.
        self.collisions = 0
        self.collision_events = collections.deque()   # filled by the sensor thread
        self.last_collision = ""
        self._last_contact = (None, -1000)
        self.requests_opened = 0
        self.blocked_time = 0.0
        self.maneuvers_done = 0

    # ----------------------------------------------------------- properties
    @property
    def speed_kmh(self):
        return self.agent.speed_mps * 3.6

    def location(self):
        return self.vehicle.get_location()

    def available_actions(self):
        """Actions worth offering to the operator in the current state."""
        if self.state == STATE_EXECUTING:
            return [ra.ACTION_ABORT]
        if self.state in (STATE_IDLE, STATE_DWELLING):
            return []
        return list(self._evaluate_options()[0])

    def unavailable_reasons(self):
        """{action: why the vehicle would reject it} for the current state."""
        if self.state == STATE_EXECUTING:
            return {action: "busy with {}; ABORT first".format(self.maneuver.action)
                    for action in ra.ALL_ACTIONS if action != ra.ACTION_ABORT}
        if self.state in (STATE_IDLE, STATE_DWELLING):
            return {action: "vehicle is {}".format(self.state.lower()) for action in ra.ALL_ACTIONS}
        reasons = dict(self._evaluate_options()[1])
        reasons.setdefault(ra.ACTION_ABORT, "nothing to abort")
        return reasons

    def _evaluate_options(self, force=False):
        """Dry-run every command (cached for 1 s) to know which ones the vehicle would accept."""
        snapshot = self._snapshot
        options, reasons, stamp = self._options
        now = snapshot.sim_time if snapshot is not None else 0.0
        if snapshot is None or (not force and 0.0 <= now - stamp < 1.0):
            return options, reasons
        options, reasons = [], {}
        ctx = self._context()
        if self.state == STATE_PULLED_OVER:
            candidates = [(ra.ACTION_RESUME, mv.ResumeManeuver(ctx, replan=True))]
            reasons.update({a: "pulled over: RESUME first" for a in ra.ALL_ACTIONS
                            if a not in (ra.ACTION_RESUME, ra.ACTION_ABORT)})
        else:
            seeds = None
            if mv.blocking_hazard(self.agent) is None and self._suspended_bypass:
                seeds = self._suspended_bypass[1]
            candidates = [
                (ra.ACTION_BYPASS_LEFT, mv.BypassManeuver(ctx, "left", seed_ids=seeds)),
                (ra.ACTION_BYPASS_RIGHT, mv.BypassManeuver(ctx, "right", seed_ids=seeds)),
                (ra.ACTION_TURN_LEFT, mv.TurnManeuver(ctx, "left")),
                (ra.ACTION_TURN_RIGHT, mv.TurnManeuver(ctx, "right")),
                (ra.ACTION_PULL_OVER, mv.PullOverManeuver(ctx)),
            ]
            if self._resume_kind() is None:
                reasons[ra.ACTION_RESUME] = "nothing to resume; choose a maneuver"
            else:
                candidates.append((ra.ACTION_RESUME, None))
        for action, maneuver in candidates:
            if maneuver is None:
                options.append(action)
                continue
            try:
                maneuver.check(snapshot)
                options.append(action)
            except mv.ManeuverRejected as exc:
                reasons[action] = str(exc)
            except Exception as exc:  # a dry run must never break the control loop
                LOG.debug("%s: dry run of %s failed: %s", self.display_name, action, exc)
                reasons[action] = "cannot evaluate ({})".format(type(exc).__name__)
        self._options = (options, reasons, now)
        return options, reasons

    def _resume_kind(self):
        """What RESUME would do now: 'bypass', 'merge', 'event', 'curb' or None."""
        if self.state == STATE_PULLED_OVER:
            return "curb"
        if self._suspended_bypass is not None:
            return "bypass"
        if self._hold_position:
            return "merge"
        if self._event_request:
            return "event"
        return None

    # ------------------------------------------------------------ main step
    def step(self, snapshot, sim_time, dt):
        """Return the control for this tick and advance the state machine."""
        self._snapshot = snapshot
        self._drain_collisions(sim_time)
        self._handle_commands(snapshot, sim_time)

        if self.state == STATE_IDLE:
            self._start_next_trip(sim_time)
            if self.state == STATE_IDLE:
                return self._hold()

        if self.state == STATE_DWELLING:
            if sim_time >= self._dwell_until:
                self._set_state(STATE_IDLE, sim_time)
            return self._hold()

        if self.state == STATE_PULLED_OVER:
            return self._hold(hand_brake=True)

        if self.state == STATE_EXECUTING:
            return self._step_maneuver(snapshot, sim_time)

        control = self.agent.run_step(snapshot)
        if self._hold_position:
            # After an ABORT the car stays where it is until the next instruction.
            self.status_note = "holding position"
            return self._hold()

        if self._arrived(sim_time):
            self.mission.arrived()
            self._close_request(sim_time, "CANCELLED", "trip_ended")
            self._dwell_until = sim_time + self._dwell
            self._set_state(STATE_DWELLING, sim_time)
            return self._hold()

        self._supervise(snapshot, sim_time, dt)
        return control

    # ------------------------------------------------------------- missions
    def _start_next_trip(self, sim_time):
        if sim_time < self._retry_trip_at:
            return
        avoid = self.site_spots() if self.site_spots is not None else ()
        index, transform = self.mission.next_destination(self.location(), avoid)
        try:
            route = self.agent.plan_route(transform.location)
        except networkx.NetworkXNoPath:
            LOG.warning("%s: no route to spawn point %d, trying again in 2 s",
                        self.display_name, index)
            self._retry_trip_at = sim_time + 2.0
            return
        self.via_site = None
        offers = self.site_hint() if self.site_hint is not None else []
        site = self.mission.choose_site(offers) if offers else None
        if site is not None:
            route = self._route_via(site, transform.location, route)
        self.agent.set_route(route)
        self.destination = transform.location
        self._arrival_since = None
        self._set_state(STATE_CRUISING, sim_time)
        LOG.info("%s: %s -> spawn point %d%s", self.display_name, self.mission.phase, index,
                 " via site {}".format(self.via_site.name) if self.via_site else "")

    def _route_via(self, site, destination, direct):
        """The trip routed through ``site`` if that is a reasonable detour, else ``direct``."""
        try:
            route = self.agent.plan_route(destination, via=site.via_locations)
        except networkx.NetworkXNoPath:
            return direct
        distance = site.route_distance(self.location(), route)
        if distance is None or distance < self._via_min_distance:
            return direct   # too close to the site already: the incident would pop up in view
        if _late_lane_change(route, site, 25.0):
            return direct   # the route would only enter the closed lane right in front of the zone
        if self.agent.route_length(route) - self.agent.route_length(direct) > self._via_max_detour:
            return direct
        self.via_site = site
        self.mission.last_site = site
        return route

    def _arrived(self, sim_time):
        if not self.agent.has_route():
            return True
        # Fallback: stopped within a couple of metres of the route end.
        if self.agent.route_length_ahead(5.0) < 2.5 and self.agent.speed_mps < 0.3:
            if self._arrival_since is None:
                self._arrival_since = sim_time
            return sim_time - self._arrival_since > 1.5
        self._arrival_since = None
        return False

    # ----------------------------------------------------------- supervisor
    def _blocking_hazard(self, snapshot):
        """The hazard currently holding the car, if it is one worth escalating."""
        hazard = self.agent.hazard
        if self.agent.speed_mps > self._stopped_speed:
            return None
        if hazard.kind in (HAZARD_STATIC, HAZARD_WALKER):
            return hazard
        if hazard.kind == HAZARD_VEHICLE:
            if hazard.actor_id in self.fleet_names:
                # Queued behind another robotaxi: its own request covers the incident.
                self.status_note = "queued behind {}".format(self.fleet_names[hazard.actor_id])
                return None
            blocker = snapshot.vehicles_by_id.get(hazard.actor_id)
            if blocker is None:
                return None
            if blocker.get_velocity().length() > 0.2:
                return None
            try:
                if blocker.is_at_traffic_light():
                    return None
            except RuntimeError:
                return None   # destroyed since the snapshot was taken
            return hazard
        return None

    def _supervise(self, snapshot, sim_time, dt):
        self.status_note = ""
        hazard = self._blocking_hazard(snapshot)

        if hazard is not None:
            self._clear_since = None
            if self._blocked_since is None or hazard.actor_id != self._blocked_actor:
                self._blocked_since = sim_time
                self._blocked_actor = hazard.actor_id
            blocked_for = sim_time - self._blocked_since
            if self.state in (STATE_BLOCKED, STATE_AWAITING_RA):
                self.blocked_time += dt
            if self.state == STATE_CRUISING and blocked_for >= self._enter_blocked:
                self._set_state(STATE_BLOCKED, sim_time)
            reason, wait = self._classify(hazard)
            if self.state == STATE_BLOCKED and blocked_for >= wait:
                self._open_request(reason, sim_time, hazard)
            return

        # No blocking hazard this tick (with hysteresis before leaving BLOCKED/AWAITING_RA).
        self._blocked_since = None
        self._blocked_actor = None
        if self.state == STATE_AWAITING_RA and (self._event_request or self._suspended_bypass):
            return   # waits for the operator, not for an obstacle to disappear
        if self.state in (STATE_BLOCKED, STATE_AWAITING_RA):
            if self._clear_since is None:
                self._clear_since = sim_time
            confirm = 0.5 if self.state == STATE_BLOCKED else self._clear_confirm
            if self.agent.speed_mps > 0.5 or sim_time - self._clear_since >= confirm:
                if self.state == STATE_AWAITING_RA:
                    self._close_request(sim_time, "RESOLVED", "obstacle_cleared")
                self._set_state(STATE_CRUISING, sim_time)

    def _classify(self, hazard):
        """(reason code, seconds to wait before asking) for a blocking hazard."""
        if hazard.kind == HAZARD_STATIC:
            if self.agent.obstacle_beyond_junction:
                return ra.REASON_ROAD_CLOSED, self._waits[HAZARD_STATIC]
            if hazard.type_id.startswith("vehicle."):
                return ra.REASON_STOPPED_VEHICLE, self._stalled_wait
        return _REASONS[hazard.kind], self._waits[hazard.kind]

    def raise_event(self, reason, note, sim_time, speed_cap_kmh=None):
        """Vehicle-initiated request without an obstacle (passenger, sensor fault...).

        The car keeps driving (optionally slower) until the operator answers.
        Returns False if the car cannot take the event now.
        """
        if self.state != STATE_CRUISING or self.request is not None:
            return False
        self._event_request = True
        self.agent.speed_cap_kmh = speed_cap_kmh
        self._open_request(reason, sim_time, note=note)
        return True

    def cancel_event(self, sim_time):
        """Operator cleared the event (e.g. the passenger is fine after all)."""
        if not self._event_request:
            return False
        self._close_request(sim_time, "CANCELLED", "operator_cleared")
        if self.state == STATE_AWAITING_RA:
            self._set_state(STATE_CRUISING, sim_time)
        return True

    def _end_event(self):
        if self._event_request:
            self._event_request = False
            self.agent.speed_cap_kmh = None

    def _feasible_options(self):
        """Options the vehicle itself considers possible right now."""
        waypoint = mv.route_lane_waypoint(self.agent)
        left = compat.describe_side_lane(waypoint, "left")
        right = compat.describe_side_lane(waypoint, "right")
        options = []
        if left in ("driving_same", "driving_opposite"):
            options.append(ra.ACTION_BYPASS_LEFT)
        if right in ("driving_same", "driving_opposite", "shoulder", "parking"):
            options.append(ra.ACTION_BYPASS_RIGHT)
        junction_distance, _ = mv.next_junction_distance(self.agent)
        hazard = mv.blocking_hazard(self.agent)
        if junction_distance is not None and (
                hazard is None
                or hazard.distance + self.agent.perception.front_extent > junction_distance):
            options += [ra.ACTION_TURN_LEFT, ra.ACTION_TURN_RIGHT]
        options.append(ra.ACTION_PULL_OVER)
        return options, {"left_lane": left, "right_lane": right}

    def _open_request(self, reason, sim_time, hazard=None, note=""):
        if self.request is not None:
            self._set_state(STATE_AWAITING_RA, sim_time)
            return
        _, lanes = self._feasible_options()
        options, unavailable = self._evaluate_options(force=True)
        location = self.location()
        detail = {"note": note} if note else {}
        detail["unavailable"] = unavailable
        if hazard is not None:
            detail.update({
                "obstacle_id": hazard.actor_id,
                "obstacle_type": hazard.type_id,
                "obstacle_distance_m": round(hazard.distance, 2),
                "scenario": self._obstacle_owner(hazard.actor_id),
                "blocked_for_s": round(sim_time - (self._blocked_since or sim_time), 1),
            })
        detail.update(lanes)
        self.request = self.bus.open_request(
            self.display_name, reason, sim_time, (location.x, location.y, location.z), options, detail)
        self.requests_opened += 1
        self._set_state(STATE_AWAITING_RA, sim_time)

    def _close_request(self, sim_time, status, resolution):
        if self.request is not None:
            self.bus.close_request(self.request.request_id, status, sim_time, resolution)
            self.request = None
        self._end_event()

    # ------------------------------------------------------------- commands
    def _handle_commands(self, snapshot, sim_time):
        for command in self.bus.take_commands(self.display_name):
            accepted, note = self._execute_command(command, snapshot, sim_time)
            self.bus.reply(command, accepted, sim_time, note)
            LOG.info("%s %s %s: %s", self.display_name, "accepted" if accepted else "rejected",
                     command.action, note)

    def _context(self):
        return mv.ManeuverContext(self.agent, self._maneuver_cfg, self.destination,
                                  self.fleet_names.keys())

    def _execute_command(self, command, snapshot, sim_time):
        action = command.action
        if action == ra.ACTION_ABORT:
            if self.state != STATE_EXECUTING:
                return False, "nothing to abort"
            aborted = self.maneuver
            aborted.cancel(keep_offset=True)
            self.maneuver = None
            self._hold_position = True
            if isinstance(aborted, mv.BypassManeuver):
                self._suspended_bypass = (aborted.side, list(aborted.cluster_ids))
            self.bus.log_event("maneuver_aborted", {"vehicle": self.display_name, "action": aborted.action,
                                                    "sim_time": round(sim_time, 2)})
            self._open_request(ra.REASON_MANEUVER_ABORTED, sim_time,
                               note="{} aborted".format(aborted.action))
            return True, "stopped; holding position (RESUME continues)"

        if self.state == STATE_EXECUTING:
            return False, "busy with {}; send ABORT first".format(self.maneuver.action)
        if self.state in (STATE_IDLE, STATE_DWELLING):
            return False, "vehicle is {}".format(self.state.lower())

        ctx = self._context()
        if action == ra.ACTION_RESUME:
            kind = self._resume_kind()
            if kind is None:
                return False, "nothing to resume; choose a maneuver"
            if kind == "event":
                self._close_request(sim_time, "RESOLVED", "RESUME")
                self._set_state(STATE_CRUISING, sim_time)
                return True, "acknowledged; continuing the trip"
            maneuver = None
            if kind == "bypass":
                side, cluster_ids = self._suspended_bypass
                candidate = mv.BypassManeuver(ctx, side, seed_ids=cluster_ids, resume=True)
                try:
                    note = candidate.start(snapshot, sim_time)
                    maneuver = candidate
                except mv.ManeuverRejected as exc:
                    if str(exc) != mv.GONE:
                        return False, "cannot continue the bypass: {}".format(exc)
                    self._suspended_bypass = None   # obstacle cleared meanwhile: just merge back
            if maneuver is None:
                maneuver = mv.ResumeManeuver(ctx, replan=(kind == "curb"))
        elif action in (ra.ACTION_BYPASS_LEFT, ra.ACTION_BYPASS_RIGHT):
            if self.state not in (STATE_BLOCKED, STATE_AWAITING_RA):
                return False, "not blocked: nothing to bypass"
            seeds = None
            if mv.blocking_hazard(self.agent) is None and self._suspended_bypass:
                seeds = self._suspended_bypass[1]
            maneuver = mv.BypassManeuver(ctx, "left" if action == ra.ACTION_BYPASS_LEFT else "right",
                                         seed_ids=seeds, resume=seeds is not None)
        elif action in (ra.ACTION_TURN_LEFT, ra.ACTION_TURN_RIGHT):
            if self.state == STATE_PULLED_OVER:
                return False, "pulled over: RESUME first"
            maneuver = mv.TurnManeuver(ctx, "left" if action == ra.ACTION_TURN_LEFT else "right")
        elif action == ra.ACTION_PULL_OVER:
            if self.state == STATE_PULLED_OVER:
                return False, "already pulled over"
            maneuver = mv.PullOverManeuver(ctx)
        else:
            return False, "unknown action {}".format(action)

        if maneuver.started_at is None:
            try:
                note = maneuver.start(snapshot, sim_time)
            except mv.ManeuverRejected as exc:
                return False, str(exc)
        self.maneuver = maneuver
        self._hold_position = False
        if isinstance(maneuver, mv.BypassManeuver):
            self._suspended_bypass = None
        self.status_note = maneuver.describe()
        self._set_state(STATE_EXECUTING, sim_time)
        lights = _MANEUVER_LIGHTS.get(maneuver.action)
        if lights is not None:
            self.vehicle.set_light_state(lights)
        self.bus.log_event("maneuver_started", {"vehicle": self.display_name, "action": maneuver.action,
                                                "note": note, "sim_time": round(sim_time, 2),
                                                "request_id": self.request.request_id if self.request else None})
        return True, note

    def _step_maneuver(self, snapshot, sim_time):
        maneuver = self.maneuver
        control = maneuver.step(snapshot, sim_time)
        self.status_note = maneuver.describe()
        if maneuver.status == mv.DONE:
            self.maneuver = None
            self.maneuvers_done += 1
            self.bus.log_event("maneuver_done", {"vehicle": self.display_name, "action": maneuver.action,
                                                 "duration_s": round(sim_time - maneuver.started_at, 1),
                                                 "sim_time": round(sim_time, 2)})
            self.bus.notice("{} completed {}".format(self.display_name, maneuver.action))
            self._close_request(sim_time, "RESOLVED", maneuver.action)
            self._options = ([], {}, -1.0)
            self._blocked_since = None
            if maneuver.action == ra.ACTION_PULL_OVER:
                self._set_state(STATE_PULLED_OVER, sim_time)
                self.status_note = "pulled over"
            elif isinstance(maneuver, mv.ResumeManeuver) and maneuver.no_destination:
                self._set_state(STATE_IDLE, sim_time)
            else:
                self._set_state(STATE_CRUISING, sim_time)
        elif maneuver.status == mv.FAILED:
            self.maneuver = None
            maneuver.cancel(keep_offset=True)
            self._hold_position = abs(self.agent.current_offset) > 0.3
            self.bus.log_event("maneuver_failed", {"vehicle": self.display_name, "action": maneuver.action,
                                                   "note": maneuver.note, "sim_time": round(sim_time, 2)})
            self.bus.notice("{} {} failed: {}".format(self.display_name, maneuver.action, maneuver.note))
            self._open_request(ra.REASON_MANEUVER_FAILED, sim_time, mv.blocking_hazard(self.agent),
                               note="{} failed: {}".format(maneuver.action, maneuver.note))
        return control

    # ---------------------------------------------------------------- utils
    def _set_state(self, state, sim_time):
        if state == self.state:
            return
        previous = self.state
        self.state = state
        self.state_since = sim_time
        if state in (STATE_AWAITING_RA, STATE_PULLED_OVER):
            self.vehicle.set_light_state(HAZARD_LIGHTS)
        elif state in (STATE_CRUISING, STATE_IDLE, STATE_DWELLING, STATE_BLOCKED):
            self.vehicle.set_light_state(NORMAL_LIGHTS)
        self.bus.log_event("state", {"vehicle": self.display_name, "from": previous,
                                     "to": state, "sim_time": round(sim_time, 2)})

    @staticmethod
    def _hold(hand_brake=False):
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=hand_brake)

    def on_collision(self, event):
        """Collision sensor callback (runs on a CARLA worker thread)."""
        self.collision_events.append((event.other_actor.id, event.other_actor.type_id, event.frame))

    def _drain_collisions(self, sim_time):
        while self.collision_events:
            other_id, other, frame = self.collision_events.popleft()
            # The sensor reports every frame of a contact; count one collision per
            # actor unless the contact stopped for more than ~2 s.
            last_id, last_frame = self._last_contact
            self._last_contact = (other_id, frame)
            if other_id == last_id and frame - last_frame < 40:
                continue
            self.collisions += 1
            self.last_collision = other
            LOG.warning("%s collided with %s", self.display_name, other)
            self.bus.log_event("collision", {"vehicle": self.display_name, "other": other,
                                             "frame": frame, "sim_time": round(sim_time, 2)})
