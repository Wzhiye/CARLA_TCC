"""Scenario lifecycle: ARMED -> SEARCHING -> ACTIVE -> COOLDOWN -> (ARMED | DONE).

* ARMED      waiting for its trigger.
* SEARCHING  looking for a valid zone on the target robotaxi's route (route_ahead,
             operator injections) or for a robotaxi to raise an event on.
* ACTIVE     props are in the world; a lane guard keeps background traffic out.
* COOLDOWN   props removed; re-arms if the scenario repeats.

Three kinds of instances:

* Site instances (``trigger.type: site``): one per fixed site on the map. The
  props appear when a robotaxi's route approaches the site (or stay there all
  the time with ``activation: always``); every robotaxi that drives there meets
  the incident. They disappear once no robotaxi is heading there any more, and
  the site re-arms after ``repeat.rearm_after_s``.
* Route instances (``route_ahead``, and every operator injection with T /
  F1..F5): the zone is placed on one robotaxi's own route.
* Event instances (``event:``) place nothing: a robotaxi raises a request (a
  passenger pressed the help button) and the instance stays ACTIVE until it is
  answered. With ``target_vehicle: random`` and ``trigger.interval_s`` the event
  hits a random robotaxi with a passenger at random times.
"""

import copy
import dataclasses
import logging
import random

import carla

from ra_demo.av.maneuvers import obstacle_cluster
from ra_demo.av.robotaxi_agent import HAZARD_STATIC, HAZARD_VEHICLE
from ra_demo.core import geometry
from ra_demo.core.config import get, resolve_path
from ra_demo.core.world import destroy_actors
from ra_demo.fleet.mission import PHASE_TO_DROPOFF
from ra_demo.scenario import patterns
from ra_demo.scenario.loader import load_scenarios
from ra_demo.scenario.placement import find_junction_exit_anchor, find_route_anchor
from ra_demo.scenario.sites import resolve_sites
from ra_demo.fleet.robotaxi import HAZARD_LIGHTS

LOG = logging.getLogger(__name__)

ARMED = "ARMED"
SEARCHING = "SEARCHING"
ACTIVE = "ACTIVE"
COOLDOWN = "COOLDOWN"
DONE = "DONE"

RANDOM_TARGET = "random"
# Robotaxi states in which a car is "dealing with" a site it is standing at.
_BUSY_STATES = ("BLOCKED", "AWAITING_RA", "EXECUTING")
_SITE_CHECK_S = 0.5

_SPAWN_Z_LADDER = (0.0, 0.05, 0.25)
_VEHICLE_Z_LADDER = (0.3, 0.6, 1.0)


class ScenarioInstance:
    """Runtime state of one scenario run."""

    def __init__(self, spec, manual=False, site=None):
        self.spec = spec
        self.id = spec["id"]
        self.manual = manual
        self.site = site
        self.state = ARMED
        self.state_since = 0.0
        self.target = None if site is not None else spec.get("target_vehicle", "RT-01")
        self.anchor = None
        self.prop_ids = []
        self.zone_polygon = None
        self.guard_polygons = []
        self.zone_end = None
        self.request_id = None
        self.nudged = {}        # vehicle id -> sim time of the last lane-change request
        self.passed_since = None
        self.away_since = None
        self.next_attempt = 0.0
        self.last_reason = ""
        self.cycles = 0
        self.due_at = None      # ARMED: sim time at which the trigger fires
        self.heading = set()    # site: robotaxis whose route runs through it
        self.passed_by = []     # site: robotaxis that got past it this cycle
        self.idle_since = None  # site: no robotaxi heading there since

    @property
    def is_event(self):
        return self.spec["trigger"]["type"] == "event"

    @property
    def random_target(self):
        return not self.manual and self.spec.get("target_vehicle") == RANDOM_TARGET

    @property
    def label(self):
        if self.site is not None:
            return "{}@{}".format(self.id.split("_")[0], self.site.name)
        return "{}{}".format(self.id, "*" if self.manual else "")

    def set_state(self, state, sim_time):
        LOG.info("Scenario %s (%s): %s -> %s", self.label, self.target, self.state, state)
        self.state = state
        self.state_since = sim_time


class ScenarioManager:
    """Loads scenario specs and runs their lifecycle every tick."""

    def __init__(self, sim, cfg, fleet, traffic, bus):
        self.sim = sim
        self.cfg = cfg.get("scenarios", {})
        self.fleet = fleet
        self.traffic = traffic
        self.bus = bus
        self.instances = []
        self._specs = []
        self._owner = {}       # prop actor id -> ScenarioInstance
        self._rng = random.Random(int(get(cfg, "world.seed", 0) or 0) * 31 + 7)
        # Lets unattended runs (no operator) remove props automatically. 0 disables it.
        self.auto_clear_s = float(self.cfg.get("auto_clear_after_s", 0) or 0)
        self.manual_min_distance = float(self.cfg.get("manual_min_distance_m", 40.0))

    def load(self):
        directory = resolve_path(self.cfg.get("dir", "scenarios"))
        enabled = self.cfg.get("enabled")
        for spec in load_scenarios(directory, enabled):
            self._specs.append(spec)
            trigger = spec["trigger"]
            if not trigger.get("auto", True):
                continue
            if trigger["type"] == "site":
                sites = resolve_sites(spec, self.sim.map)
                if not sites:
                    LOG.warning("Scenario %s has no sites on %s: only the inject keys (T / F-keys) "
                                "start it", spec["id"], self.sim.map.name)
                for site in sites:
                    self.instances.append(ScenarioInstance(spec, site=site))
            elif trigger["type"] in ("route_ahead", "event"):
                self.instances.append(ScenarioInstance(spec))
        if not self._specs:
            LOG.warning("No scenarios loaded from %s", directory)

    def scenario_ids(self):
        return [spec["id"] for spec in self._specs]

    def owner_of(self, actor_id):
        instance = self._owner.get(actor_id)
        return instance.id if instance is not None else None

    def instance_for(self, target):
        """The route/event instance currently assigned to robotaxi ``target`` (sites excluded)."""
        for instance in self.instances:
            if instance.site is None and instance.target == target and instance.state in (SEARCHING, ACTIVE):
                return instance
        return None

    def site_instances(self):
        return [instance for instance in self.instances if instance.site is not None]

    def site_spots(self):
        """Locations of every site (zone start and the point just after it)."""
        spots = []
        for instance in self.site_instances():
            spots += [instance.site.location, instance.site.via_location]
        return spots

    def site_offers(self):
        """Sites a ride may be routed through: [(site, state, number of robotaxis heading there)]."""
        offers = []
        for instance in self.site_instances():
            if instance.state not in (ARMED, ACTIVE):
                continue
            claims = sum(1 for robotaxi in self.fleet.robotaxis
                         if getattr(robotaxi, "via_site", None) is instance.site)
            offers.append((instance.site, instance.state, claims))
        return offers

    # ------------------------------------------------------------ triggers
    def trigger(self, sim_time, target, scenario_id=None):
        """Inject a scenario on ``target`` now (one-shot, independent of the YAML trigger)."""
        if not self._specs:
            self.bus.notice("No scenario loaded")
            return None
        spec = next((s for s in self._specs if scenario_id in (None, s["id"])), None)
        if spec is None:
            self.bus.notice("Unknown scenario {}".format(scenario_id))
            return None
        existing = self.instance_for(target)
        if existing is not None:
            self.bus.notice("{} already has {} ({})".format(target, existing.label, existing.state))
            return None
        spec = copy.deepcopy(spec)
        spec["trigger"]["min_distance_m"] = min(float(spec["trigger"]["min_distance_m"]),
                                                self.manual_min_distance)
        spec["repeat"]["enabled"] = False
        instance = ScenarioInstance(spec, manual=True)
        instance.target = target
        instance.next_attempt = sim_time
        instance.set_state(SEARCHING, sim_time)
        self.instances.append(instance)
        self.bus.notice("{} requested on {}: searching the route".format(spec["id"], target))
        return instance

    def clear_for_vehicle(self, robotaxi, snapshot, sim_time):
        """Remove whatever blocks ``robotaxi`` (operator key C): its scenario props or,
        failing that, the obstacle it is reporting (stray props, a stalled car)."""
        cleared = []
        obstacle_id = None
        hazard = robotaxi.agent.hazard
        if hazard.kind in (HAZARD_STATIC, HAZARD_VEHICLE) and hazard.actor_id is not None:
            obstacle_id = hazard.actor_id
        elif robotaxi.request is not None:
            obstacle_id = robotaxi.request.detail.get("obstacle_id")
        # Decide ownership before any scenario is finished (finishing forgets its props).
        owner = self._owner.get(obstacle_id) if obstacle_id is not None else None
        plan = robotaxi.agent.plan()
        location = robotaxi.location()
        removed = set()
        for instance in list(self.instances):
            if instance.site is not None:
                # A site: clear it if it is what blocks the car or the next incident on its route.
                ahead = instance.state == ACTIVE and instance.site.route_distance(
                    location, plan, float(instance.spec["trigger"]["activate_within_m"])) is not None
                if instance is not owner and not ahead:
                    continue
            elif instance.target != robotaxi.display_name:
                continue
            removed.update(instance.prop_ids)
            if instance.state == ACTIVE:
                self._finish(instance, sim_time, "operator")
                cleared.append(instance.label)
            elif instance.state == SEARCHING:
                self._cancel_search(instance, sim_time)
                cleared.append(instance.label + " (search)")

        if robotaxi.cancel_event(sim_time):
            cleared.append("event request")
        if obstacle_id is not None and owner is None and obstacle_id not in removed:
            fleet_ids = self.fleet.vehicle_ids()
            if obstacle_id in snapshot.vehicles_by_id and obstacle_id not in fleet_ids:
                destroy_actors(self.sim.client, [obstacle_id])
                self.traffic.forget(obstacle_id)
                cleared.append("towed {}".format(snapshot.vehicles_by_id[obstacle_id].type_id))
            else:
                cluster = obstacle_cluster(snapshot, obstacle_id, fleet_ids)
                props = [actor_id for actor_id in cluster
                         if actor_id not in snapshot.vehicles_by_id and self.owner_of(actor_id) is None]
                if props:
                    destroy_actors(self.sim.client, props)
                    cleared.append("{} stray prop(s)".format(len(props)))
        if cleared:
            self.bus.log_event("operator_clear", {"vehicle": robotaxi.display_name, "cleared": cleared,
                                                  "sim_time": round(sim_time, 2)})
            self.bus.notice("{}: cleared {}".format(robotaxi.display_name, ", ".join(cleared)))
        else:
            self.bus.notice("{}: nothing to clear".format(robotaxi.display_name))
        return cleared

    def clear_all(self, sim_time, reason="operator"):
        for instance in list(self.instances):
            if instance.state == ACTIVE:
                self._finish(instance, sim_time, reason)
            elif instance.state == SEARCHING:
                self._cancel_search(instance, sim_time)

    # ----------------------------------------------------------------- step
    def step(self, snapshot, sim_time):
        for instance in list(self.instances):
            if instance.state == ARMED:
                if instance.site is not None:
                    self._step_site_armed(instance, snapshot, sim_time)
                else:
                    self._step_armed(instance, sim_time)
            elif instance.state == SEARCHING:
                self._step_searching(instance, snapshot, sim_time)
            elif instance.state == ACTIVE:
                if instance.site is not None:
                    self._step_site_active(instance, snapshot, sim_time)
                else:
                    self._step_active(instance, snapshot, sim_time)
            elif instance.state == COOLDOWN:
                self._step_cooldown(instance, sim_time)
        # Finished one-shot instances are dropped.
        self.instances = [i for i in self.instances if not (i.manual and i.state == DONE)]

    def _arm_delay(self, instance):
        """Seconds from arming to triggering: the start delay, or a random interval."""
        trigger = instance.spec["trigger"]
        start = float(trigger["start_after_s"]) if instance.cycles == 0 else 0.0
        interval = trigger.get("interval_s")
        if interval:
            delay = self._rng.uniform(float(interval[0]), float(interval[1]))
            return max(start, delay)
        return start

    def _step_armed(self, instance, sim_time):
        if instance.due_at is None:
            instance.due_at = instance.state_since + self._arm_delay(instance)
        if sim_time < instance.due_at:
            return
        if not instance.random_target and self.instance_for(instance.target) is not None:
            return   # the operator already injected something on this car
        instance.due_at = None
        instance.next_attempt = sim_time
        instance.set_state(SEARCHING, sim_time)

    def _cancel_search(self, instance, sim_time):
        instance.set_state(DONE if instance.manual else COOLDOWN, sim_time)

    def _step_searching(self, instance, snapshot, sim_time):
        trigger = instance.spec["trigger"]
        if sim_time < instance.next_attempt:
            return
        instance.next_attempt = sim_time + float(trigger["retry_interval_s"])

        if sim_time - instance.state_since > float(trigger["give_up_after_s"]):
            LOG.warning("Scenario %s: no placement found (%s)", instance.label, instance.last_reason)
            self.bus.notice("{} on {}: no place found ({})".format(instance.label, instance.target,
                                                                     instance.last_reason[:40]))
            instance.set_state(DONE if instance.manual else ARMED, sim_time)
            return

        if instance.random_target:
            robotaxi = self._pick_event_target(instance)
            if robotaxi is None:
                instance.last_reason = "no robotaxi with a passenger is driving"
                return
        else:
            robotaxi = self.fleet.by_display_name(instance.target)
        if robotaxi is None:
            instance.last_reason = "unknown target {}".format(instance.target)
            return
        if robotaxi.state != "CRUISING":
            instance.last_reason = "{} is {}".format(robotaxi.display_name, robotaxi.state)
            return
        if instance.is_event:
            self._start_event(instance, robotaxi, sim_time)
            return
        plan = robotaxi.agent.get_local_planner().get_plan()
        if instance.spec["placement"]["mode"] == "junction_exit":
            anchor, reason = find_junction_exit_anchor(robotaxi.vehicle, plan, instance.spec, snapshot)
        else:
            anchor, reason = find_route_anchor(robotaxi.vehicle, plan, instance.spec, snapshot)
        instance.last_reason = reason
        if anchor is None:
            return
        if self._spawn(instance, anchor):
            instance.set_state(ACTIVE, sim_time)
            location = anchor.start
            self.bus.log_event("scenario_started", {
                "scenario": instance.id, "manual": instance.manual, "target": instance.target,
                "zone_start": [round(location.x, 2), round(location.y, 2), round(location.z, 2)],
                "route_distance_m": round(anchor.route_distance, 1),
                "road_id": anchor.waypoint.road_id, "lane_id": anchor.waypoint.lane_id,
                "left_lane": anchor.left_lane, "right_lane": anchor.right_lane,
                "props": len(instance.prop_ids), "sim_time": round(sim_time, 2),
                "expected_request": instance.spec.get("expected_request"),
                "expected_actions": instance.spec.get("expected_actions"),
            })
            self.bus.notice("{} placed {:.0f} m ahead of {}".format(
                instance.label, anchor.route_distance, instance.target))

    def _pick_event_target(self, instance):
        """A random robotaxi that is driving (with a passenger, if required) and free."""
        need_passenger = bool(instance.spec["trigger"].get("require_passenger", True))
        candidates = [
            robotaxi for robotaxi in self.fleet.robotaxis
            if robotaxi.state == "CRUISING" and robotaxi.request is None
            and self.instance_for(robotaxi.display_name) is None
            and (not need_passenger or robotaxi.mission.phase == PHASE_TO_DROPOFF)
        ]
        return self._rng.choice(candidates) if candidates else None

    def _start_event(self, instance, robotaxi, sim_time):
        event = instance.spec["event"]
        notes = event.get("notes") or [event.get("note", "")]
        note = self._rng.choice(notes)
        if not robotaxi.raise_event(event["reason"], note, sim_time, event.get("speed_cap_kmh")):
            instance.last_reason = "{} cannot take the event now".format(robotaxi.display_name)
            return
        instance.target = robotaxi.display_name
        instance.request_id = robotaxi.request.request_id
        instance.set_state(ACTIVE, sim_time)
        self.bus.notice("{}: {}".format(robotaxi.display_name, note))
        self.bus.log_event("scenario_started", {
            "scenario": instance.id, "manual": instance.manual, "target": instance.target,
            "event": event["reason"], "note": note, "sim_time": round(sim_time, 2),
            "expected_request": instance.spec.get("expected_request"),
            "expected_actions": instance.spec.get("expected_actions"),
        })

    def _step_active(self, instance, snapshot, sim_time):
        cleanup = instance.spec["cleanup"]
        if instance.is_event:
            robotaxi = self.fleet.by_display_name(instance.target)
            request = robotaxi.request if robotaxi is not None else None
            if request is None or request.request_id != instance.request_id:
                self._finish(instance, sim_time, "answered")
            return
        if instance.spec["lane_guard"].get("enabled", True) and instance.guard_polygons:
            self._guard_lane(instance, snapshot, sim_time)

        robotaxi = self.fleet.by_display_name(instance.target)
        if robotaxi is not None:
            location = robotaxi.location()
            ahead = geometry.longitudinal_offset(location, instance.zone_end,
                                                 instance.anchor.forward)
            if instance.passed_since is None and ahead > 5.0 and \
                    geometry.planar_distance(location, instance.zone_end) < 40.0:
                instance.passed_since = sim_time
            if instance.passed_since is not None and \
                    sim_time - instance.passed_since >= float(cleanup["after_passed_s"]):
                self._finish(instance, sim_time, "passed")
                return
            far = geometry.planar_distance(location, instance.anchor.start) > 150.0
            if far:
                if instance.away_since is None:
                    instance.away_since = sim_time
                if sim_time - instance.away_since > 10.0:
                    self._finish(instance, sim_time, "target_left_area")
                    return
            else:
                instance.away_since = None

        if self.auto_clear_s and sim_time - instance.state_since > self.auto_clear_s:
            self._finish(instance, sim_time, "auto_clear")
        elif sim_time - instance.state_since > float(cleanup["max_active_s"]):
            self._finish(instance, sim_time, "timeout")

    # ------------------------------------------------------------- sites
    def _heading_to_site(self, instance):
        """Robotaxis whose route runs through the site (within the activation distance),
        plus any that is standing at it, dealing with the incident. {name: distance}"""
        site = instance.site
        limit = float(instance.spec["trigger"]["activate_within_m"])
        heading = {}
        for robotaxi in self.fleet.robotaxis:
            location = robotaxi.location()
            distance = site.route_distance(location, robotaxi.agent.plan(), limit)
            if distance is None and robotaxi.state in _BUSY_STATES \
                    and geometry.planar_distance(location, site.location) < 25.0:
                distance = 0.0
            if distance is not None:
                heading[robotaxi.display_name] = distance
        return heading

    def _zone_occupied(self, instance, snapshot):
        """True if any vehicle stands where the site's props would go."""
        site = instance.site
        for lane in patterns.closed_lanes(site.waypoint, instance.spec["actors"]):
            strip = self._lane_strip(lane, -3.0, site.zone_length + 3.0, lane.lane_width / 2.0)
            if strip is None:
                continue
            for vehicle in snapshot.vehicles:
                if strip.intersects(geometry.actor_footprint(vehicle)):
                    return True
        return False

    def _step_site_armed(self, instance, snapshot, sim_time):
        if sim_time < instance.next_attempt:
            return
        instance.next_attempt = sim_time + _SITE_CHECK_S
        trigger = instance.spec["trigger"]
        site = instance.site
        target, distance = None, 0.0
        if trigger["activation"] == "approach":
            closest = None
            for name, d in self._heading_to_site(instance).items():
                if d >= float(trigger["approach_min_m"]) and (closest is None or d < closest[1]):
                    closest = (name, d)
            if closest is None:
                instance.last_reason = "waiting for a robotaxi"
                return
            target, distance = closest
        if self._zone_occupied(instance, snapshot):
            instance.last_reason = "a vehicle is in the zone"
            return
        anchor = dataclasses.replace(site.anchor, route_distance=distance)
        instance.target = target
        if not self._spawn(instance, anchor):
            instance.target = None
            return
        instance.heading = set()
        instance.passed_by = []
        instance.idle_since = None
        instance.set_state(ACTIVE, sim_time)
        location = site.location
        self.bus.log_event("scenario_started", {
            "scenario": instance.id, "site": site.name, "manual": False, "target": target,
            "zone_start": [round(location.x, 2), round(location.y, 2), round(location.z, 2)],
            "route_distance_m": round(distance, 1), "road_id": site.road_id,
            "lane_id": site.waypoint.lane_id, "s": round(site.waypoint.s, 1),
            "left_lane": site.anchor.left_lane, "right_lane": site.anchor.right_lane,
            "props": len(instance.prop_ids), "sim_time": round(sim_time, 2),
            "expected_request": instance.spec.get("expected_request"),
            "expected_actions": instance.spec.get("expected_actions"),
        })
        if target:
            self.bus.notice("{}: incident ahead of {} ({:.0f} m)".format(instance.label, target, distance))

    def _step_site_active(self, instance, snapshot, sim_time):
        if instance.spec["lane_guard"].get("enabled", True) and instance.guard_polygons:
            self._guard_lane(instance, snapshot, sim_time)
        if self.auto_clear_s and sim_time - instance.state_since > self.auto_clear_s:
            self._finish(instance, sim_time, "auto_clear")
            return
        if sim_time < instance.next_attempt:
            return
        instance.next_attempt = sim_time + _SITE_CHECK_S
        heading = self._heading_to_site(instance)
        site = instance.site
        for name in instance.heading - set(heading):
            # This car no longer drives through the site: past it, or re-routed (turned away).
            robotaxi = self.fleet.by_display_name(name)
            past = robotaxi is not None and geometry.longitudinal_offset(
                robotaxi.location(), site.location, site.anchor.forward) > site.zone_length
            if past:
                instance.passed_by.append(name)
            self.bus.log_event("site_left", {"scenario": instance.id, "site": site.name, "vehicle": name,
                                             "passed": past, "sim_time": round(sim_time, 2)})
        instance.heading = set(heading)
        if instance.spec["trigger"]["activation"] == "always":
            return
        if heading:
            instance.idle_since = None
        elif instance.idle_since is None:
            instance.idle_since = sim_time
        elif sim_time - instance.idle_since >= float(instance.spec["cleanup"]["after_passed_s"]):
            self._finish(instance, sim_time, "passed" if instance.passed_by else "unused")
            return
        if sim_time - instance.state_since > float(instance.spec["cleanup"]["max_active_s"]):
            self._finish(instance, sim_time, "timeout")

    def _step_cooldown(self, instance, sim_time):
        repeat = instance.spec["repeat"]
        if instance.site is not None:
            # A site is a place on the map: it always comes back.
            if sim_time - instance.state_since >= float(repeat.get("rearm_after_s", 60.0)):
                instance.set_state(ARMED, sim_time)
            return
        if instance.manual or not repeat.get("enabled", False):
            instance.set_state(DONE, sim_time)
        elif sim_time - instance.state_since >= float(repeat.get("rearm_after_s", 60.0)):
            instance.set_state(ARMED, sim_time)

    # -------------------------------------------------------------- spawning
    def _spawn(self, instance, anchor):
        world = self.sim.world
        library = world.get_blueprint_library()
        placements = patterns.expand(anchor.waypoint, instance.spec["actors"], anchor.zone_length)
        spawned = []
        for placement in placements:
            blueprint = None
            for candidate in placement.blueprints:
                try:
                    blueprint = library.find(candidate)
                    break
                except IndexError:
                    continue
            if blueprint is None:
                LOG.warning("Scenario %s: none of %s exist", instance.label, placement.blueprints)
                continue
            if placement.kind == "vehicle" and blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "scenario")
            actor = None
            ladder = _VEHICLE_Z_LADDER if placement.kind == "vehicle" else _SPAWN_Z_LADDER
            for dz in ladder:
                transform = carla.Transform(placement.transform.location + carla.Location(z=dz),
                                            placement.transform.rotation)
                actor = world.try_spawn_actor(blueprint, transform)
                if actor is not None:
                    break
            if actor is None:
                LOG.debug("Scenario %s: could not spawn %s", instance.label, blueprint.id)
                continue
            if placement.kind == "vehicle":
                # A stalled vehicle: physics on (it settles on the road), hand brake, hazards.
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                if placement.hazard_lights:
                    actor.set_light_state(HAZARD_LIGHTS)
            else:
                actor.set_simulate_physics(False)
            spawned.append(actor)

        if not spawned:
            LOG.warning("Scenario %s: no prop could be spawned", instance.label)
            return False

        instance.anchor = anchor
        instance.prop_ids = [actor.id for actor in spawned]
        for actor in spawned:
            self._owner[actor.id] = instance
        instance.zone_end = patterns.lane_transform(anchor.waypoint, anchor.zone_length, 0.0).location
        instance.zone_polygon = self._lane_strip(anchor, 0.0, anchor.zone_length, anchor.lane_width / 2.0)
        instance.request_id = None
        guard = instance.spec["lane_guard"]
        instance.guard_polygons = []
        for lane in patterns.closed_lanes(anchor.waypoint, instance.spec["actors"]):
            strip = self._lane_strip(lane, -float(guard["upstream_m"]), anchor.zone_length,
                                     lane.lane_width / 2.0 - 0.2)
            if strip is not None:
                instance.guard_polygons.append(strip)
        instance.passed_since = None
        instance.away_since = None
        instance.nudged = {}
        LOG.info("Scenario %s: spawned %d props %.0f m ahead of %s (road %d, lane %d, s %.1f)",
                 instance.label, len(spawned), anchor.route_distance, instance.target or "-",
                 anchor.waypoint.road_id, anchor.waypoint.lane_id, anchor.waypoint.s)
        return True

    @staticmethod
    def _lane_strip(anchor, s_from, s_to, half_width):
        waypoint = getattr(anchor, "waypoint", anchor)
        points = []
        s = s_from
        while s <= s_to + 0.01:
            try:
                points.append(geometry.xy(patterns.lane_transform(waypoint, s, 0.0).location))
            except IndexError:
                pass   # no lane that far upstream (e.g. the road starts at a junction)
            s += 2.0
        line = geometry.polyline(points)
        return geometry.corridor(line, half_width) if line is not None else None

    def _finish(self, instance, sim_time, reason):
        destroy_actors(self.sim.client, instance.prop_ids)
        for prop_id in instance.prop_ids:
            self._owner.pop(prop_id, None)
        event = {"scenario": instance.id, "manual": instance.manual, "target": instance.target,
                 "reason": reason, "sim_time": round(sim_time, 2)}
        if instance.site is not None:
            event.update(site=instance.site.name, passed_by=list(instance.passed_by))
            self.bus.notice("{} cleared ({})".format(instance.label, reason))
        else:
            self.bus.notice("{} on {} cleared ({})".format(instance.label, instance.target, reason))
        self.bus.log_event("scenario_finished", event)
        instance.prop_ids = []
        instance.nudged = {}
        instance.cycles += 1
        instance.due_at = None
        if instance.site is not None:
            instance.target = None
            instance.heading = set()
        elif not instance.manual:
            instance.target = instance.spec.get("target_vehicle", "RT-01")
        instance.set_state(COOLDOWN, sim_time)

    # ------------------------------------------------------------ lane guard
    def _guard_lane(self, instance, snapshot, sim_time):
        """Keep Traffic-Manager cars (which cannot see props) out of the closed lane(s).

        Cars upstream in a closed lane are asked to change lanes; a car that is
        still in it right in front of the closure is removed, so it can never
        end up stuck between a robotaxi and the props.
        """
        guard = instance.spec["lane_guard"]
        remove_within = float(guard.get("remove_within_m", 8.0))
        fleet_ids = self.fleet.vehicle_ids()
        background = set(self.traffic.vehicle_ids)
        open_side = self._open_side(instance)
        can_nudge = open_side is not None and \
            (instance.anchor.left_lane if open_side == "left" else instance.anchor.right_lane) == "driving_same"
        start = instance.anchor.start
        forward = instance.anchor.forward

        for vehicle in snapshot.vehicles:
            if vehicle.id in fleet_ids or vehicle.id not in background:
                continue
            centre = geometry.actor_footprint(vehicle).centroid
            if not any(polygon.contains(centre) for polygon in instance.guard_polygons):
                continue
            if can_nudge and sim_time - instance.nudged.get(vehicle.id, -100.0) > 2.0:
                self.traffic.nudge(vehicle, open_side == "left")
                instance.nudged[vehicle.id] = sim_time
            upstream = -geometry.longitudinal_offset(vehicle.get_location(), start, forward)
            if upstream < remove_within:
                destroy_actors(self.sim.client, [vehicle.id])
                self.traffic.forget(vehicle.id)
                self.bus.log_event("guard_removed", {"scenario": instance.id, "vehicle": vehicle.type_id,
                                                     "sim_time": round(sim_time, 2)})
                LOG.info("Lane guard removed %s from the closed lane of %s", vehicle.type_id,
                         instance.label)

    @staticmethod
    def _open_side(instance):
        """Side background traffic should move to, or None (whole road closed)."""
        configured = instance.spec["lane_guard"].get("open_side")
        if configured:
            return configured
        for actor in instance.spec["actors"]:
            pattern = actor.get("pattern")
            if pattern == "lane_block":
                return actor.get("open_side", "left")
            if pattern == "road_block":
                return None
        return "left"

    # ------------------------------------------------------------ debug/HUD
    def draw_debug(self, drawer):
        for instance in self.instances:
            if instance.site is not None and instance.state != ACTIVE and drawer.full:
                drawer.text(instance.site.location, "{} {}".format(instance.label, instance.state.lower()),
                            color="zone", z=3.0)
            if instance.state != ACTIVE or instance.zone_polygon is None:
                continue
            z = instance.anchor.start.z + 0.3
            drawer.polygon(instance.zone_polygon, z, color="zone", thickness=0.05)
            drawer.text(instance.anchor.start, "{} ({})".format(instance.label, instance.target or "-"),
                        color="zone", z=3.0)

    def status_lines(self, sim_time):
        lines = []
        armed_sites = 0
        for instance in self.instances:
            if instance.state == DONE:
                continue
            if instance.site is not None and instance.state == ARMED:
                armed_sites += 1
                continue
            if instance.site is not None:
                who = ",".join(sorted(instance.heading)) or "-"
            else:
                who = instance.target or "-"
            text = "{:<17} {:<9} {}".format(instance.label[:17], instance.state, who)
            if instance.state == SEARCHING and instance.last_reason:
                text += "  ({})".format(instance.last_reason[:34])
            elif instance.state == ACTIVE:
                text += "  {:.0f}s".format(sim_time - instance.state_since)
            elif instance.state == ARMED and instance.due_at is not None:
                text += "  in {:.0f}s".format(max(0.0, instance.due_at - sim_time))
            lines.append(text)
        if armed_sites:
            lines.append("{} more site(s) waiting for a robotaxi".format(armed_sites))
        return lines

    # -------------------------------------------------------------- cleanup
    def destroy(self):
        ids = []
        for instance in self.instances:
            ids += instance.prop_ids
            instance.prop_ids = []
        destroy_actors(self.sim.client, ids)
