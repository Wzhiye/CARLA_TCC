"""Spawns the robotaxi fleet and steps it every tick."""

import logging
import random

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

from ra_demo.av.robotaxi_agent import RobotaxiAgent
from ra_demo.core import geometry
from ra_demo.core.config import get
from ra_demo.core.world import destroy_actors
from ra_demo.fleet.mission import RideMission
from ra_demo.fleet.robotaxi import NORMAL_LIGHTS, Robotaxi

LOG = logging.getLogger(__name__)


def spread_spawn_indices(spawn_points, count, rng, min_gap=40.0):
    """Pick ``count`` spawn points far apart from each other (farthest-point sampling)."""
    if not spawn_points:
        return []
    chosen = [rng.randrange(len(spawn_points))]
    while len(chosen) < min(count, len(spawn_points)):
        best_index, best_gap = None, -1.0
        for index, point in enumerate(spawn_points):
            if index in chosen:
                continue
            gap = min(geometry.planar_distance(point.location, spawn_points[c].location)
                      for c in chosen)
            if gap > best_gap:
                best_index, best_gap = index, gap
        if best_gap < min_gap:
            LOG.warning("Robotaxi spawn points are only %.1f m apart", best_gap)
        chosen.append(best_index)
    return chosen


class FleetManager:
    """Owns the robotaxi vehicles, their agents and their collision sensors."""

    def __init__(self, sim, cfg, bus, obstacle_owner=None):
        self.sim = sim
        self.cfg = cfg
        self.bus = bus
        self._obstacle_owner = obstacle_owner
        self.robotaxis = []
        self.spawn_indices = []
        self._sensors = []
        self.route_planner = None
        # Callables set once the scenarios are loaded: [(site, state, claims)] for trips
        # routed through a site, and [Location] of every site.
        self.site_hint = None
        self.site_spots = None

    # ---------------------------------------------------------------- spawn
    def spawn(self):
        fleet_cfg = self.cfg.get("fleet", {})
        count = int(fleet_cfg.get("count", 3))
        world = self.sim.world
        library = world.get_blueprint_library()
        blueprint_id = fleet_cfg.get("blueprint", "vehicle.lincoln.mkz_2020")
        try:
            blueprint = library.find(blueprint_id)
        except IndexError:
            LOG.warning("Blueprint %s not found, using vehicle.tesla.model3", blueprint_id)
            blueprint = library.find("vehicle.tesla.model3")
        color = fleet_cfg.get("color")
        if color and blueprint.has_attribute("color"):
            blueprint.set_attribute("color", str(color))

        spawn_points = self.sim.spawn_points()
        rng = random.Random(get(self.cfg, "world.seed", 0))
        wanted = list(fleet_cfg.get("spawn_points") or [])
        if len(wanted) < count:
            extra = [i for i in spread_spawn_indices(spawn_points, count + 5, rng) if i not in wanted]
            wanted += extra

        vehicles = []
        for index in wanted:
            if len(vehicles) == count:
                break
            if index >= len(spawn_points):
                LOG.warning("Spawn point %d does not exist on this map", index)
                continue
            blueprint.set_attribute("role_name", "robotaxi_{:02d}".format(len(vehicles) + 1))
            vehicle = world.try_spawn_actor(blueprint, spawn_points[index])
            if vehicle is None:
                LOG.warning("Spawn point %d is occupied, trying the next one", index)
                continue
            vehicles.append(vehicle)
            self.spawn_indices.append(index)
        if len(vehicles) < count:
            LOG.warning("Only %d of %d robotaxis could be spawned", len(vehicles), count)

        # In synchronous mode the new actors only get valid transforms after a tick.
        self.sim.tick()

        sampling = float(get(self.cfg, "fleet.driving.sampling_resolution_m", 2.0))
        self.route_planner = GlobalRoutePlanner(self.sim.map, sampling)
        collision_bp = library.find("sensor.other.collision")
        mission_cfg = fleet_cfg.get("mission", {})
        fixed_stops = mission_cfg.get("fixed_stops") or {}
        base_seed = int(get(self.cfg, "world.seed", 0) or 0)

        for number, vehicle in enumerate(vehicles, start=1):
            display_name = "RT-{:02d}".format(number)
            agent = RobotaxiAgent(vehicle, fleet_cfg, self.sim.fixed_delta,
                                  self.sim.map, self.route_planner)
            mission = RideMission(spawn_points, mission_cfg, seed=base_seed * 100 + number,
                                  fixed_stops=fixed_stops.get(display_name))
            robotaxi = Robotaxi(vehicle.attributes.get("role_name"), display_name, vehicle,
                                agent, mission, fleet_cfg, self.bus, self.sim.map,
                                obstacle_owner=self._obstacle_owner)
            sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
            sensor.listen(robotaxi.on_collision)
            self._sensors.append(sensor)
            vehicle.set_light_state(NORMAL_LIGHTS)
            robotaxi.site_hint = self._site_offers
            robotaxi.site_spots = self._site_spots
            self.robotaxis.append(robotaxi)
            LOG.info("Spawned %s (%s) at spawn point %d", display_name, vehicle.type_id,
                     self.spawn_indices[number - 1])
        # Every robotaxi knows the others, e.g. to recognise a queue behind a colleague.
        for robotaxi in self.robotaxis:
            robotaxi.fleet_names = {other.vehicle.id: other.display_name
                                    for other in self.robotaxis if other is not robotaxi}
        self.sim.tick()
        return self.robotaxis

    def _site_offers(self):
        return self.site_hint() if self.site_hint is not None else []

    def _site_spots(self):
        return self.site_spots() if self.site_spots is not None else []

    def spawn_transforms(self):
        points = self.sim.spawn_points()
        return [points[i] for i in self.spawn_indices]

    # ----------------------------------------------------------------- step
    def step(self, snapshot, sim_time):
        batch = []
        for robotaxi in self.robotaxis:
            control = robotaxi.step(snapshot, sim_time, self.sim.fixed_delta)
            batch.append(carla.command.ApplyVehicleControl(robotaxi.vehicle.id, control))
        if batch:
            self.sim.client.apply_batch(batch)

    def by_display_name(self, display_name):
        for robotaxi in self.robotaxis:
            if robotaxi.display_name == display_name:
                return robotaxi
        return None

    def vehicle_ids(self):
        return {robotaxi.vehicle.id for robotaxi in self.robotaxis}

    # ---------------------------------------------------------- debug/stats
    def draw_debug(self, drawer):
        for robotaxi in self.robotaxis:
            z = robotaxi.location().z
            if robotaxi.maneuver is not None:
                drawer.path(robotaxi.maneuver.debug_points(), z)
            else:
                drawer.route(robotaxi.agent.get_local_planner().get_plan())
            perception = robotaxi.agent.last_perception
            if perception is not None and perception.corridor is not None:
                drawer.polygon(perception.corridor, z + 0.2, color="corridor")
            label = "{} {}".format(robotaxi.display_name, robotaxi.state)
            drawer.text(robotaxi.location(), label,
                        color="request" if robotaxi.request else "text")

    def summary(self):
        lines = []
        for robotaxi in self.robotaxis:
            lines.append("{}: trips {}, RA requests {}, maneuvers {}, blocked {:.0f}s, collisions {}".format(
                robotaxi.display_name, robotaxi.mission.trips_completed, robotaxi.requests_opened,
                robotaxi.maneuvers_done, robotaxi.blocked_time, robotaxi.collisions))
        return lines

    # -------------------------------------------------------------- cleanup
    def destroy(self):
        for sensor in self._sensors:
            try:
                if sensor.is_listening():  # a method in 0.9.16
                    sensor.stop()
            except RuntimeError:
                pass
        destroy_actors(self.sim.client, [s.id for s in self._sensors])
        destroy_actors(self.sim.client, [r.vehicle.id for r in self.robotaxis])
        self._sensors = []
        self.robotaxis = []
