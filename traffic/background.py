"""Background traffic, adapted from generate_traffic_realtime.py.

The world is already in synchronous mode (SimWorld owns the tick), so this
module only spawns actors, hands vehicles to the Traffic Manager and starts the
pedestrian controllers.
"""

import logging
import random

import carla
from carla.command import DestroyActor, FutureActor, SetAutopilot, SpawnActor

from ra_demo.core import geometry
from ra_demo.core.config import get

LOG = logging.getLogger(__name__)


class BackgroundTraffic:
    """Spawns and cleans up Traffic-Manager vehicles and AI pedestrians."""

    def __init__(self, sim, cfg):
        self.sim = sim
        self.cfg = cfg.get("traffic", {})
        self.speed_difference = float(self.cfg.get("speed_difference_pct", 20.0))
        self._rng = random.Random(get(cfg, "world.seed", 0))
        self._robotaxi_blueprint = get(cfg, "fleet.blueprint", "")
        self.vehicle_ids = []
        self.walker_ids = []
        self.controller_ids = []

    # ---------------------------------------------------------------- spawn
    def spawn(self, reserved_transforms=()):
        if not self.cfg.get("enabled", True):
            LOG.info("Background traffic disabled")
            return
        self._configure_traffic_manager()
        self._spawn_vehicles(reserved_transforms)
        self._spawn_walkers()
        LOG.info("Background traffic: %d vehicles, %d pedestrians",
                 len(self.vehicle_ids), len(self.walker_ids))

    def _configure_traffic_manager(self):
        tm = self.sim.traffic_manager
        tm.set_global_distance_to_leading_vehicle(float(self.cfg.get("distance_to_leading_m", 2.5)))
        tm.global_percentage_speed_difference(self.speed_difference)
        if self.cfg.get("hybrid_physics", False):
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(70.0)

    def _vehicle_blueprints(self):
        library = self.sim.world.get_blueprint_library()
        blueprints = list(library.filter(self.cfg.get("vehicle_filter", "vehicle.*")))
        if self.cfg.get("cars_only", True):
            blueprints = [bp for bp in blueprints
                          if bp.has_attribute("base_type") and bp.get_attribute("base_type").as_str() == "car"]
        # Keep the robotaxi model exclusive so the fleet is easy to recognise.
        blueprints = [bp for bp in blueprints if bp.id != self._robotaxi_blueprint]
        return sorted(blueprints, key=lambda bp: bp.id)

    def _spawn_vehicles(self, reserved_transforms):
        count = int(self.cfg.get("vehicles", 25))
        if count <= 0:
            return
        blueprints = self._vehicle_blueprints()
        if not blueprints:
            LOG.warning("No vehicle blueprints match the traffic filter")
            return

        keep_clear = float(self.cfg.get("spawn_clearance_m", 15.0))
        spawn_points = [
            point for point in self.sim.spawn_points()
            if all(geometry.planar_distance(point.location, reserved.location) > keep_clear
                   for reserved in reserved_transforms)
        ]
        self._rng.shuffle(spawn_points)
        if count > len(spawn_points):
            LOG.warning("Requested %d vehicles but only %d free spawn points", count, len(spawn_points))
            count = len(spawn_points)

        tm_port = self.sim.traffic_manager.get_port()
        batch = []
        for transform in spawn_points[:count]:
            blueprint = self._rng.choice(blueprints)
            if blueprint.has_attribute("color"):
                blueprint.set_attribute("color", self._rng.choice(
                    blueprint.get_attribute("color").recommended_values))
            if blueprint.has_attribute("driver_id"):
                blueprint.set_attribute("driver_id", self._rng.choice(
                    blueprint.get_attribute("driver_id").recommended_values))
            blueprint.set_attribute("role_name", "background")
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, tm_port)))

        for response in self.sim.client.apply_batch_sync(batch, True):
            if response.error:
                LOG.debug("Vehicle spawn failed: %s", response.error)
            else:
                self.vehicle_ids.append(response.actor_id)

        if self.cfg.get("car_lights_on", True):
            for actor in self.sim.world.get_actors(self.vehicle_ids):
                self.sim.traffic_manager.update_vehicle_lights(actor, True)

    def _spawn_walkers(self):
        count = int(self.cfg.get("walkers", 20))
        if count <= 0:
            return
        world = self.sim.world
        library = world.get_blueprint_library()
        walker_bps = list(library.filter(self.cfg.get("walker_filter", "walker.pedestrian.*")))
        if not walker_bps:
            return
        seed = get(self.cfg, "walker_seed")
        if seed is not None:
            world.set_pedestrians_seed(int(seed))

        # 1) Spawn the pedestrians at random navigation-mesh locations.
        batch, speeds = [], []
        for _ in range(count):
            location = world.get_random_location_from_navigation()
            if location is None:
                continue
            blueprint = self._rng.choice(walker_bps)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "background")
            if blueprint.has_attribute("speed"):
                speeds.append(float(blueprint.get_attribute("speed").recommended_values[1]))
            else:
                speeds.append(1.2)
            batch.append(SpawnActor(blueprint, carla.Transform(location)))
        walker_speeds = []
        for response, speed in zip(self.sim.client.apply_batch_sync(batch, True), speeds):
            if not response.error:
                self.walker_ids.append(response.actor_id)
                walker_speeds.append(speed)

        # 2) One AI controller per pedestrian.
        controller_bp = library.find("controller.ai.walker")
        batch = [SpawnActor(controller_bp, carla.Transform(), walker_id) for walker_id in self.walker_ids]
        paired = []
        for walker_id, speed, response in zip(self.walker_ids, walker_speeds,
                                               self.sim.client.apply_batch_sync(batch, True)):
            if response.error:
                LOG.debug("Walker controller spawn failed: %s", response.error)
                continue
            self.controller_ids.append(response.actor_id)
            paired.append((response.actor_id, speed))

        # 3) Controllers need one tick before they can be started.
        self.sim.tick()
        world.set_pedestrians_cross_factor(float(self.cfg.get("walker_cross_factor", 0.0)))
        for controller_id, speed in paired:
            controller = world.get_actor(controller_id)
            if controller is None:
                continue
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(speed)

    # -------------------------------------------------------- lane guarding
    def nudge(self, vehicle, to_left):
        """Ask the Traffic Manager to change lane (to the left when ``to_left``)."""
        # force_lane_change: True = right, False = left.
        self.sim.traffic_manager.force_lane_change(vehicle, not to_left)

    def forget(self, vehicle_id):
        """Drop a vehicle that was destroyed elsewhere (lane guard, operator tow)."""
        if vehicle_id in self.vehicle_ids:
            self.vehicle_ids.remove(vehicle_id)

    # -------------------------------------------------------------- cleanup
    def destroy(self):
        world = self.sim.world
        for controller_id in self.controller_ids:
            controller = world.get_actor(controller_id)
            if controller is not None:
                try:
                    controller.stop()
                except RuntimeError:
                    pass
        ids = self.vehicle_ids + self.controller_ids + self.walker_ids
        if ids:
            self.sim.client.apply_batch_sync([DestroyActor(actor_id) for actor_id in ids])
        LOG.info("Destroyed %d background actors", len(ids))
        self.vehicle_ids, self.walker_ids, self.controller_ids = [], [], []
