"""Simulation world wrapper.

``SimWorld`` owns the connection, the synchronous-mode settings and the
world tick. Exactly one SimWorld (one process) may tick the server; every other
client (tools, viewers in other processes) must stay read-only.
"""

import logging
import time

import carla

from ra_demo.core import compat
from ra_demo.core.config import get

LOG = logging.getLogger(__name__)


class RealtimePacer:
    """Sleeps between ticks so simulated time advances at a chosen wall-clock rate.

    A fixed simulation step does not limit how fast the server runs, so without
    pacing a light scene can run several times faster than real time.
    ``factor`` 1.0 means real time, 0.5 half speed, 0 disables pacing.
    """

    def __init__(self, fixed_delta, factor):
        self._wall_step = fixed_delta / factor if factor and factor > 0 else None
        self._next = time.perf_counter()

    def reset(self):
        self._next = time.perf_counter()

    def wait(self):
        if self._wall_step is None:
            return
        self._next += self._wall_step
        remaining = self._next - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            # The simulator is slower than requested: do not build up a backlog.
            self._next = time.perf_counter()


class SimWorld:
    """Connection, synchronous settings, ticking and restoring the world."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None
        self.fixed_delta = float(get(cfg, "world.fixed_delta_seconds", 0.05))
        self.sim_time = 0.0
        self.frame = 0
        self._start_sim_time = None
        self._original_settings = None
        self._pacer = RealtimePacer(self.fixed_delta, float(get(cfg, "world.realtime_factor", 1.0)))

    # ------------------------------------------------------------------ setup
    def connect(self):
        host = get(self.cfg, "carla.host", "127.0.0.1")
        port = int(get(self.cfg, "carla.port", 2000))
        self.client = carla.Client(host, port)
        self.client.set_timeout(float(get(self.cfg, "carla.timeout", 20.0)))

        LOG.info("CARLA client %s, server %s",
                 compat.client_version(self.client), compat.server_version(self.client))

        world = self.client.get_world()
        wanted_map = get(self.cfg, "world.map")
        if wanted_map and not compat.same_map(world.get_map().name, wanted_map):
            LOG.info("Loading map %s (current: %s) ...", wanted_map,
                     compat.short_map_name(world.get_map().name))
            # Loading a map can take a while on the first run.
            self.client.set_timeout(120.0)
            world = self.client.load_world(wanted_map)
            self.client.set_timeout(float(get(self.cfg, "carla.timeout", 20.0)))
        self.world = world

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta
        settings.no_rendering_mode = bool(get(self.cfg, "world.no_rendering", False))

        tm_port = int(get(self.cfg, "carla.tm_port", 8000))
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        seed = get(self.cfg, "world.seed")
        if seed is not None:
            self.traffic_manager.set_random_device_seed(int(seed))

        self.world.apply_settings(settings)
        self.map = self.world.get_map()
        self.tick()
        LOG.info("World ready: %s, synchronous %.3fs step, realtime factor %s",
                 compat.short_map_name(self.map.name), self.fixed_delta,
                 get(self.cfg, "world.realtime_factor", 1.0))

    # ------------------------------------------------------------------- tick
    def tick(self):
        """Advance the world by one fixed step and pace it to wall-clock time."""
        self.frame = self.world.tick()
        self._pacer.wait()
        elapsed = self.world.get_snapshot().timestamp.elapsed_seconds
        if self._start_sim_time is None:
            self._start_sim_time = elapsed
        self.sim_time = elapsed - self._start_sim_time
        return self.frame

    def reset_pacer(self):
        """Call after long blocking work (spawning) so pacing does not try to catch up."""
        self._pacer.reset()

    # ---------------------------------------------------------------- helpers
    def spawn_points(self):
        return self.map.get_spawn_points()

    def blueprint(self, blueprint_id):
        return self.world.get_blueprint_library().find(blueprint_id)

    # ---------------------------------------------------------------- cleanup
    def close(self):
        """Restore the settings the world had before this run."""
        if self.world is None:
            return
        try:
            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(False)
            if self._original_settings is not None:
                restore = self._original_settings
                # Never leave the server in synchronous mode without a ticking client.
                restore.synchronous_mode = False
                self.world.apply_settings(restore)
            LOG.info("World settings restored")
        except RuntimeError as exc:
            LOG.warning("Could not restore world settings: %s", exc)


LEFTOVER_ROLES = ("background", "autopilot")


def cleanup_leftovers(client, world):
    """Destroy actors left behind by an earlier run that crashed before cleaning up.

    Matches static props, robotaxis (role_name robotaxi_*), background traffic
    (role_name background/autopilot), their sensors and walker controllers.
    """
    actors = list(world.get_actors())
    doomed = set()
    for actor in actors:
        role = actor.attributes.get("role_name", "")
        if actor.type_id.startswith("static.prop."):
            doomed.add(actor.id)
        elif actor.type_id.startswith(("vehicle.", "walker.")) and (
                role.startswith("robotaxi_") or role in LEFTOVER_ROLES):
            doomed.add(actor.id)
    for actor in actors:
        parent = actor.parent
        if parent is not None and parent.id in doomed and \
                actor.type_id.startswith(("sensor.", "controller.")):
            if actor.type_id.startswith("controller.ai.walker"):
                try:
                    actor.stop()
                except RuntimeError:
                    pass
            doomed.add(actor.id)
    if doomed:
        destroy_actors(client, list(doomed))
        world.tick()
        LOG.info("Removed %d actors left over from a previous run", len(doomed))
    return len(doomed)


def destroy_actors(client, actor_ids):
    """Destroy actors by id in one batch, ignoring ones that are already gone."""
    ids = [actor_id for actor_id in actor_ids if actor_id is not None]
    if not ids:
        return
    try:
        client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in ids])
    except RuntimeError as exc:
        LOG.warning("Batch destroy failed: %s", exc)
