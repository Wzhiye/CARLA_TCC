"""Entry point: wires the world, fleet, traffic, scenarios and viewer together.

Run from the myproject folder while the CARLA server is running:

    .venv\\Scripts\\python -m ra_demo
    .venv\\Scripts\\python -m ra_demo --no-viewer --duration 1800   # headless soak test

The main loop is the only place that ticks the world (synchronous mode):

    snapshot -> fleet.step (commands, maneuvers) -> scenarios.step -> debug draw
             -> tick -> viewer (operator input)
"""

import argparse
import logging
import sys

from ra_demo.core.bus import RABus
from ra_demo.core.config import get, load_config, resolve_path
from ra_demo.core.debug_draw import DebugDrawer
from ra_demo.core.snapshot import ActorSnapshot
from ra_demo.core.world import SimWorld, cleanup_leftovers
from ra_demo.fleet.fleet_manager import FleetManager
from ra_demo.scenario.manager import ScenarioManager
from ra_demo.traffic.background import BackgroundTraffic

LOG = logging.getLogger("ra_demo")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Robotaxi remote-assistance demo (CARLA 0.9.16)")
    parser.add_argument("--config", help="simulation config YAML (default: config/sim.yaml)")
    parser.add_argument("--fleet", help="fleet config YAML (default: config/fleet.yaml)")
    parser.add_argument("--host", help="CARLA host override")
    parser.add_argument("--port", type=int, help="CARLA port override")
    parser.add_argument("--map", help="map override, e.g. Town03_Opt")
    parser.add_argument("--seed", type=int, help="random seed override")
    parser.add_argument("--no-viewer", action="store_true", help="run without the pygame window")
    parser.add_argument("--no-traffic", action="store_true", help="skip background traffic")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after this many simulated seconds (0 = run until closed)")
    parser.add_argument("--auto-clear", type=float,
                        help="remove active scenario props after N seconds (unattended runs)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def build_overrides(args):
    overrides = {}
    if args.host:
        overrides.setdefault("carla", {})["host"] = args.host
    if args.port:
        overrides.setdefault("carla", {})["port"] = args.port
    if args.map:
        overrides.setdefault("world", {})["map"] = args.map
    if args.seed is not None:
        overrides.setdefault("world", {})["seed"] = args.seed
    if args.no_traffic:
        overrides.setdefault("traffic", {})["enabled"] = False
    if args.auto_clear is not None:
        overrides.setdefault("scenarios", {})["auto_clear_after_s"] = args.auto_clear
    return overrides


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.fleet, build_overrides(args))

    sim = SimWorld(cfg)
    bus = RABus(resolve_path(get(cfg, "logging.dir", "../logs")))
    fleet = traffic = scenarios = viewer = None
    try:
        sim.connect()
        if get(cfg, "world.cleanup_leftovers", True):
            cleanup_leftovers(sim.client, sim.world)
        drawer = DebugDrawer(sim.world, sim.fixed_delta, mode=str(get(cfg, "debug.mode", "subtle")))

        # Scenario ownership lookup is needed by the fleet before scenarios exist.
        owner_lookup = {"fn": lambda actor_id: None}
        fleet = FleetManager(sim, cfg, bus, obstacle_owner=lambda actor_id: owner_lookup["fn"](actor_id))
        fleet.spawn()

        traffic = BackgroundTraffic(sim, cfg)
        traffic.spawn(reserved_transforms=fleet.spawn_transforms())

        scenarios = ScenarioManager(sim, cfg, fleet, traffic, bus)
        scenarios.load()
        owner_lookup["fn"] = scenarios.owner_of
        fleet.site_hint = scenarios.site_offers
        fleet.site_spots = scenarios.site_spots

        if not args.no_viewer and get(cfg, "viewer.enabled", True):
            from ra_demo.console.viewer import Viewer  # pygame is only needed with a window
            viewer = Viewer(sim, fleet, scenarios, bus, drawer, cfg)
            viewer.init()

        sim.reset_pacer()
        LOG.info("Running. %s", "Close the window or press ESC to stop." if viewer else "Ctrl+C to stop.")
        running = True
        while running:
            snapshot = ActorSnapshot(sim.world, sim.frame, sim.sim_time, fleet.vehicle_ids())
            fleet.step(snapshot, sim.sim_time)
            scenarios.step(snapshot, sim.sim_time)
            if drawer.due():
                fleet.draw_debug(drawer)
                scenarios.draw_debug(drawer)
            sim.tick()
            if viewer is not None:
                running = viewer.handle_events(snapshot)
                viewer.render()
            if args.duration and sim.sim_time >= args.duration:
                LOG.info("Reached --duration %.0fs", args.duration)
                running = False
    except KeyboardInterrupt:
        LOG.info("Interrupted")
    finally:
        _shutdown(sim, bus, fleet, traffic, scenarios, viewer)


def _shutdown(sim, bus, fleet, traffic, scenarios, viewer):
    """Destroy everything we spawned (sensors first), then restore the world settings."""
    if fleet is not None:
        for line in fleet.summary():
            LOG.info("Summary  %s", line)
    for name, part in (("viewer", viewer), ("scenarios", scenarios),
                       ("traffic", traffic), ("fleet", fleet)):
        if part is None:
            continue
        try:
            part.destroy()
        except Exception as exc:  # keep cleaning up whatever fails
            LOG.warning("Cleanup of %s failed: %s", name, exc)
    try:
        if sim.world is not None:
            sim.world.tick()
    except RuntimeError:
        pass
    sim.close()
    LOG.info("RA event log: %s", bus.log_path)
    bus.close()


if __name__ == "__main__":
    main()
